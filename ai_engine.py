import httpx
import json
import time
import os
import sys
import asyncio
from typing import Optional, Dict, Any, Tuple
from notification_system import NotificationSystem
from request_repository import RequestRepository

class AIEngine:
    def __init__(self, quota_manager: Any, base_dir: str = "."):
        self.base_dir = base_dir
        self.quota_manager = quota_manager
        self.model_id_mapping = {}
        self.notification_system = NotificationSystem(base_dir)
        self.request_repo = RequestRepository(os.path.join(base_dir, "requests.db"))
        # We need to initialize the repo async, but __init__ is sync. 
        # We'll do lazy init or init on first request.
        self._repo_initialized = False

    async def _ensure_repo(self):
        if not self._repo_initialized:
            await self.request_repo.initialize()
            self._repo_initialized = True

    async def generate_content(self, model_name: Optional[str], text: str, images: list = None, response_format: str = "text", tools: list = None) -> Any:
        """
        Generate content using available AI models with fallback support.
        This is the main entry point for AI generation.
        """
        try:
            # Set a global timeout for the entire generation process (e.g., 2 minutes)
            return await asyncio.wait_for(self._generate_content_internal(model_name, text, images, response_format, tools), timeout=120.0)
        except asyncio.TimeoutError:
            print(f"❌ Generation Timed Out after 120s")
            return "Error: Request timed out. The AI took too long to respond."
        except Exception as e:
            print(f"❌ Unexpected Error in generate_content: {e}")
            return f"Error: {str(e)}"

    async def _generate_content_internal(self, model_name: Optional[str], text: str, images: list = None, response_format: str = "text", tools: list = None) -> Any:
        await self._ensure_repo()
        
        # 0. Handle Empty Prompt -> Show Cache/History
        if not text and not images and not tools:
            recent = await self.request_repo.get_recent_requests(limit=5)
            if not recent:
                return "ℹ️ Cache is empty. Send a message to start!"
            
            response = "## 🕒 Recent Cached Requests\n\n"
            for r in recent:
                short_input = r['input'][:50] + "..." if len(r['input']) > 50 else r['input']
                t_str = time.strftime('%H:%M:%S', time.localtime(r['timestamp']))
                response += f"- **{t_str}** [{r['model']}]: `{short_input}`\n"
            
            response += "\n*Tip: This list is shown because you sent an empty prompt."
            return response
            
        # 0.1 Handle Empty Prompt with Images -> Try to match ANY previous request with same images
        if not text and images:
            img_hash = self.request_repo.compute_image_hash(images)
            if img_hash:
                cached_pair = await self.request_repo.get_cached_response_by_image_hash(img_hash)
                if cached_pair:
                    cached_resp, original_prompt = cached_pair
                    print(f"✨ Image Cache Hit! Returning stored response for image hash {img_hash[:8]}...")
                    return f"**[Cached Result from prompt: '{original_prompt}']**\n\n{cached_resp}"
        
        # 1. Check for 100% Match Cache
        # Only use cache if response_format is text (default), to avoid caching non-json when json was requested or vice versa
        # Or include format in hash? For now, skip cache for json mode to be safe/fresh.
        if response_format == "text":
            input_hash = self.request_repo.compute_hash(text, images)
            cached_response = await self.request_repo.get_cached_response(input_hash)
            if cached_response:
                print(f"✨ Cache Hit! Returning stored response for hash {input_hash[:8]}...")
                return cached_response

        # 2. Check for Template Match (Minimize Tokens)
        # Only if no images (templates usually for text)
        final_prompt = text
        if not images and response_format == "text":
            # find_matching_template is synchronous CPU bound (regex)
            # Should be fast enough, but if regex is complex, it might block. 
            # Let's wrap it just in case or assume it's fine. 
            # It's just regex matching, should be microsecond scale.
            optimized_prompt, debug_info = self.request_repo.find_matching_template(text)
            if optimized_prompt:
                print(f"🧩 Template Matched! optimizing prompt: {len(text)} -> {len(optimized_prompt)} chars")
                final_prompt = optimized_prompt

        attempted_models = set()
        current_model_name = model_name
        
        max_attempts = 5
        attempts = 0
        
        while attempts < max_attempts:
            # Get a model to try
            actual_model_name, api_config = await self.quota_manager.get_model_for_request(current_model_name)
            sys.stderr.write(f"DEBUG: get_model_for_request returned {actual_model_name}\n")
            sys.stderr.flush()
            
            if not actual_model_name:
                return "Error: No available models found (all quotas exhausted or no models loaded)."
                
            config_file = api_config.get("config_file")
            attempt_key = f"{actual_model_name}|{config_file}"
            
            if attempt_key in attempted_models:
                # We've already tried this specific combo in this request cycle
                current_model_name = None
                attempts += 1  # Prevent infinite loop
                continue

            attempted_models.add(attempt_key)
            attempts += 1
            
            api_endpoint = api_config.get("api_endpoint")
            api_key = api_config.get("api_key")
            
            if not api_endpoint or not api_key:
                current_model_name = None
                continue

            provider = api_config.get("provider", "google")
            model_id = await self._resolve_model_id(actual_model_name, api_endpoint, api_key)
            params = api_config.get("params", {})
            
            try:
                print(f"Trying model: {actual_model_name} ({model_id}) via {provider}...")
                
                # Use final_prompt (which might be optimized)
                if provider == "google":
                    result = await self._call_google_api(api_endpoint, api_key, model_id, final_prompt, images, response_format, tools)
                elif provider == "openai":
                    result = await self._call_openai_api(api_endpoint, api_key, model_id, final_prompt, params, images, response_format)
                else:
                    print(f"Unknown provider {provider} for model {actual_model_name}")
                    current_model_name = None
                    continue
                    
                # Update Quota
                total_tokens = result.get("usage", 0)
                response_text = result.get("text", "")
                tool_calls = result.get("tool_calls", None)
                
                await self.quota_manager.update_quota(actual_model_name, tokens_used=total_tokens, request_count=1, config_file=config_file)
                
                # 3. Save to Cache (Map ORIGINAL hash to result)
                # Only cache text responses for now
                if response_format == "text" and not tool_calls:
                    input_hash = self.request_repo.compute_hash(text, images)
                    image_hash = self.request_repo.compute_image_hash(images) if images else None
                    await self.request_repo.save_request(input_hash, text, response_text, actual_model_name, image_hash)

                if tool_calls:
                    return {"text": response_text, "tool_calls": tool_calls}
                return response_text

            except Exception as e:
                error_msg = str(e)
                print(f"Exception with {actual_model_name}: {error_msg}")
                
                # Check for blocking errors
                if error_msg.startswith("HTTP_ERROR:"):
                    try:
                        parts = error_msg.split(":", 2)
                        status_code = int(parts[1])
                        reason = parts[2] if len(parts) > 2 else "Unknown"
                        
                        if status_code in [401, 403]:
                            msg = f"Model {actual_model_name} BLOCKED due to Auth Error ({status_code}). Reason: {reason}"
                            await self.quota_manager.mark_provider_blocked(actual_model_name, msg, config_file=config_file)
                            self.notification_system.notify(msg, "BLOCKED")
                        
                        elif status_code == 400 and ("API_KEY_INVALID" in reason or "API key not valid" in reason):
                            msg = f"Model {actual_model_name} BLOCKED due to Invalid API Key ({status_code}). Reason: {reason}"
                            await self.quota_manager.mark_provider_blocked(actual_model_name, msg, config_file=config_file)
                            self.notification_system.notify(msg, "BLOCKED")

                        elif status_code == 429:
                            # Quota exceeded
                            msg = f"Model {actual_model_name} BLOCKED due to Quota Exceeded ({status_code}). Reason: {reason}"
                            await self.quota_manager.mark_provider_blocked(actual_model_name, msg, config_file=config_file)
                            self.notification_system.notify(msg, "BLOCKED")
                            
                    except Exception as parse_err:
                        print(f"Error parsing exception: {parse_err}")

                current_model_name = None # Clear preference
                
        return "Error: Failed to generate content after multiple attempts."

    async def _resolve_model_id(self, model_name: str, api_endpoint: str, api_key: str) -> str:
        if not self.model_id_mapping:
            await self._fetch_model_mapping(api_endpoint, api_key)
            
        lower_name = model_name.lower()
        if lower_name in self.model_id_mapping:
            return self.model_id_mapping[lower_name]
            
        if model_name.lower().startswith("gemini-"):
            return model_name
            
        return model_name

    async def _fetch_model_mapping(self, api_endpoint: str, api_key: str):
        if not api_key or not api_endpoint:
            return

        url = api_endpoint
        if not url.endswith("models"):
            url = f"{url.rstrip('/')}/models"
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}?key={api_key}", timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    if "models" in data:
                        for m in data["models"]:
                            model_id = m['name'].replace("models/", "")
                            display_name = m.get("displayName")
                            if display_name:
                                self.model_id_mapping[display_name.lower()] = model_id
                            self.model_id_mapping[model_id.lower()] = model_id
        except Exception as e:
            print(f"Warning: Failed to fetch model mapping: {e}")

    async def _call_google_api(self, api_endpoint: str, api_key: str, model_id: str, text: str, images: list = None, response_format: str = "text", tools: list = None) -> dict:
        base_url = api_endpoint
        if not base_url.endswith("models"):
            base_url = f"{base_url.rstrip('/')}/models"
            
        url = f"{base_url}/{model_id}:generateContent?key={api_key}"
        
        headers = {"Content-Type": "application/json"}
        
        parts = [{"text": text}]
        if images:
            for img in images:
                parts.append({
                    "inline_data": {
                        "mime_type": img["mime_type"],
                        "data": img["data"]
                    }
                })

        payload = {"contents": [{"parts": parts}]}
        
        if tools:
            payload["tools"] = tools

        # Enforce JSON mode if requested (Gemini 1.5+ feature)
        if response_format == "json":
            payload["generationConfig"] = {"responseMimeType": "application/json"}
        
        # Reduced timeout to 45s to fail faster and allow retries within global limit
        sys.stderr.write(f"DEBUG: sending post to {url} with payload keys {list(payload.keys())}\n")
        sys.stderr.flush()
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=headers, json=payload)
        sys.stderr.write(f"DEBUG: response received {response.status_code}\n")
        sys.stderr.flush()
        
        if response.status_code != 200:
            raise Exception(f"HTTP_ERROR:{response.status_code}:{response.text}")
            
        data = response.json()
        
        usage = data.get("usageMetadata", {})
        total_tokens = usage.get("totalTokenCount", 0)
        
        response_text = ""
        tool_calls = []
        try:
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                for part in parts:
                    if "text" in part:
                        response_text += part["text"]
                    if "functionCall" in part:
                        tool_calls.append(part["functionCall"])
        except:
            pass
            
        if total_tokens == 0:
            total_tokens = int((len(text) + len(response_text)) / 4)
            
        return {"text": response_text if response_text else json.dumps(data, indent=2), "usage": total_tokens, "tool_calls": tool_calls}

    async def _call_openai_api(self, api_endpoint: str, api_key: str, model_id: str, text: str, params: dict = None, images: list = None, response_format: str = "text") -> dict:
        url = f"{api_endpoint.rstrip('/')}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        content_payload = text
        if images:
            content_payload = [{"type": "text", "text": text}]
            for img in images:
                content_payload.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img['mime_type']};base64,{img['data']}"
                    }
                })
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": content_payload}]
        }
        
        # Merge extra params (e.g. for Nvidia thinking)
        if params:
            payload.update(params)
            
        # Enforce JSON mode if requested (OpenAI feature)
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        
        # print(f"DEBUG: sending payload to {url}: {json.dumps(payload)}") # Don't print huge payloads
        
        # Reduced timeout to 45s
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
             raise Exception(f"HTTP_ERROR:{response.status_code}:{response.text}")
             
        data = response.json()
        
        usage = data.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)
        
        response_text = ""
        try:
            if "choices" in data and len(data["choices"]) > 0:
                response_text = data["choices"][0]["message"]["content"]
        except:
            pass
            
        return {"text": response_text if response_text else json.dumps(data, indent=2), "usage": total_tokens}

    async def generate_patch(self, prompt: str, model_name: Optional[str] = None, progress_callback: Optional[Any] = None) -> Dict[str, Any]:
        """
        Generates a patch based on the prompt and project context.
        """
        if progress_callback:
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback("📂 Reading project structure...")
            else:
                progress_callback("📂 Reading project structure...")

        # We can't easily pass async callback to sync running in thread, 
        # so we'll just log before/after or refactor _get_project_context to be async later. 
        # For now, let's keep it simple and just log the major steps we control here.
        
        context = await asyncio.to_thread(self._get_project_context)
        
        if progress_callback:
            msg = f"✅ Context loaded ({len(context)} chars). Constructing prompt..."
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(msg)
            else:
                progress_callback(msg)
        
        system_prompt = (
            "You are an expert software engineer acting as an autonomous agent (like Trae AI).\n"
            "Your task is to modify the codebase according to the user's request.\n"
            "You MUST return a VALID JSON object with the following structure:\n"
            "{\n"
            "  \"thought\": \"Detailed explanation of your reasoning, analysis of the files, and architectural decisions.\",\n"
            "  \"plan\": [\"Step 1: ...\", \"Step 2: ...\"],\n"
            "  \"shell_commands\": [\"echo 'example' > file.txt\"], // Equivalent shell commands for the user to see what's happening\n"
            "  \"patches\": [\n"
            "    {\n"
            "      \"file\": \"relative/path/to/file.py\",\n"
            "      \"action\": \"create\" | \"replace\" | \"delete\",\n"
            "      \"content\": \"FULL NEW CONTENT HERE\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "\n"
            "For 'replace', provide the FULL content of the file. Do not use diffs.\n"
            "Ensure you understand the context before generating code.\n"
            "If the file is large, you may use 'search_replace' action with 'search' and 'replace' fields, "
            "but 'replace' (full content) is preferred for safety if file is < 500 lines.\n"
            "\n"
            "## Architecture Guidelines\n"
            "- **Components/Plugins**: This project uses a Unified Component Architecture. Both `server.py` (MCP) and `ui_server.py` (FastAPI) load components from the `components/` directory.\n"
            "- **Component Structure**: Create new features as standalone Python modules in `components/`. Each component MUST define a `setup(mcp=None, app=None)` function.\n"
            "- **Injection**: Use `if mcp: @mcp.tool()` to register tools, and `if app: @app.get(...)` to register routes.\n"
            "- **Hot Reload**: Components are hot-reloadable. Do not modify core server files unless necessary; prefer creating new components.\n"
            "\n"
            "Project Context:\n"
        )
        
        full_prompt = f"{system_prompt}\n{context}\n\nUser Request: {prompt}\n\nRespond ONLY with valid JSON."
        
        if progress_callback:
            msg = f"🚀 Sending request to AI Model ({model_name or 'Auto'})..."
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(msg)
            else:
                progress_callback(msg)

        # Call generation with a longer timeout implicit in generate_content (120s)
        # Force JSON mode
        response_text = await self.generate_content(model_name, full_prompt, response_format="json")
        
        if progress_callback:
            msg = "📥 Response received. Parsing JSON..."
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(msg)
            else:
                progress_callback(msg)

        # Parse JSON
        try:
            # Try to find JSON block
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response_text[start:end]
                result = json.loads(json_str)
                
                if progress_callback:
                    msg = "✨ Patch generation complete!"
                    if asyncio.iscoroutinefunction(progress_callback):
                        await progress_callback(msg)
                    else:
                        progress_callback(msg)
                        
                return result
            else:
                # If no JSON block is found, return the raw response with an error message
                print(f"❌ No JSON found in response. Raw response:\n{response_text}")
                return {"error": "No JSON found in response. Please ensure the AI returns valid JSON.", "raw_response": response_text}
        except json.JSONDecodeError as e:
            # Specific error for JSON decoding issues
            print(f"❌ JSON Decode Error: {e}. Raw response:\n{response_text}")
            return {"error": f"JSON Decode Error: {e}. The response was:\n{response_text}", "raw_response": response_text}
        except Exception as e:
            # Catch-all for other unexpected errors during parsing
            print(f"❌ Unexpected error parsing JSON: {e}")
            return {"error": f"An unexpected error occurred during JSON parsing: {e}", "raw_response": response_text}

    MAX_CONTEXT_CHARS = 20000

    def _get_project_context(self) -> str:
        context = []
        allowed_ext = {'.py', '.html', '.js', '.css', '.json', '.md'}
        ignored_dirs = {'.git', '__pycache__', 'node_modules', 'venv', 'env', '.idea', '.vscode', 'quotas', 'requests.db', 'key_harvester', 'trust_store'}
        
        # 1. Build File Tree first (so model knows everything that exists)
        file_tree = ["Project Structure:"]
        all_files_map = [] # (rel_path, full_path, size)
        
        for root, dirs, files in os.walk(self.base_dir):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            level = root.replace(self.base_dir, '').count(os.sep)
            indent = '  ' * level
            file_tree.append(f"{indent}{os.path.basename(root)}/")
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in allowed_ext: continue
                
                path = os.path.join(root, file)
                rel_path = os.path.relpath(path, self.base_dir)
                size = os.path.getsize(path)
                
                file_tree.append(f"{indent}  {file}")
                all_files_map.append((rel_path, path, size))
                
        context.append("\n".join(file_tree))
        context.append("\n\nFile Contents:")
        
        # 2. Add Content (Priority First)
        total_chars = len(context[0])
        priority_files = ['ui_server.py', 'quota_server.py', 'ai_engine.py', 'dashboard.html']
        
        # Helper to add file
        def add_file(rel_path, path, size, is_priority=False):
            nonlocal total_chars
            if total_chars > self.MAX_CONTEXT_CHARS: 
                return False
                
            try:
                # If file is too big, truncate it
                limit = 30000 if is_priority else 10000
                if size > limit:
                     with open(path, 'r', encoding='utf-8') as f:
                        content = f.read(limit)
                        chunk = f"\nFile: {rel_path} (First {limit} chars)\n```\n{content}\n...\n```\n"
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        chunk = f"\nFile: {rel_path}\n```\n{content}\n```\n"
                
                if total_chars + len(chunk) > self.MAX_CONTEXT_CHARS:
                    return False
                
                context.append(chunk)
                total_chars += len(chunk)
                return True
            except Exception as e:
                context.append(f"\nFile: {rel_path} (Error reading: {e})\n")
                return True

        # Add Priority Files
        processed_files = set()
        for p_file in priority_files:
            # Find in map
            found = next((x for x in all_files_map if x[0] == p_file or x[0].endswith(os.sep + p_file)), None)
            if found:
                add_file(*found, is_priority=True)
                processed_files.add(found[0])
        
        # Add Remaining Files
        for rel_path, path, size in all_files_map:
            if rel_path in processed_files: continue
            if not add_file(rel_path, path, size):
                # If we hit the limit, stop adding content
                # But we already have them in the file tree, so that's fine.
                break
                    
        return "\n".join(context)
