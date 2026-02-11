import json
import os
import time
import traceback
import uuid
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from testing_system import ComponentTestingSystem

class ErrorManager:
    def __init__(self, base_dir: str, ai_engine=None, trust_system=None):
        self.base_dir = base_dir
        self.ai_engine = ai_engine
        self.trust_system = trust_system
        self.testing_system = ComponentTestingSystem(base_dir)
        self.logs_dir = os.path.join(base_dir, "logs")
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
            
        self.errors_file = os.path.join(self.logs_dir, "errors.json")
        self.patches_file = os.path.join(self.logs_dir, "patches.json")
        self.config_file = os.path.join(self.logs_dir, "autofix_config.json")
        
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        
        self.config = {
            "auto_fix_enabled": False,
            "schedule_interval_minutes": 60,
            "auto_apply_confidence_threshold": 0.9
        }
        
        self._load_config()
        self._schedule_patching()

    def _load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config.update(json.load(f))
            except Exception as e:
                print(f"Error loading autofix config: {e}")

    def _save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def update_config(self, new_config: Dict[str, Any]):
        self.config.update(new_config)
        self._save_config()
        self._schedule_patching()

    def _schedule_patching(self):
        self.scheduler.remove_all_jobs()
        if self.config["auto_fix_enabled"]:
            self.scheduler.add_job(
                self.apply_pending_patches, 
                'interval', 
                minutes=self.config["schedule_interval_minutes"]
            )
            print(f"🔧 Auto-fix scheduled every {self.config['schedule_interval_minutes']} minutes")

    def log_error(self, error: Exception, context: str = "") -> str:
        """Logs an error and returns the error ID."""
        error_id = str(uuid.uuid4())
        tb = traceback.format_exc()
        
        entry = {
            "id": error_id,
            "timestamp": datetime.now().isoformat(),
            "message": str(error),
            "traceback": tb,
            "context": context,
            "status": "new",  # new, analyzing, analyzed, patched, ignored
            "analysis": None
        }
        
        errors = self._read_json(self.errors_file)
        errors.append(entry)
        self._write_json(self.errors_file, errors)
        
        print(f"❌ Error logged: {error_id} - {str(error)[:50]}...")
        
        # Trigger immediate analysis in background if enabled
        if self.config["auto_fix_enabled"]:
            self.scheduler.add_job(self.analyze_error, args=[error_id])
            
        return error_id

    def analyze_error(self, error_id: str):
        """Uses AI to analyze error and suggest a patch."""
        errors = self._read_json(self.errors_file)
        error_entry = next((e for e in errors if e["id"] == error_id), None)
        
        if not error_entry or error_entry["status"] != "new":
            return

        print(f"🧠 Analyzing error {error_id}...")
        error_entry["status"] = "analyzing"
        self._write_json(self.errors_file, errors)

        try:
            if self.ai_engine:
                # Real AI Analysis
                prompt = f"""
                You are an expert Python debugger. Analyze the following error and provide a fix suggestion.
                
                Error Message: {error_entry['message']}
                Context: {error_entry.get('context', 'No context')}
                Traceback:
                {error_entry['traceback']}
                
                Return a JSON object with the following fields:
                - what_happened: A clear explanation of the error.
                - why_happened: The root cause of the issue.
                - remedy_steps: A list of steps to fix the issue.
                - code_change: A string containing the code to be patched (or null if manual intervention is needed).
                - verification_code: A standalone Python script to verify the fix works. It should exit with 0 on success and non-zero on failure.
                - confidence: A float between 0.0 and 1.0 indicating your confidence.
                
                Respond ONLY with the JSON string, no markdown formatting.
                """
                
                response_text = self.ai_engine.generate_content("gemini-2.0-flash", prompt)
                
                # Clean up response if it contains markdown code blocks
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                    
                try:
                    analysis_result = json.loads(response_text)
                    what_happened = analysis_result.get("what_happened", "Unknown error")
                    why_happened = analysis_result.get("why_happened", "Unknown cause")
                    remedy_steps = analysis_result.get("remedy_steps", [])
                    patch_code = analysis_result.get("code_change", "")
                    verification_code = analysis_result.get("verification_code", "")
                    confidence = analysis_result.get("confidence", 0.5)
                except json.JSONDecodeError:
                    print(f"Failed to parse AI response: {response_text}")
                    raise Exception("Invalid JSON from AI")
            else:
                 raise Exception("No AI Engine available")

        except Exception as e:
            print(f"AI Analysis failed ({e}), falling back to heuristics...")
            # Fallback Heuristics
            what_happened = "Error detected based on traceback."
            why_happened = "Automatic analysis failed."
            remedy_steps = ["Check logs", "Manual intervention required"]
            patch_code = ""
            verification_code = ""
            confidence = 0.5
            
            if "JSONDecodeError" in error_entry["traceback"]:
                what_happened = "JSON Decoding Error"
                why_happened = "File corruption or invalid format"
                remedy_steps = ["Backup existing file", "Reset to default empty JSON"]
                confidence = 0.8

        # Generate a Patch entry
        patch_id = str(uuid.uuid4())
        patch = {
            "id": patch_id,
            "error_id": error_id,
            "created_at": datetime.now().isoformat(),
            "analysis": {
                "what": what_happened,
                "why": why_happened,
                "remedy": remedy_steps
            },
            "code_change": patch_code,
            "verification_code": verification_code,
            "status": "pending", # pending, applied, rejected
            "confidence": confidence,
            "simulation_result": None # To store simulation outcome
        }
        
        patches = self._read_json(self.patches_file)
        patches.append(patch)
        self._write_json(self.patches_file, patches)
        
        error_entry["status"] = "analyzed"
        error_entry["analysis"] = what_happened
        self._write_json(self.errors_file, errors)
        
        print(f"💡 Patch suggested for {error_id}: {what_happened}")

    def simulate_patch(self, patch_id: str) -> Dict:
        """Simulates applying a patch to verify it."""
        patches = self._read_json(self.patches_file)
        patch = next((p for p in patches if p["id"] == patch_id), None)
        
        if not patch:
            return {"success": False, "message": "Patch not found"}
            
        if not patch["code_change"]:
             return {"success": False, "message": "No code change to simulate"}

        print(f"🧪 Simulating patch {patch_id}...")
        
        try:
            # Use Testing System
            test_results = self.testing_system.run_full_suite(patch_id, patch["code_change"])
            
            if test_results["passed"]:
                message = "✅ All system tests passed."
                details = "Checks: " + ", ".join([f"{c['name']}: {c['status']}" for c in test_results["checks"]])
            else:
                message = "❌ System tests failed."
                details = "Errors: " + ", ".join(test_results["errors"])
            
            result = {
                "success": test_results["passed"],
                "message": message,
                "details": details,
                "raw_results": test_results
            }
            
            patch["simulation_result"] = result
            self._write_json(self.patches_file, patches)
            return result
            
        except Exception as e:
            return {"success": False, "message": f"Simulation failed: {str(e)}"}

    def apply_pending_patches(self):
        """Applies patches that meet the criteria."""
        patches = self._read_json(self.patches_file)
        pending = [p for p in patches if p["status"] == "pending"]
        
        for patch in pending:
            if patch["confidence"] >= self.config["auto_apply_confidence_threshold"]:
                self.apply_patch(patch["id"])

    def apply_patch(self, patch_id: str):
        """Applies a specific patch with backup and rollback."""
        patches = self._read_json(self.patches_file)
        patch = next((p for p in patches if p["id"] == patch_id), None)
        
        if not patch:
            return {"success": False, "message": "Patch not found"}
            
        print(f"🛠️ Applying patch {patch_id}...")
        
        # Determine target file (Default to server.py if not specified)
        target_file = os.path.join(self.base_dir, "server.py")
        
        if not os.path.exists(target_file):
             return {"success": False, "message": f"Target file {target_file} not found"}

        # 1. Create Backup (Trust System or Local)
        if self.trust_system:
            try:
                snapshot_id = self.trust_system.create_snapshot(f"pre-patch-{patch_id}")
                if not snapshot_id:
                     raise Exception("Trust System failed to create snapshot")
                print(f"🛡️ Trust System snapshot created: {snapshot_id}")
            except Exception as e:
                return {"success": False, "message": f"Snapshot failed: {str(e)}"}
            
            # Local backup for immediate rollback (still useful for file-level operations)
            backup_file = target_file + ".bak"
            try:
                with open(target_file, 'r') as f:
                    original_content = f.read()
                with open(backup_file, 'w') as f:
                    f.write(original_content)
            except Exception:
                pass # Trust system is the main backup now
        else:
            # Fallback to local backup
            backup_file = target_file + ".bak"
            try:
                with open(target_file, 'r') as f:
                    original_content = f.read()
                
                with open(backup_file, 'w') as f:
                    f.write(original_content)
                    
                print(f"📦 Backup created: {backup_file}")
                
            except Exception as e:
                return {"success": False, "message": f"Backup failed: {str(e)}"}

        # 2. Apply Patch
        # Assuming code_change is the NEW CONTENT for simplicity/robustness in this demo
        # If it's a snippet, we would need sophisticated find/replace logic.
        try:
            new_content = patch.get("code_change")
            if not new_content:
                return {"success": False, "message": "No code change provided"}

            # If the patch is a snippet (heuristic check), we can't apply it blindly.
            # But if the AI is instructed to return the full file or we use a diff tool...
            # For safety, let's assume the AI provides a valid python script to REPLACE server.py
            # OR better: The AI provides a 'patch plugin' script that we execute?
            # User said: "it will write python code to heal the server.py"
            
            # Let's try to detect if it's a full file replacement or a snippet
            if "import " in new_content and "FastMCP" in new_content:
                # Likely a full file
                with open(target_file, 'w') as f:
                    f.write(new_content)
            else:
                # It's a snippet. Appending it might be safer than replacing?
                # Or maybe it's a diff?
                # For this demo, let's just append it to the end if it looks like a helper function
                # This is risky but "Applying patches" is inherently risky without unified diffs.
                with open(target_file, 'a') as f:
                    f.write("\n\n# --- AUTO-FIX PATCH ---\n")
                    f.write(new_content)
                    f.write("\n# ----------------------\n")

            # 3. Verification (Syntax Check)
            try:
                with open(target_file, 'r') as f:
                    compile(f.read(), target_file, 'exec')
            except Exception as e:
                raise Exception(f"Syntax Error in patched file: {e}")

            patch["status"] = "applied"
            patch["applied_at"] = datetime.now().isoformat()
            self._write_json(self.patches_file, patches)
            
            return {"success": True, "message": "Patch applied successfully. Backup created."}
            
        except Exception as e:
            # Rollback
            print(f"⚠️ Patch failed ({e}). Rolling back...")
            try:
                with open(backup_file, 'r') as f:
                    restore_content = f.read()
                with open(target_file, 'w') as f:
                    f.write(restore_content)
                return {"success": False, "message": f"Patch failed and rolled back: {str(e)}"}
            except Exception as rb_e:
                return {"success": False, "message": f"CRITICAL: Patch failed AND Rollback failed: {str(e)} | {str(rb_e)}"}

    def get_recent_errors(self, limit: int = 20) -> List[Dict]:
        errors = self._read_json(self.errors_file)
        # Sort by timestamp desc
        errors.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return errors[:limit]

    def get_pending_patches(self) -> List[Dict]:
        patches = self._read_json(self.patches_file)
        return [p for p in patches if p.get("status") == "pending"]

    def get_errors(self):
        return self._read_json(self.errors_file)
        
    def get_patches(self):
        return self._read_json(self.patches_file)

    def _read_json(self, path: str) -> List[Dict]:
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return []

    def _write_json(self, path: str, data: List[Dict]):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
