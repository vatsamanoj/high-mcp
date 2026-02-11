import os
import sys
import importlib.util
import traceback
import subprocess
import ast
import uuid
from typing import Dict, Any, List, Optional

class ComponentTestingSystem:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.components_dir = os.path.join(base_dir, "components")
        if not os.path.exists(self.components_dir):
            os.makedirs(self.components_dir)

    def test_component_code(self, component_name: str, code_content: str) -> Dict[str, Any]:
        """
        Tests component code content before it is saved/applied.
        """
        results = {
            "name": component_name,
            "timestamp": str(uuid.uuid4()),
            "checks": [],
            "passed": False,
            "errors": []
        }
        
        # 1. Syntax Check
        try:
            ast.parse(code_content)
            results["checks"].append({"name": "Syntax Check", "status": "passed", "message": "Code structure is valid"})
        except SyntaxError as e:
            msg = f"Line {e.lineno}: {e.msg}"
            results["checks"].append({"name": "Syntax Check", "status": "failed", "message": msg})
            results["errors"].append(msg)
            return results # Stop if syntax is bad
        except Exception as e:
            results["checks"].append({"name": "Syntax Check", "status": "failed", "message": str(e)})
            results["errors"].append(str(e))
            return results

        # 2. Structure/Contract Check
        try:
            tree = ast.parse(code_content)
            has_setup = False
            has_router = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name == 'setup':
                        has_setup = True
                # Check for router usage (heuristic)
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and "router" in target.id.lower():
                            has_router = True
            
            if has_setup:
                results["checks"].append({"name": "Contract Check", "status": "passed", "message": "Found 'setup' function"})
            else:
                results["checks"].append({"name": "Contract Check", "status": "warning", "message": "No 'setup' function found (ComponentManager compatibility warning)"})

        except Exception as e:
            results["checks"].append({"name": "Contract Check", "status": "failed", "message": str(e)})

        # 3. Dynamic/Runtime Check (Isolation)
        # We write the content to a temporary file and try to import it in a subprocess
        # This prevents crashing the main server
        
        # Use a unique name to avoid conflicts
        temp_id = uuid.uuid4().hex[:8]
        temp_module_name = f"test_{component_name}_{temp_id}"
        temp_file_name = f"{temp_module_name}.py"
        temp_file_path = os.path.join(self.components_dir, temp_file_name)
        
        try:
            # Write temp file
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(code_content)
                
            # Run subprocess to test import
            # We import it as components.temp_file_name
            cmd = [
                sys.executable, 
                "-c", 
                f"import sys; sys.path.append(r'{self.base_dir}'); import components.{temp_module_name} as c; print('Import Successful')"
            ]
            
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if proc.returncode == 0 and "Import Successful" in proc.stdout:
                 results["checks"].append({"name": "Runtime Import", "status": "passed", "message": "Component loads without crashing"})
            else:
                 err_msg = proc.stderr.strip() or proc.stdout.strip()
                 results["checks"].append({"name": "Runtime Import", "status": "failed", "message": err_msg})
                 results["errors"].append(f"Runtime Import Failed: {err_msg}")
                 
        except subprocess.TimeoutExpired:
             results["checks"].append({"name": "Runtime Import", "status": "failed", "message": "Import timed out (infinite loop?)"})
             results["errors"].append("Runtime Check Timeout")
        except Exception as e:
             results["checks"].append({"name": "Runtime Import", "status": "failed", "message": str(e)})
             results["errors"].append(f"Runtime Check Error: {e}")
        finally:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass

        # Final Decision
        results["passed"] = len(results["errors"]) == 0
        return results

    def run_full_suite(self, patch_id: str, code_content: str) -> Dict[str, Any]:
        """
        Runs the full testing suite for a proposed patch content.
        """
        # Assume the patch is for a component if it's in the components dir logic
        # For now, we treat all python code patches as potential components or scripts
        
        return self.test_component_code("patched_component", code_content)
