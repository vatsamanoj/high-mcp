import json
import os
import httpx
import uuid
import time

BASE_URL = "http://localhost:8004"
PATCHES_FILE = r"c:\Users\HP\Documents\high-mcp\logs\patches.json"

def create_patch(code, name="test_patch"):
    patch_id = str(uuid.uuid4())
    patch = {
        "id": patch_id,
        "error_id": "manual-test",
        "created_at": "2024-01-01T00:00:00",
        "analysis": {"what": "Test"},
        "code_change": code,
        "verification_code": "",
        "status": "pending",
        "confidence": 1.0,
        "simulation_result": None
    }
    
    if os.path.exists(PATCHES_FILE):
        with open(PATCHES_FILE, 'r') as f:
            try:
                patches = json.load(f)
            except:
                patches = []
    else:
        patches = []
        
    patches.append(patch)
    
    with open(PATCHES_FILE, 'w') as f:
        json.dump(patches, f, indent=2)
        
    return patch_id

def test_simulation():
    print("🧪 Testing System Verification Started")
    
    # 1. Valid Component
    valid_code = """
def setup(app):
    print("Setup called")
"""
    pid1 = create_patch(valid_code)
    print(f"1. Valid Component (ID: {pid1})")
    resp = httpx.post(f"{BASE_URL}/api/patches/{pid1}/simulate", timeout=30.0)
    print(f"   Status: {resp.status_code}")
    try:
        print(f"   Response: {resp.json()}")
        if resp.json().get("success") == True:
            print("   ✅ Valid component passed")
        else:
            print("   ❌ Valid component failed")
    except:
        print(f"   Response Text: {resp.text}")

    # 2. Syntax Error
    invalid_syntax = """
def setup(app)
    print("Missing colon")
"""
    pid2 = create_patch(invalid_syntax)
    print(f"\n2. Syntax Error Component (ID: {pid2})")
    resp = httpx.post(f"{BASE_URL}/api/patches/{pid2}/simulate", timeout=30.0)
    print(f"   Status: {resp.status_code}")
    print(f"   Response: {resp.json()}")
    if resp.json().get("success") == False and "Syntax Check" in str(resp.json()):
        print("   ✅ Syntax error caught")
    else:
        print("   ❌ Syntax error not caught correctly")

    # 3. Runtime Error (Import)
    runtime_error = """
import non_existent_module_xyz_123

def setup(app):
    pass
"""
    pid3 = create_patch(runtime_error)
    print(f"\n3. Runtime Error Component (ID: {pid3})")
    resp = httpx.post(f"{BASE_URL}/api/patches/{pid3}/simulate", timeout=30.0)
    print(f"   Status: {resp.status_code}")
    print(f"   Response: {resp.json()}")
    if resp.json().get("success") == False and "Runtime Import" in str(resp.json()):
        print("   ✅ Runtime error caught")
    else:
        print("   ❌ Runtime error not caught correctly")

if __name__ == "__main__":
    try:
        test_simulation()
    except Exception as e:
        print(f"Error: {e}")
