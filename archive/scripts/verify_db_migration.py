import os
import sqlite3
from quota_manager import QuotaManager

DB_PATH = "quota.db"

def test_db_migration_and_logic():
    # 1. Clean start
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Removed existing quota.db")

    print("Initializing QuotaManager (should trigger migration)...")
    qm = QuotaManager(os.getcwd())

    # 2. Verify Import
    models = qm.get_all_models()
    print(f"Loaded {len(models)} models from DB.")
    if len(models) == 0:
        print("FAIL: No models imported.")
        return

    # Check specific model
    gemini_flash = next((m for m in models if "Flash" in m["model"]), None)
    if gemini_flash:
        print(f"Found model: {gemini_flash['model']}")
    else:
        print("FAIL: Gemini Flash not found.")

    # 3. Test Availability
    print("Testing availability...")
    is_avail = qm.is_model_available(gemini_flash["model"])
    print(f"Is {gemini_flash['model']} available? {is_avail}")

    # 4. Test Update
    print("Testing quota update...")
    initial_rpm = gemini_flash["rpm"]["used"]
    qm.update_quota(gemini_flash["model"], request_count=10)
    
    # Verify update
    updated_models = qm.get_all_models()
    updated_flash = next(m for m in updated_models if m["model"] == gemini_flash["model"])
    new_rpm = updated_flash["rpm"]["used"]
    print(f"RPM: {initial_rpm} -> {new_rpm}")
    
    if new_rpm == initial_rpm + 10:
        print("PASS: Quota updated.")
    else:
        print("FAIL: Quota not updated correctly.")

    # 5. Test Fallback Logic (Mocking high usage)
    print("Testing fallback logic...")
    # Manually set usage to limit for a model to force fallback
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE models SET rpm_used = rpm_limit + 1 WHERE name = ?", (gemini_flash["model"],))
    
    print(f"Set {gemini_flash['model']} to over-limit.")
    
    # Request the over-limit model
    model_name, config = qm.get_model_for_request(gemini_flash["model"])
    print(f"Requested {gemini_flash['model']}, got: {model_name}")
    
    if model_name != gemini_flash["model"]:
        print(f"PASS: Fallback successful (switched to {model_name})")
    else:
        print("FAIL: Did not fallback, returned exhausted model.")

if __name__ == "__main__":
    test_db_migration_and_logic()
