import os
import shutil
import json
import time
import zipfile
from datetime import datetime
from typing import List, Dict, Optional

class TrustSystem:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.trust_store_dir = os.path.join(base_dir, "trust_store")
        self.backups_dir = os.path.join(self.trust_store_dir, "backups")
        self.manifest_file = os.path.join(self.trust_store_dir, "manifest.json")
        
        if not os.path.exists(self.backups_dir):
            os.makedirs(self.backups_dir)
            
        self._load_manifest()

    def _load_manifest(self):
        if os.path.exists(self.manifest_file):
            try:
                with open(self.manifest_file, 'r') as f:
                    self.manifest = json.load(f)
            except:
                self.manifest = {"versions": [], "current_version": None}
        else:
            self.manifest = {"versions": [], "current_version": None}

    def _save_manifest(self):
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def create_snapshot(self, label: str = "auto-backup") -> str:
        """Creates a snapshot of the current system state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        version_id = f"v_{timestamp}"
        backup_path = os.path.join(self.backups_dir, f"{version_id}.zip")
        
        print(f"🛡️ TrustSystem: Creating snapshot {version_id}...")
        
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                ignore_dirs = {
                    ".git",
                    ".venv",
                    "venv",
                    "__pycache__",
                    "node_modules",
                    "trust_store",
                }
                ignore_files = {
                    "redis_dump.json",
                }

                for root, dirs, files in os.walk(self.base_dir):
                    rel_root = os.path.relpath(root, self.base_dir)
                    if rel_root == ".":
                        rel_root = ""

                    dirs[:] = [d for d in dirs if d not in ignore_dirs]

                    for file in files:
                        if file in ignore_files:
                            continue
                        if file.endswith(".pyc"):
                            continue

                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.base_dir)
                        zipf.write(file_path, arcname)
            
            # Update manifest
            entry = {
                "id": version_id,
                "timestamp": datetime.now().isoformat(),
                "label": label,
                "path": backup_path,
                "status": "healthy" 
            }
            self.manifest["versions"].append(entry)
            self.manifest["current_version"] = version_id
            self._save_manifest()
            
            print(f"✅ TrustSystem: Snapshot {version_id} created.")
            return version_id
            
        except Exception as e:
            print(f"❌ TrustSystem: Snapshot failed: {e}")
            return None

    def rollback(self, version_id: str) -> bool:
        """Rolls back the system to a specific version."""
        print(f"⚠️ TrustSystem: Rolling back to {version_id}...")
        
        version = next((v for v in self.manifest["versions"] if v["id"] == version_id), None)
        if not version:
            print(f"❌ TrustSystem: Version {version_id} not found.")
            return False
            
        backup_path = version["path"]
        if not os.path.exists(backup_path):
            print(f"❌ TrustSystem: Backup file missing for {version_id}.")
            return False
            
        try:
            # Extract and overwrite
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(self.base_dir)
                
            print(f"✅ TrustSystem: Rollback successful. System restored to {version_id}.")
            self.manifest["current_version"] = version_id
            self._save_manifest()
            return True
            
        except Exception as e:
            print(f"❌ TrustSystem: Rollback failed: {e}")
            return False

    def get_latest_version(self) -> Optional[str]:
        if not self.manifest["versions"]:
            return None
        return self.manifest["versions"][-1]["id"]

    @property
    def current_version(self) -> Optional[str]:
        """Returns the currently active snapshot version id."""
        return self.manifest.get("current_version")

    def list_snapshots(self) -> List[Dict]:
        """Returns snapshot metadata in reverse-chronological order."""
        versions = list(self.manifest.get("versions", []))
        return sorted(versions, key=lambda v: v.get("timestamp", ""), reverse=True)

    def restore_snapshot(self, version_id: str) -> bool:
        """Compatibility wrapper used by UI layer."""
        return self.rollback(version_id)
