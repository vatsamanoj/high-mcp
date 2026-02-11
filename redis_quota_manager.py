import json
import os
import time
import threading
import fakeredis
from typing import Dict, Any, Optional, Tuple, List, Union
import glob
from notification_system import NotificationSystem

class RedisQuotaManager:
    def __init__(self, base_dir: str, db_filename: str = "quota.rdb", persistence_enabled: bool = True):
        self.base_dir = base_dir
        self.quota_dir = os.path.join(base_dir, "quotas")
        self.persistence_enabled = persistence_enabled
        self.notification_system = NotificationSystem(base_dir)
        
        # Ensure quota directory exists
        if not os.path.exists(self.quota_dir):
            os.makedirs(self.quota_dir)

        # We simulate persistence by dumping to a JSON file on exit/interval
        # fakeredis is in-memory only, so we need a backing store
        self.persistence_file = os.path.join(base_dir, "redis_dump.json")
        
        # Initialize fakeredis
        self.redis = fakeredis.FakeStrictRedis(version=6)
        
        # Track file timestamps for hot-reloading
        self._file_timestamps = {}
        
        # Load initial state
        self._load_state()
        self._sync_configuration_from_json()
        
        self._stop_event = threading.Event()
        
        if self.persistence_enabled:
            # Start background persistence thread
            self._persistence_thread = threading.Thread(target=self._persistence_loop, daemon=True)
            self._persistence_thread.start()
        
        # Start file watcher thread
        self._watcher_thread = threading.Thread(target=self._watcher_loop, daemon=True)
        self._watcher_thread.start()

    def _persistence_loop(self):
        """Periodically saves Redis state to disk."""
        while not self._stop_event.is_set():
            time.sleep(5) # Save every 5 seconds
            self._save_state()

    def _save_state(self):
        """Dumps all keys to JSON file with retry logic."""
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                data = {}
                for key in self.redis.keys("*"):
                    key_str = key.decode('utf-8')
                    type_ = self.redis.type(key).decode('utf-8')
                    if type_ == 'string':
                        val = self.redis.get(key)
                        data[key_str] = val.decode('utf-8') if val else None
                        ttl = self.redis.ttl(key)
                        if ttl > 0:
                            data[f"{key_str}:ttl"] = ttl
                    elif type_ == 'set':
                        val = self.redis.smembers(key)
                        data[key_str] = [v.decode('utf-8') for v in val]
                        # Sets usually don't have TTL in this use case, but if they did:
                        ttl = self.redis.ttl(key)
                        if ttl > 0:
                            data[f"{key_str}:ttl"] = ttl
                
                temp_file = self.persistence_file + ".tmp"
                with open(temp_file, 'w') as f:
                    json.dump(data, f)
                
                # Atomic replace
                if os.path.exists(self.persistence_file):
                    try:
                        os.replace(temp_file, self.persistence_file)
                    except OSError:
                        # Fallback for Windows if replace fails (e.g. file locked)
                        os.remove(self.persistence_file)
                        os.rename(temp_file, self.persistence_file)
                else:
                    os.rename(temp_file, self.persistence_file)
                    
                return # Success
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.2) # Short wait before retry
                    continue
                print(f"Error saving state after {max_retries} attempts: {e}")

    def _load_state(self):
        """Loads keys from JSON file."""
        if not os.path.exists(self.persistence_file):
            return
            
        try:
            with open(self.persistence_file, 'r') as f:
                data = json.load(f)
                
            for key, val in data.items():
                if key.endswith(":ttl"):
                    continue
                
                if isinstance(val, list):
                    # It's a Set
                    # Clear existing to be safe
                    self.redis.delete(key)
                    for item in val:
                        self.redis.sadd(key, item)
                else:
                    # It's a String
                    self.redis.set(key, val)
                
                # Set TTL if exists
                ttl_key = f"{key}:ttl"
                if ttl_key in data:
                    self.redis.expire(key, int(data[ttl_key]))
        except Exception as e:
            print(f"Error loading state: {e}")

    def _watcher_loop(self):
        """Monitors *.json files for changes."""
        while not self._stop_event.is_set():
            time.sleep(2) # Check every 2 seconds
            pattern = os.path.join(self.quota_dir, "*.json")
            files = glob.glob(pattern)
            should_reload = False
            
            for f in files:
                try:
                    mtime = os.path.getmtime(f)
                    # Check if new or modified
                    if f not in self._file_timestamps or mtime > self._file_timestamps.get(f, 0):
                        print(f"Change detected in {os.path.basename(f)}, reloading...")
                        should_reload = True
                        break
                except OSError:
                    continue
            
            if should_reload:
                self._sync_configuration_from_json()

    def _sync_configuration_from_json(self):
        """
        Syncs configuration (limits, keys, endpoints) from all *.json files.
        This allows new files or updated limits to be loaded without resetting usage counters.
        """
        print("Syncing configuration from JSON files...")
        
        for key in self.redis.keys("model:*:health"):
            self.redis.delete(key)

        # CLEAR old mappings (Critical for when files are deleted or models removed)
        # We find all config_files sets and delete them so they can be rebuilt
        for key in self.redis.keys("model:*:config_files"):
            self.redis.delete(key)
            
        pattern = os.path.join(self.quota_dir, "*.json")
        for file_path in glob.glob(pattern):
            try:
                # Update timestamp tracking
                self._file_timestamps[file_path] = os.path.getmtime(file_path)

                # Clear blocked status on reload (assuming user fixed the issue)
                file_basename = os.path.basename(file_path)
                if self.redis.exists(f"config:{file_basename}:blocked"):
                    print(f"♻️ Clearing blocked status for {file_basename} due to file update.")
                    self.redis.delete(f"config:{file_basename}:blocked")
                    self.redis.delete(f"config:{file_basename}:blocked_at")

                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    models_list = data
                    api_endpoint = None
                    api_key = None
                    provider = None
                else:
                    models_list = data.get("models", [])
                    api_endpoint = data.get("api_endpoint")
                    api_key = data.get("api_key")
                    provider = data.get("provider", "google") # Default to google
                
                # Store config
                if api_endpoint:
                    self.redis.set(f"config:{os.path.basename(file_path)}:endpoint", api_endpoint)
                if api_key:
                    self.redis.set(f"config:{os.path.basename(file_path)}:key", api_key)
                if provider:
                    self.redis.set(f"config:{os.path.basename(file_path)}:provider", provider)

                for m in models_list:
                    name = m.get("model")
                    
                    # Store static limits (No TTL) - Overwrites existing config
                    self.redis.set(f"model:{name}:rpm:limit", m.get("rpm", {}).get("limit", -1))
                    self.redis.set(f"model:{name}:tpm:limit", m.get("tpm", {}).get("limit", -1))
                    self.redis.set(f"model:{name}:rpd:limit", m.get("rpd", {}).get("limit", -1))
                    
                    # Store config reference
                    self.redis.set(f"model:{name}:config_file", os.path.basename(file_path))
                    
                    # Store category
                    self.redis.set(f"model:{name}:category", m.get("category", "unknown"))
                    
                    # Store tier (default to standard)
                    self.redis.set(f"model:{name}:tier", m.get("tier", "standard"))
                    
                    # Store extra params (e.g. for Nvidia thinking)
                    if "params" in m:
                        self.redis.set(f"model:{name}:params", json.dumps(m["params"]))

                    # --- LOAD BALANCING SUPPORT ---
                    # Add this config file to the set of providers for this model
                    self.redis.sadd(f"model:{name}:config_files", os.path.basename(file_path))
                    
                    # Store PER-CONFIG limits (Critical for aggregated capacity)
                    # We store them with the config_file prefix so each key contributes its own quota
                    self.redis.set(f"config:{os.path.basename(file_path)}:{name}:rpm:limit", m.get("rpm", {}).get("limit", -1))
                    self.redis.set(f"config:{os.path.basename(file_path)}:{name}:tpm:limit", m.get("tpm", {}).get("limit", -1))
                    self.redis.set(f"config:{os.path.basename(file_path)}:{name}:rpd:limit", m.get("rpd", {}).get("limit", -1))


            except Exception as e:
                print(f"Error importing {file_path}: {e}")

    def is_config_available(self, config_file: str, model_name: str) -> bool:
        """Checks if a specific configuration (key) for a model is available."""
        # 0. Check Blocked Status
        blocked_reason = self.redis.get(f"config:{config_file}:blocked")
        if blocked_reason:
            return False

        # 1. Get Limits (Per Config)
        rpm_limit = float(self.redis.get(f"config:{config_file}:{model_name}:rpm:limit") or -1)
        rpd_limit = float(self.redis.get(f"config:{config_file}:{model_name}:rpd:limit") or -1)
        
        # 2. Get Usage (Per Config)
        rpm_used = float(self.redis.get(f"quota:{config_file}:{model_name}:rpm:used") or 0)
        rpd_used = float(self.redis.get(f"quota:{config_file}:{model_name}:rpd:used") or 0)
        
        # 3. Check 90% Rule
        def check(used, limit):
            if limit > 0 and (used / limit) >= 0.9:
                return False
            return True
            
        if not check(rpm_used, rpm_limit): return False
        if not check(rpd_used, rpd_limit): return False
        
        # 4. Check Health (set by FreeAISensor) - This is usually per model, but could be per config if extended
        # For now, if the MODEL is marked down globally (e.g. API outage), all configs are down.
        health = self.redis.get(f"model:{model_name}:health")
        if health and health.decode('utf-8') == "down":
            return False
            
        return True

    def is_model_available(self, model_name: str) -> bool:
        """Checks if ANY config for a model is available."""
        # Get all configs
        config_files_bytes = self.redis.smembers(f"model:{model_name}:config_files")
        if not config_files_bytes:
            # Fallback to legacy single config if set hasn't been populated yet (shouldn't happen with new sync)
            return False
            
        for cf in config_files_bytes:
            if self.is_config_available(cf.decode('utf-8'), model_name):
                return True
                
        return False

    def mark_provider_blocked(self, model_name: str, reason: str, config_file: Optional[str] = None):
        """Marks the provider configuration for a model as blocked."""
        # If config_file is not provided, we try to guess (legacy) or block ALL?
        # Ideally AIEngine passes the config_file.
        
        target_configs = []
        if config_file:
            target_configs.append(config_file)
        else:
             # Block all configs for this model? Or just the current active one?
             # Without config_file, we can't know which one failed.
             # Fallback: Get all configs
             cfs = self.redis.smembers(f"model:{model_name}:config_files")
             target_configs = [c.decode('utf-8') for c in cfs]

        for cf in target_configs:
            print(f"🚫 BLOCKING provider {cf} for {model_name}. Reason: {reason}")
            
            # Set blocked flag
            self.redis.set(f"config:{cf}:blocked", reason)
            self.redis.set(f"config:{cf}:blocked_at", time.time())
            
        # Immediately save state
        if self.persistence_enabled:
            self._save_state()

    def update_quota(self, model_name: str, tokens_used: int = 0, request_count: int = 1, config_file: Optional[str] = None):
        """Updates quota usage with atomic increments and TTL."""
        
        if not config_file:
            # Try to find a default or active config?
            # Without config_file, we can't attribute usage correctly in the new system.
            # We might just log a warning and return, or attribute to the first available?
            # For backward compatibility, let's try to find a config file from the legacy key
            legacy_config = self.redis.get(f"model:{model_name}:config_file")
            if legacy_config:
                config_file = legacy_config.decode('utf-8')
            else:
                # Last resort: pick any config
                cfs = self.redis.smembers(f"model:{model_name}:config_files")
                if cfs:
                    config_file = list(cfs)[0].decode('utf-8')
                else:
                    print(f"⚠️ update_quota called for {model_name} but no config found!")
                    return

        # RPM (1 minute TTL)
        key_rpm = f"quota:{config_file}:{model_name}:rpm:used"
        new_rpm = self.redis.incrby(key_rpm, request_count)
        if new_rpm == request_count:
            self.redis.expire(key_rpm, 60)
        
        # Check RPM Exhaustion
        rpm_limit = float(self.redis.get(f"config:{config_file}:{model_name}:rpm:limit") or -1)
        if rpm_limit > 0:
            utilization = new_rpm / rpm_limit
            if utilization >= 1.0:
                 # Check if we already notified recently (1 min cooldown)
                notify_key = f"notify:{config_file}:{model_name}:rpm:exhausted"
                if not self.redis.exists(notify_key):
                    self.notification_system.notify(
                        f"⚠️ Key {config_file} for {model_name} exhausted RPM limit! ({new_rpm}/{int(rpm_limit)})", 
                        "BLOCKED"
                    )
                    self.redis.setex(notify_key, 60, "1")

        # RPD (24 hour TTL)
        key_rpd = f"quota:{config_file}:{model_name}:rpd:used"
        new_rpd = self.redis.incrby(key_rpd, request_count)
        if new_rpd == request_count:
            self.redis.expire(key_rpd, 86400)

        # Check RPD Exhaustion
        rpd_limit = float(self.redis.get(f"config:{config_file}:{model_name}:rpd:limit") or -1)
        if rpd_limit > 0:
            utilization = new_rpd / rpd_limit
            if utilization >= 1.0:
                # Check if we already notified recently (1 hour cooldown)
                notify_key = f"notify:{config_file}:{model_name}:rpd:exhausted"
                if not self.redis.exists(notify_key):
                    self.notification_system.notify(
                        f"⚠️ Key {config_file} for {model_name} exhausted Daily limit! ({new_rpd}/{int(rpd_limit)})", 
                        "BLOCKED"
                    )
                    self.redis.setex(notify_key, 3600, "1")

        # TPM (1 minute TTL)
        if tokens_used > 0:
            key_tpm = f"quota:{config_file}:{model_name}:tpm:used"
            new_tpm = self.redis.incrby(key_tpm, tokens_used)
            if new_tpm == tokens_used:
                self.redis.expire(key_tpm, 60)

    def set_speed_override(self, enabled: bool):
        """Sets the global speed override flag."""
        val = "1" if enabled else "0"
        self.redis.set("config:global:speed_override", val)
        if self.persistence_enabled:
            self._save_state()

    def get_speed_override(self) -> bool:
        """Gets the global speed override flag."""
        val = self.redis.get("config:global:speed_override")
        return val and val.decode('utf-8') == "1"

    def get_all_models(self) -> List[Dict[str, Any]]:
        """Returns detailed information about all models."""
        model_keys = self.redis.keys("model:*:rpm:limit")
        all_model_names = [k.decode('utf-8').split(':')[1] for k in model_keys]
        
        results = []
        for name in all_model_names:
            tier = (self.redis.get(f"model:{name}:tier") or b"standard").decode('utf-8')
            category = (self.redis.get(f"model:{name}:category") or b"unknown").decode('utf-8')
            available = self.is_model_available(name)
            
            config_files = self.redis.smembers(f"model:{name}:config_files")
            providers = [cf.decode('utf-8') for cf in config_files]
            
            rpm_limit = float(self.redis.get(f"model:{name}:rpm:limit") or -1)
            tpm_limit = float(self.redis.get(f"model:{name}:tpm:limit") or -1)
            rpd_limit = float(self.redis.get(f"model:{name}:rpd:limit") or -1)
            
            results.append({
                "model": name,
                "tier": tier,
                "category": category,
                "available": available,
                "providers": providers,
                "rpm": {"limit": rpm_limit},
                "tpm": {"limit": tpm_limit},
                "rpd": {"limit": rpd_limit}
            })
        return results

    def get_model_for_request(self, preferred_model: Optional[str] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Finds an available model."""
        # 1. Get all known models
        # redis.keys() returns bytes
        model_keys = self.redis.keys("model:*:rpm:limit")
        all_models = [k.decode('utf-8').split(':')[1] for k in model_keys]
        
        candidates = []
        if preferred_model:
            candidates = [m for m in all_models if preferred_model in m]
        
        # Get tiers and categories for all models
        model_tiers = {}
        model_categories = {}
        for m in all_models:
            tier_bytes = self.redis.get(f"model:{m}:tier")
            model_tiers[m] = tier_bytes.decode('utf-8') if tier_bytes else "standard"
            
            cat_bytes = self.redis.get(f"model:{m}:category")
            model_categories[m] = cat_bytes.decode('utf-8') if cat_bytes else "unknown"
            
        # Helper for sorting tiers:
        # Tier 1 = free (Highest Priority) -> 1
        # Tier 2 = economical -> 2
        # Tier 3 = standard -> 3
        # Tier 4 = premium -> 4
        # Unknown -> 3 (Treat as standard)
        def get_tier_value(model):
            tier = model_tiers.get(model, "standard")
            if tier == "free": return 1
            if tier == "economical": return 2
            if tier == "standard": return 3
            if tier == "premium": return 4
            # Handle numeric strings if they sneak in
            if str(tier) == "1": return 1
            if str(tier) == "2": return 2
            if str(tier) == "3": return 3
            if str(tier) == "4": return 4
            return 3
            
        # Helper for sorting speed:
        # 1 = Fast (flash, instant, mini, 8b, fast category)
        # 2 = Normal/Smart (pro, ultra, 70b, smart category)
        def get_speed_value(model):
            cat = model_categories.get(model, "unknown").lower()
            name = model.lower()
            
            # Explicit fast category
            if cat == "fast": return 1
            
            # Heuristic based on name
            fast_keywords = ["flash", "instant", "mini", "8b", "7b", "haiku", "gemma-2-9b"]
            if any(k in name for k in fast_keywords): return 1
            
            return 2
            
        # Check Speed Override
        speed_override = self.get_speed_override()
        
        # Add rest, sorted by logic
        remaining = [m for m in all_models if m not in candidates]
        
        if speed_override:
            # If override is ON, sort primarily by SPEED, then by Tier
            # Speed 1 (Fast) comes before Speed 2 (Slow)
            remaining.sort(key=lambda x: (get_speed_value(x), get_tier_value(x)))
        else:
            # Default: Tier THEN Speed
            remaining.sort(key=lambda x: (get_tier_value(x), get_speed_value(x)))
        
        candidates.extend(remaining)
                
        for model in candidates:
            # --- LOAD BALANCING LOGIC ---
            # Get all config files for this model
            config_files = list(self.redis.smembers(f"model:{model}:config_files"))
            if not config_files:
                # Legacy fallback? Or just skip if no configs found
                continue
                
            # Sort for deterministic behavior
            config_files.sort(key=lambda x: x.decode('utf-8'))
            
            # Get Round-Robin Index
            rr_key = f"model:{model}:rr_index"
            rr_idx = int(self.redis.incr(rr_key)) % len(config_files)
            
            # Rotate list to start from RR index (Load Balancing)
            rotated_configs = config_files[rr_idx:] + config_files[:rr_idx]
            
            for cf_bytes in rotated_configs:
                config_file = cf_bytes.decode('utf-8')
                
                if self.is_config_available(config_file, model):
                    # Found a working config!
                    endpoint = self.redis.get(f"config:{config_file}:endpoint").decode('utf-8')
                    key = self.redis.get(f"config:{config_file}:key").decode('utf-8')
                    
                    # Get provider
                    provider_bytes = self.redis.get(f"config:{config_file}:provider")
                    provider = provider_bytes.decode('utf-8') if provider_bytes else "google"
                    
                    # Get extra params
                    params_bytes = self.redis.get(f"model:{model}:params")
                    params = json.loads(params_bytes) if params_bytes else {}
                    
                    # Return with config_file for tracking
                    return model, {
                        "api_endpoint": endpoint, 
                        "api_key": key, 
                        "provider": provider, 
                        "params": params,
                        "config_file": config_file
                    }
                
        return None, None

    def get_all_models(self) -> List[Dict[str, Any]]:
        """Returns a list of all models with their configurations."""
        model_keys = self.redis.keys("model:*:rpm:limit")
        models = []
        for k in model_keys:
            model_name = k.decode('utf-8').split(':')[1]
            
            # Retrieve basic info
            rpm_limit = float(self.redis.get(f"model:{model_name}:rpm:limit") or -1)
            rpd_limit = float(self.redis.get(f"model:{model_name}:rpd:limit") or -1)
            
            cat_bytes = self.redis.get(f"model:{model_name}:category")
            category = cat_bytes.decode('utf-8') if cat_bytes else "unknown"

            tier_bytes = self.redis.get(f"model:{model_name}:tier")
            tier = tier_bytes.decode('utf-8') if tier_bytes else "standard"
            
            # --- AGGREGATED METRICS (Cluster View) ---
            config_files = self.redis.smembers(f"model:{model_name}:config_files")
            
            total_rpm_limit = 0.0
            total_rpd_limit = 0.0
            total_rpm_used = 0.0
            total_rpd_used = 0.0
            
            # Check if we have any valid limits defined (to avoid showing 0 if all are -1)
            has_rpm_limit = False
            has_rpd_limit = False

            if config_files:
                for cf_bytes in config_files:
                    cf = cf_bytes.decode('utf-8')
                    
                    # Limits
                    lim = float(self.redis.get(f"config:{cf}:{model_name}:rpm:limit") or -1)
                    if lim > 0: 
                        total_rpm_limit += lim
                        has_rpm_limit = True
                    
                    lim_d = float(self.redis.get(f"config:{cf}:{model_name}:rpd:limit") or -1)
                    if lim_d > 0: 
                        total_rpd_limit += lim_d
                        has_rpd_limit = True
                    
                    # Usage
                    used = float(self.redis.get(f"quota:{cf}:{model_name}:rpm:used") or 0)
                    total_rpm_used += used
                    
                    used_d = float(self.redis.get(f"quota:{cf}:{model_name}:rpd:used") or 0)
                    total_rpd_used += used_d
            else:
                 # Legacy fallback
                 total_rpm_limit = float(self.redis.get(f"model:{model_name}:rpm:limit") or -1)
                 total_rpd_limit = float(self.redis.get(f"model:{model_name}:rpd:limit") or -1)
                 if total_rpm_limit > 0: has_rpm_limit = True
                 if total_rpd_limit > 0: has_rpd_limit = True
            
            # If no limits found, set to -1
            if not has_rpm_limit: total_rpm_limit = -1
            if not has_rpd_limit: total_rpd_limit = -1

            # Availability
            available = self.is_model_available(model_name)

            # Providers
            providers = set()
            if config_files:
                for cf_bytes in config_files:
                    cf = cf_bytes.decode('utf-8')
                    p_bytes = self.redis.get(f"config:{cf}:provider")
                    if p_bytes:
                        providers.add(p_bytes.decode('utf-8'))
                    else:
                        providers.add("google") # Default
            else:
                 # Legacy fallback
                 p_bytes = self.redis.get(f"config:{self.redis.get(f'model:{model_name}:config_file').decode('utf-8') if self.redis.get(f'model:{model_name}:config_file') else ''}:provider")
                 if p_bytes:
                     providers.add(p_bytes.decode('utf-8'))

            models.append({
                "model": model_name,
                "category": category,
                "tier": tier,
                "providers": sorted(list(providers)),
                "rpm_limit": total_rpm_limit,
                "rpd_limit": total_rpd_limit,
                "rpm_used": total_rpm_used,
                "rpd_used": total_rpd_used,
                "available": available
            })
        return models

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Returns the API configuration for a specific model."""
        config_file = self.redis.get(f"model:{model_name}:config_file")
        if not config_file:
            return {}
        
        config_file = config_file.decode('utf-8')
        endpoint = self.redis.get(f"config:{config_file}:endpoint").decode('utf-8')
        key = self.redis.get(f"config:{config_file}:key").decode('utf-8')
        
        provider_bytes = self.redis.get(f"config:{config_file}:provider")
        provider = provider_bytes.decode('utf-8') if provider_bytes else "google"
        
        return {
            "api_endpoint": endpoint,
            "api_key": key,
            "provider": provider
        }

    def close(self):
        """Clean shutdown."""
        self._stop_event.set()
        self._save_state()
