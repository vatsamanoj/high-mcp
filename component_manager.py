import os
import time
import threading
import importlib
import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set

@dataclass
class PluginRecord:
    name: str
    module_key: str
    file_path: str
    module: Any
    routes: List[Any] = field(default_factory=list)
    service_names: Set[str] = field(default_factory=set)

class MicroKernel:
    def __init__(self, manager: "ComponentManager", plugin_name: str):
        self.manager = manager
        self.plugin_name = plugin_name

    @property
    def base_dir(self) -> str:
        return self.manager.base_dir

    @property
    def app(self):
        return self.manager.app

    @property
    def mcp(self):
        return self.manager.mcp

    @property
    def trust_system(self):
        return self.manager.trust_system

    def register_service(self, name: str, service: Any):
        self.manager.services[name] = service
        rec = self.manager.plugin_records.get(self.plugin_name)
        if rec:
            rec.service_names.add(name)

    def get_service(self, name: str) -> Any:
        return self.manager.services.get(name)

    def unregister_service(self, name: str):
        self.manager.services.pop(name, None)
        rec = self.manager.plugin_records.get(self.plugin_name)
        if rec:
            rec.service_names.discard(name)

class ComponentManager:
    def __init__(self, base_dir: str, trust_system, mcp_server=None, fastapi_app=None):
        self.base_dir = base_dir
        self.components_dir = os.path.join(base_dir, "components")
        self.plugins_dir = os.path.join(base_dir, "plugins")
        self.mcp = mcp_server
        self.app = fastapi_app
        self.trust_system = trust_system
        self.loaded_components: Dict[str, Any] = {}
        self.plugin_records: Dict[str, PluginRecord] = {}
        self.services: Dict[str, Any] = {}

        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_stop = threading.Event()
        self._watcher_state: Dict[str, float] = {}
        
        if not os.path.exists(self.components_dir):
            os.makedirs(self.components_dir)
        if not os.path.exists(self.plugins_dir):
            os.makedirs(self.plugins_dir)

    def load_all_components(self):
        """Loads all python modules found in components/ and plugins/ directories."""
        print("🧩 ComponentManager: Scanning for components and plugins...")
        
        # Snapshot before loading (Integrity Checkpoint)
        # if self.trust_system:
        #      self.trust_system.create_snapshot("pre-load-checkpoint")
        
        candidates = set()
        
        # 1. Scan Components
        if os.path.exists(self.components_dir):
            for filename in os.listdir(self.components_dir):
                if filename.endswith(".py") and not filename.startswith("_"):
                    candidates.add(filename[:-3])

        # 2. Scan Plugins (Runtime Injections)
        if os.path.exists(self.plugins_dir):
            for filename in os.listdir(self.plugins_dir):
                if filename.endswith(".py") and not filename.startswith("_"):
                    candidates.add(filename[:-3])
        
        for component_name in candidates:
            self.load_component(component_name)

    def _module_key(self, name: str) -> str:
        return f"high_mcp.components.{name}"

    def _capture_routes(self) -> List[Any]:
        if not self.app:
            return []
        router = getattr(self.app, "router", None)
        routes = getattr(router, "routes", None)
        if isinstance(routes, list):
            return list(routes)
        return []

    def _diff_routes(self, before: List[Any], after: List[Any]) -> List[Any]:
        before_ids = {id(r) for r in before}
        return [r for r in after if id(r) not in before_ids]

    def _remove_routes(self, routes_to_remove: List[Any]):
        if not self.app or not routes_to_remove:
            return
        router = getattr(self.app, "router", None)
        routes = getattr(router, "routes", None)
        if not isinstance(routes, list):
            return
        remove_ids = {id(r) for r in routes_to_remove}
        router.routes = [r for r in routes if id(r) not in remove_ids]

    def load_component(self, name: str) -> bool:
        """Loads a single component by name."""
        # Check Plugins First (Runtime Injection/Patch)
        file_path = os.path.join(self.plugins_dir, f"{name}.py")
        if not os.path.exists(file_path):
            # Fallback to Core Components
            file_path = os.path.join(self.components_dir, f"{name}.py")
        
        if not os.path.exists(file_path):
            print(f"❌ ComponentManager: Component {name} not found.")
            return False
            
        print(f"🧩 ComponentManager: Loading {name} from {file_path}...")
        
        try:
            module_key = self._module_key(name)
            if module_key in sys.modules:
                self.unload_component(name)

            before_routes = self._capture_routes()

            spec = importlib.util.spec_from_file_location(module_key, file_path)
            if not spec or not spec.loader:
                raise RuntimeError("Failed to create module spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_key] = module
            spec.loader.exec_module(module)

            rec = PluginRecord(name=name, module_key=module_key, file_path=file_path, module=module)
            self.plugin_records[name] = rec
            kernel = MicroKernel(self, plugin_name=name)

            initialize = getattr(module, "initialize", None)
            start = getattr(module, "start", None)
            setup = getattr(module, "setup", None)

            if callable(initialize):
                initialize(kernel)
            elif callable(setup):
                try:
                    setup(mcp=self.mcp, app=self.app)
                except TypeError:
                    try:
                        setup(self.mcp)
                    except TypeError:
                        setup()

            if callable(start):
                start(kernel)

            after_routes = self._capture_routes()
            rec.routes = self._diff_routes(before_routes, after_routes)

            self.loaded_components[name] = module
            print(f"✅ ComponentManager: {name} loaded successfully.")
            return True
            
        except Exception as e:
            print(f"❌ ComponentManager: Failed to load {name}: {e}")
            traceback.print_exc()
            return False

    def unload_component(self, name: str) -> bool:
        rec = self.plugin_records.get(name)
        if not rec:
            return False

        print(f"🧹 ComponentManager: Unloading {name}...")
        kernel = MicroKernel(self, plugin_name=name)
        stop = getattr(rec.module, "stop", None)
        if callable(stop):
            try:
                stop(kernel)
            except Exception:
                traceback.print_exc()

        self._remove_routes(rec.routes)
        for svc in list(rec.service_names):
            self.services.pop(svc, None)

        self.loaded_components.pop(name, None)
        self.plugin_records.pop(name, None)
        sys.modules.pop(rec.module_key, None)
        print(f"✅ ComponentManager: {name} unloaded successfully.")
        return True

    def attach_component(self, name: str) -> bool:
        """Manually attach (load) a discovered component/plugin by name."""
        if name in self.loaded_components:
            return True
        return self.load_component(name)

    def detach_component(self, name: str) -> bool:
        """Manually detach (unload) a currently attached component/plugin by name."""
        if name not in self.loaded_components:
            return False
        return self.unload_component(name)

    def reload_component(self, name: str):
        """Reloads a component."""
        print(f"🔄 ComponentManager: Reloading {name}...")
        rec = self.plugin_records.get(name)
        if not rec:
            return self.load_component(name)

        kernel = MicroKernel(self, plugin_name=name)
        stop = getattr(rec.module, "stop", None)
        if callable(stop):
            try:
                stop(kernel)
            except Exception:
                traceback.print_exc()

        self._remove_routes(rec.routes)
        for svc in list(rec.service_names):
            self.services.pop(svc, None)
        rec.routes = []
        rec.service_names.clear()

        before_routes = self._capture_routes()

        try:
            reloaded = importlib.reload(rec.module)
            rec.module = reloaded
            self.loaded_components[name] = reloaded

            reload_hook = getattr(reloaded, "reload", None)
            initialize = getattr(reloaded, "initialize", None)
            start = getattr(reloaded, "start", None)
            setup = getattr(reloaded, "setup", None)

            if callable(reload_hook):
                reload_hook(kernel)
            else:
                if callable(initialize):
                    initialize(kernel)
                elif callable(setup):
                    try:
                        setup(mcp=self.mcp, app=self.app)
                    except TypeError:
                        try:
                            setup(self.mcp)
                        except TypeError:
                            setup()
                if callable(start):
                    start(kernel)

            after_routes = self._capture_routes()
            rec.routes = self._diff_routes(before_routes, after_routes)
            print(f"✅ ComponentManager: {name} reloaded successfully.")
            return True
        except Exception as e:
            print(f"❌ ComponentManager: Failed to reload {name}: {e}")
            traceback.print_exc()
            return False

    def list_components(self) -> List[Dict[str, Any]]:
        """
        Return discovered components/plugins with attach status.

        attached_to_project/in_use=True means the module is currently loaded.
        attached_to_project/in_use=False means the file exists but failed to load or is not loaded.
        """
        out: List[Dict[str, Any]] = []

        discovered: Dict[str, Dict[str, Any]] = {}

        def _discover_from_dir(base_dir: str, source: str):
            if not os.path.exists(base_dir):
                return
            for filename in os.listdir(base_dir):
                if not filename.endswith('.py') or filename.startswith('_'):
                    continue
                name = filename[:-3]
                file_path = os.path.join(base_dir, filename)
                discovered[name] = {
                    "name": name,
                    "file_path": file_path,
                    "source": source,
                }

        # Components discovered first, then plugins overwrite by name (runtime override behavior)
        _discover_from_dir(self.components_dir, "component")
        _discover_from_dir(self.plugins_dir, "plugin")

        for name, info in discovered.items():
            rec = self.plugin_records.get(name)
            in_use = rec is not None and name in self.loaded_components
            file_path = rec.file_path if rec else info["file_path"]
            try:
                mtime = os.path.getmtime(file_path)
            except Exception:
                mtime = None

            out.append({
                "name": name,
                "file_path": file_path,
                "source": info["source"],
                "attached_to_project": in_use,
                "in_use": in_use,
                "needs_plug_in": not in_use,
                "can_attach": not in_use,
                "can_detach": in_use,
                "routes": len(rec.routes) if rec else 0,
                "services": sorted(list(rec.service_names)) if rec else [],
            })

        return sorted(out, key=lambda x: x["name"])

    def start_watcher(self, poll_interval_seconds: float = 1.0):
        if self._watcher_thread and self._watcher_thread.is_alive():
            return

        self._watcher_stop.clear()
        self._watcher_state = {}

        def scan() -> Dict[str, float]:
            state: Dict[str, float] = {}
            
            # Helper to scan a directory
            def _scan_dir(d):
                if not os.path.exists(d): return
                try:
                    for filename in os.listdir(d):
                        if not filename.endswith(".py") or filename.startswith("_"):
                            continue
                        path = os.path.join(d, filename)
                        try:
                            # If name exists, this overwrites it (Plugins overwrite Components)
                            state[filename[:-3]] = os.path.getmtime(path)
                        except Exception:
                            continue
                except Exception:
                    pass

            # Scan components first, then plugins (so plugins overwrite state)
            _scan_dir(self.components_dir)
            _scan_dir(self.plugins_dir)
            
            return state

        self._watcher_state = scan()

        def loop():
            while not self._watcher_stop.is_set():
                time.sleep(max(0.2, float(poll_interval_seconds)))
                cur = scan()

                for name in list(self._watcher_state.keys()):
                    if name not in cur:
                        try:
                            self.unload_component(name)
                        except Exception:
                            traceback.print_exc()

                for name, mtime in cur.items():
                    prev = self._watcher_state.get(name)
                    if prev is None:
                        try:
                            self.load_component(name)
                        except Exception:
                            traceback.print_exc()
                        continue
                    if mtime != prev:
                        try:
                            self.reload_component(name)
                        except Exception:
                            traceback.print_exc()

                self._watcher_state = cur

        self._watcher_thread = threading.Thread(target=loop, daemon=True)
        self._watcher_thread.start()

    def stop_watcher(self):
        self._watcher_stop.set()
        t = self._watcher_thread
        self._watcher_thread = None
        if t and t.is_alive():
            t.join(timeout=2.0)
