(function (global) {
  const plugins = [];

  global.DashboardPluginRegistry = {
    register(plugin) {
      if (!plugin || typeof plugin.init !== 'function') {
        throw new Error('Invalid dashboard plugin: missing init()');
      }
      plugins.push(plugin);
    },
    initAll(ctx = {}) {
      plugins.forEach((plugin) => {
        try {
          plugin.init(ctx);
        } catch (error) {
          console.error(`[DashboardPlugin] ${plugin.name || 'unknown'} failed:`, error);
        }
      });
    },
    list() {
      return plugins.map((p) => p.name || 'anonymous-plugin');
    },
  };
})(window);
