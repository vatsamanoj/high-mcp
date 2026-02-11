(function (global) {
  global.DashboardPluginRegistry.register({
    name: 'core-bootstrap',
    init() {
      if (typeof global.fetchStatus === 'function') {
        setInterval(global.fetchStatus, 3000);
        global.fetchStatus();
      }
      if (typeof global.fetchAutoFixData === 'function') {
        setInterval(global.fetchAutoFixData, 10000);
        global.fetchAutoFixData();
      }
      if (typeof global.fetchFiles === 'function') {
        global.fetchFiles();
      }
      if (typeof global.fetchVersions === 'function') {
        global.fetchVersions();
      }
      if (typeof global.fetchCoderModels === 'function') {
        global.fetchCoderModels();
      }

      const firstNavButton = document.querySelector('nav button');
      if (firstNavButton) {
        firstNavButton.classList.add('active');
      }
    },
  });
})(window);
