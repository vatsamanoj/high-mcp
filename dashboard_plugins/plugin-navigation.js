(function (global) {
  global.DashboardPluginRegistry.register({
    name: 'navigation-enhancer',
    init() {
      const originalSwitchView = global.switchView;
      if (typeof originalSwitchView === 'function') {
        global.switchView = function patchedSwitchView(viewName, clickEvent) {
          if (clickEvent) {
            global.event = clickEvent;
          }
          return originalSwitchView(viewName);
        };
      }

      const chatInput = document.getElementById('chat-input');
      const hasInlineEnterHandler = !!(chatInput && chatInput.getAttribute('onkeydown'));
      if (chatInput && typeof global.sendMessage === 'function' && !hasInlineEnterHandler) {
        chatInput.addEventListener('keydown', (event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            global.sendMessage();
          }
        });
      }
    },
  });
})(window);
