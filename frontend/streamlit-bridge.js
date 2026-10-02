// Streamlit transport only; all product functionality is the original RoofScope app.
(() => {
  const send = (type, payload = {}) => window.parent.postMessage({isStreamlitMessage: true, type, ...payload}, '*');
  let height = Math.max(850, Math.min(1250, (window.screen.availHeight || 1100) - 90));
  const resize = () => send('streamlit:setFrameHeight', {height});
  window.addEventListener('message', (event) => {
    if (event.source !== window.parent || event.data?.type !== 'streamlit:render') return;
    if (Number.isFinite(event.data.args?.height)) height = Math.max(850, event.data.args.height);
    resize();
  });
  send('streamlit:componentReady', {apiVersion: 1});
  resize();
  window.addEventListener('resize', resize);
})();
