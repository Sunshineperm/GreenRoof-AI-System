// Keep Streamlit's non-scrolling iframe as tall as the visible RoofScope page.
(() => {
  const send = (type, payload = {}) =>
    window.parent.postMessage({isStreamlitMessage: true, type, ...payload}, '*');
  let lastHeight = 0;
  let pending = false;

  const measure = () => {
    pending = false;
    if (!document.body) return;
    // scrollHeight includes the existing iframe height and cannot shrink on
    // navigation. The body's natural box can grow AND shrink with the page.
    const body = document.body;
    let viewportHeight = window.innerHeight;
    try { viewportHeight = window.parent.innerHeight || viewportHeight; } catch (_) {}
    const viewportFloor = `${Math.max(600, viewportHeight - 2)}px`;
    // Only the sample-library CSS consumes this value. Avoid repeated style
    // writes so the mutation observer cannot create a resize loop.
    if (body.style.getPropertyValue('--roofscope-parent-height') !== viewportFloor) {
      body.style.setProperty('--roofscope-parent-height', viewportFloor);
    }
    const style = window.getComputedStyle(body);
    const height = Math.max(600, Math.ceil(
      body.getBoundingClientRect().height +
      (parseFloat(style.marginTop) || 0) +
      (parseFloat(style.marginBottom) || 0)
    ) + 2);
    if (height === lastHeight) return;
    lastHeight = height;
    send('streamlit:setFrameHeight', {height});
  };
  const schedule = () => {
    if (pending) return;
    pending = true;
    window.requestAnimationFrame(measure);
  };

  window.addEventListener('message', (event) => {
    if (event.source !== window.parent || event.data?.type !== 'streamlit:render') return;
    schedule();
  });
  window.addEventListener('resize', schedule);
  try { window.parent.addEventListener('resize', schedule); } catch (_) {}
  window.addEventListener('load', schedule);
  const start = () => {
    new ResizeObserver(schedule).observe(document.body);
    new MutationObserver(schedule).observe(document.body, {
      subtree: true, childList: true, characterData: true, attributes: true,
      attributeFilter: ['class', 'style', 'hidden', 'open']
    });
    document.addEventListener('load', schedule, true);
    document.addEventListener('animationend', schedule, true);
    document.fonts?.ready.then(schedule);
    schedule();
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start, {once: true});
  } else {
    start();
  }
  send('streamlit:componentReady', {apiVersion: 1});
})();
