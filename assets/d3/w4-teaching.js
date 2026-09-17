/* Enlarge the actual visual, preserving its controls and current state. */
(function () {
  'use strict';
  const figures = [...document.querySelectorAll('main .w4-learning')];
  if (!figures.length) return;
  const dialog = document.createElement('dialog');
  dialog.className = 'w4-visual-dialog';
  dialog.setAttribute('aria-labelledby', 'w4-visual-title');
  dialog.innerHTML = '<div class="w4-visual-toolbar"><div><h2 id="w4-visual-title">Explore the visual</h2><p>Use the controls. Press Escape to return to the lecture.</p></div><button type="button" class="w4-visual-close" aria-label="Close visual">Close</button></div><div class="w4-visual-stage"></div>';
  document.body.append(dialog);
  const stage = dialog.querySelector('.w4-visual-stage');
  let active = null;
  dialog.querySelector('.w4-visual-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => {
    if (!active) return;
    const { figure, placeholder, button, scrollY } = active;
    placeholder.replaceWith(figure);
    document.body.classList.remove('w4-visual-open');
    button.focus({ preventScroll: true });
    window.scrollTo({ top: scrollY, behavior: 'instant' });
    active = null;
  });
  figures.forEach(figure => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'w4-enlarge';
    button.textContent = 'Enlarge visual';
    button.setAttribute('aria-haspopup', 'dialog');
    button.addEventListener('click', () => {
      if (active) return;
      const placeholder = document.createElement('div');
      placeholder.style.height = `${figure.getBoundingClientRect().height}px`;
      active = { figure, placeholder, button, scrollY: window.scrollY };
      figure.replaceWith(placeholder);
      stage.append(figure);
      const label = figure.querySelector('.fig-label');
      document.getElementById('w4-visual-title').textContent = label ? label.textContent.replace(/^(Interactive|Experiment)\s*·\s*/, '') : 'Explore the visual';
      document.body.classList.add('w4-visual-open');
      dialog.showModal();
      stage.scrollTop = 0;
    });
    figure.append(button);
  });
})();
