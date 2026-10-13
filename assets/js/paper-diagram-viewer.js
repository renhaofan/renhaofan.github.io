(function () {
  'use strict';

  let viewer;
  let stage;
  let canvas;
  let zoomLabel;
  let scale = 1;
  let offsetX = 0;
  let offsetY = 0;
  let naturalWidth = 800;
  let naturalHeight = 500;
  let activePointer = null;
  let lastX = 0;
  let lastY = 0;

  function makeButton(label, action, title) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.dataset.action = action;
    button.title = title;
    button.setAttribute('aria-label', title);
    return button;
  }

  function createViewer() {
    if (viewer) return;

    viewer = document.createElement('dialog');
    viewer.className = 'diagram-viewer';
    viewer.setAttribute('aria-label', '流程图放大查看器');

    const shell = document.createElement('div');
    shell.className = 'diagram-viewer-shell';
    const toolbar = document.createElement('div');
    toolbar.className = 'diagram-viewer-toolbar';
    toolbar.append(
      makeButton('−', 'zoom-out', '缩小'),
      makeButton('+', 'zoom-in', '放大'),
      makeButton('适应', 'fit', '适应窗口'),
      makeButton('100%', 'actual', '原始大小')
    );

    zoomLabel = document.createElement('span');
    zoomLabel.className = 'diagram-viewer-zoom';
    zoomLabel.setAttribute('aria-live', 'polite');
    toolbar.append(zoomLabel);

    const hint = document.createElement('span');
    hint.className = 'diagram-viewer-hint';
    hint.textContent = '滚轮缩放 · 拖拽平移 · Esc 关闭';
    toolbar.append(hint, makeButton('关闭', 'close', '关闭查看器'));

    stage = document.createElement('div');
    stage.className = 'diagram-viewer-stage';
    canvas = document.createElement('div');
    canvas.className = 'diagram-viewer-canvas';
    stage.append(canvas);
    shell.append(toolbar, stage);
    viewer.append(shell);
    document.body.append(viewer);

    toolbar.addEventListener('click', function (event) {
      const action = event.target.closest('button')?.dataset.action;
      if (action === 'zoom-in') zoomAt(1.2, stage.clientWidth / 2, stage.clientHeight / 2);
      if (action === 'zoom-out') zoomAt(1 / 1.2, stage.clientWidth / 2, stage.clientHeight / 2);
      if (action === 'fit') fitDiagram();
      if (action === 'actual') actualSize();
      if (action === 'close') viewer.close();
    });

    stage.addEventListener('wheel', function (event) {
      event.preventDefault();
      const rect = stage.getBoundingClientRect();
      zoomAt(Math.exp(-event.deltaY * 0.0015), event.clientX - rect.left, event.clientY - rect.top);
    }, { passive: false });

    stage.addEventListener('pointerdown', function (event) {
      activePointer = event.pointerId;
      lastX = event.clientX;
      lastY = event.clientY;
      stage.setPointerCapture(event.pointerId);
      stage.classList.add('is-dragging');
    });

    stage.addEventListener('pointermove', function (event) {
      if (event.pointerId !== activePointer) return;
      offsetX += event.clientX - lastX;
      offsetY += event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      renderTransform();
    });

    function endDrag(event) {
      if (event.pointerId !== activePointer) return;
      activePointer = null;
      stage.classList.remove('is-dragging');
    }
    stage.addEventListener('pointerup', endDrag);
    stage.addEventListener('pointercancel', endDrag);
    viewer.addEventListener('click', function (event) {
      if (event.target === viewer) viewer.close();
    });
  }

  function renderTransform() {
    canvas.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
    zoomLabel.textContent = `${Math.round(scale * 100)}%`;
  }

  function clampScale(value) {
    return Math.min(8, Math.max(0.12, value));
  }

  function zoomAt(factor, x, y) {
    const next = clampScale(scale * factor);
    const ratio = next / scale;
    offsetX = x - (x - offsetX) * ratio;
    offsetY = y - (y - offsetY) * ratio;
    scale = next;
    renderTransform();
  }

  function fitDiagram() {
    const padding = 56;
    const fitX = (stage.clientWidth - padding * 2) / naturalWidth;
    const fitY = (stage.clientHeight - padding * 2) / naturalHeight;
    scale = clampScale(Math.min(fitX, fitY, 2));
    offsetX = (stage.clientWidth - naturalWidth * scale) / 2;
    offsetY = (stage.clientHeight - naturalHeight * scale) / 2;
    renderTransform();
  }

  function actualSize() {
    scale = 1;
    offsetX = (stage.clientWidth - naturalWidth) / 2;
    offsetY = (stage.clientHeight - naturalHeight) / 2;
    renderTransform();
  }

  function openDiagram(diagram) {
    createViewer();
    const source = diagram.querySelector('svg');
    if (!source) return;
    const clone = source.cloneNode(true);
    clone.removeAttribute('style');

    const viewBox = source.viewBox?.baseVal;
    naturalWidth = viewBox?.width || source.getBoundingClientRect().width || 800;
    naturalHeight = viewBox?.height || source.getBoundingClientRect().height || 500;
    clone.setAttribute('width', naturalWidth);
    clone.setAttribute('height', naturalHeight);
    canvas.replaceChildren(clone);
    viewer.showModal();
    requestAnimationFrame(fitDiagram);
  }

  window.initializePaperDiagramViewer = function () {
    document.querySelectorAll('.paper-content .mermaid').forEach(function (diagram, index) {
      if (diagram.dataset.diagramViewerReady || !diagram.querySelector('svg')) return;
      diagram.dataset.diagramViewerReady = 'true';
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'diagram-expand-button';
      button.textContent = '⛶ 放大';
      button.setAttribute('aria-label', `放大查看流程图 ${index + 1}`);
      button.addEventListener('click', function (event) {
        event.stopPropagation();
        openDiagram(diagram);
      });
      diagram.prepend(button);
      diagram.addEventListener('click', function (event) {
        if (!event.target.closest('button, a')) openDiagram(diagram);
      });
    });
  };
}());
