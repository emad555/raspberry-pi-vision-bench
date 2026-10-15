const element = (id) => document.getElementById(id);
const modelHints = {
  'ssdlite-v2': 'Recommended starting point: lightweight, 20 classes.',
  ssd: '20 classes. Gave stronger boxes in the sample test.',
  'ssdlite-v3': '20 classes. Fast, but gave weaker detections in the sample test.',
  squeezenet: 'Compact alternative with 20 classes. Compare its live FPS and detections.',
  yolo: '80 classes. Smallest YOLO11 model; fastest YOLO option on this Pi.',
  'yolo-small': '80 classes. Larger than Nano; slower detection.',
  'yolo-medium': '80 classes. Largest available model; expect slower detection.',
};

let initialized = false;
let displayedModel = null;
let submitting = false;
let latestState = null;

function showHint() {
  element('hint').textContent = modelHints[element('model').value] || '';
}

function showMetric(id, value) {
  element(id).textContent = Number.isFinite(value) ? value.toFixed(1) : '—';
}

function updateStatus(state) {
  latestState = state;
  if (!initialized) {
    for (const model of state.models) {
      const option = document.createElement('option');
      option.value = model.id;
      option.textContent = `${model.name} · ${model.class_count} classes`;
      if (!model.available) option.textContent += ' · not downloaded';
      option.disabled = !model.available;
      element('model').append(option);
    }
    initialized = true;
    element('model').disabled = false;
  }
  if (!submitting) {
    element('model').value = state.requested;
    if (document.activeElement !== element('confidence')) {
      element('confidence').value = Math.round(state.confidence * 100);
      element('threshold').textContent = `${Math.round(state.confidence * 100)}%`;
    }
    showHint();
  }

  element('connection').textContent = state.stale ? 'No recent detections' : 'Live';
  showMetric('frame-fps', state.frame_fps);
  showMetric('camera-fps', state.camera_fps);
  showMetric('preview-fps', state.preview_fps);
  showMetric('fps', state.pipeline_fps);
  showMetric('frame-latency', state.capture_to_output_ms);
  showMetric('latency', state.inference_ms);
  element('objects').textContent = state.objects;

  const model = state.models.find((item) => item.id === state.active);
  element('caption').textContent = `${model?.name || state.active} · ${state.classes.length} classes`;
  element('message').textContent = state.error || (state.switching ? 'Switching model…' : 'Model ready');
  element('message').classList.toggle('error', Boolean(state.error));
  if (displayedModel !== state.active) {
    const labels = state.classes.map((name) => {
      const chip = document.createElement('span');
      chip.textContent = name;
      return chip;
    });
    element('classes').replaceChildren(...labels);
    element('classes-title').textContent = `Detectable classes · ${state.classes.length}`;
    displayedModel = state.active;
  }
}

async function refreshStatus() {
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    if (!response.ok) throw new Error('Status unavailable');
    updateStatus(await response.json());
  } catch (error) {
    element('connection').textContent = 'Disconnected';
    element('message').textContent = 'Cannot reach the camera app. Retrying…';
    element('message').classList.add('error');
  }
}

async function applySettings() {
  submitting = true;
  showHint();
  element('message').textContent = 'Switching model…';
  try {
    const response = await fetch('/api/model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: element('model').value,
        confidence: Number(element('confidence').value) / 100,
      }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || 'Switch failed');
    await refreshStatus();
  } catch (error) {
    element('message').textContent = error.message;
    element('message').classList.add('error');
  } finally {
    submitting = false;
  }
}

function drawOverlay() {
  const image = element('camera-video');
  const canvas = element('overlay');
  if (!image.naturalWidth || !latestState) return;
  if (canvas.width !== image.naturalWidth || canvas.height !== image.naturalHeight) {
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
  }
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!latestState.browser_overlay) return;
  context.font = 'bold 18px system-ui';
  context.lineWidth = 2;
  for (const detection of latestState.detections || []) {
    const [x1, y1, x2, y2] = detection.box;
    context.strokeStyle = '#20df75';
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const text = `${detection.label} ${detection.confidence.toFixed(2)}`;
    const width = context.measureText(text).width + 12;
    const x = Math.max(0, Math.min(x1, canvas.width - width));
    const y = Math.max(25, y1 - 5);
    context.fillStyle = '#141414';
    context.fillRect(x, y - 23, width, 29);
    context.fillStyle = '#fff';
    context.fillText(text, x + 6, y);
  }
  const fps = Number.isFinite(latestState.frame_fps) ? latestState.frame_fps.toFixed(1) : '—';
  const text = latestState.switching
    ? 'Switching model…'
    : `Frame FPS ${fps} · ${latestState.active}`;
  const width = context.measureText(text).width + 16;
  context.fillStyle = '#141414';
  context.fillRect(6, 5, width, 29);
  context.fillStyle = '#fff';
  context.fillText(text, 14, 26);
}

async function pollStatus() {
  await refreshStatus();
  setTimeout(pollStatus, 200);
}

element('model').addEventListener('change', applySettings);
element('confidence').addEventListener('input', () => {
  element('threshold').textContent = `${element('confidence').value}%`;
});
element('confidence').addEventListener('change', applySettings);
pollStatus();
setInterval(drawOverlay, 33);
