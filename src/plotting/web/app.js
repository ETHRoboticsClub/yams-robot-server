const cfg = JSON.parse(document.getElementById('plot-config').textContent);
const keys = cfg.keys;
const labelMapBySource = cfg.labels;
const hz = cfg.hz;
const maxBufferPoints = cfg.maxBufferPoints;
const leftKeys = keys.filter((k) => k.startsWith('left_'));
const rightKeys = keys.filter((k) => k.startsWith('right_'));

const streamStatusEl = document.getElementById('stream-status');
const historyInput = document.getElementById('history-s');
const trajectoryToggle = document.getElementById('trajectory-toggle');
const controlResult = document.getElementById('control-result');
const followerStatusEl = document.getElementById('follower-status');
const leaderStatusEl = document.getElementById('leader-status');
const cameraStatusEl = document.getElementById('camera-status');
const cameraGridEl = document.getElementById('camera-grid');

let historyS = cfg.historyS;
let maxPoints = Math.max(8, Math.round(hz * historyS));
let trajectoryRecording = false;
historyInput.value = historyS.toFixed(0);

function setStreamStatus(text, ok) {
  streamStatusEl.textContent = text;
  streamStatusEl.classList.toggle('ok', ok);
  streamStatusEl.classList.toggle('bad', !ok);
}

class JointCard {
  constructor(parent, key, label, color) {
    this.key = key;
    this.color = color;
    this.points = [];
    this.el = document.createElement('div');
    this.el.className = 'card';
    this.el.innerHTML = `<div class="name">${label}</div><canvas width="600" height="220"></canvas><div class="val">value: <span>-</span></div>`;
    parent.appendChild(this.el);
    this.canvas = this.el.querySelector('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.valueEl = this.el.querySelector('span');
  }

  trim(nowT) {
    const tMin = nowT - historyS;
    while (this.points.length > maxBufferPoints || (this.points.length && this.points[0].t < tMin)) this.points.shift();
    while (this.points.length > maxPoints) this.points.shift();
  }

  add(t, value) {
    if (value === undefined || value === null) return;
    this.points.push({ t, value });
    this.trim(t);
    this.valueEl.textContent = Number(value).toFixed(3);
  }

  render() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = '#e2e8f0';
    for (let i = 0; i <= 4; i++) {
      const y = 10 + i * ((h - 20) / 4);
      ctx.beginPath();
      ctx.moveTo(10, y);
      ctx.lineTo(w - 10, y);
      ctx.stroke();
    }

    let maxAbs = 0.1;
    for (const p of this.points) maxAbs = Math.max(maxAbs, Math.abs(p.value));
    maxAbs *= 1.1;
    const minY = -maxAbs;
    const maxY = maxAbs;
    const axisY = 10 + (1 - (0 - minY) / Math.max(1e-6, maxY - minY)) * (h - 20);
    ctx.beginPath();
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1.5;
    ctx.moveTo(10, axisY);
    ctx.lineTo(w - 10, axisY);
    ctx.stroke();

    if (this.points.length < 2) return;
    const t0 = this.points[0].t;
    const t1 = this.points[this.points.length - 1].t;
    ctx.beginPath();
    ctx.strokeStyle = this.color;
    ctx.lineWidth = 2;
    let moved = false;
    for (const p of this.points) {
      const x = 10 + ((p.t - t0) / Math.max(1e-6, t1 - t0)) * (w - 20);
      const y = 10 + (1 - (p.value - minY) / Math.max(1e-6, maxY - minY)) * (h - 20);
      if (!moved) {
        ctx.moveTo(x, y);
        moved = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    if (moved) ctx.stroke();
  }
}

class ArmSection {
  constructor(prefix, sourceKey, color, statusEl) {
    this.sourceKey = sourceKey;
    this.statusEl = statusEl;
    this.labelMap = labelMapBySource[sourceKey] || {};
    this.cards = [];
    this.lastSeenMs = 0;
    this.name = sourceKey === 'obs' ? 'follower' : 'leader';

    const leftRoot = document.getElementById(`${prefix}-left`);
    const rightRoot = document.getElementById(`${prefix}-right`);
    for (const key of leftKeys) this.cards.push(new JointCard(leftRoot, key, this.labelMap[key] || key, color));
    for (const key of rightKeys) this.cards.push(new JointCard(rightRoot, key, this.labelMap[key] || key, color));
  }

  addPoint(msg) {
    const src = msg[this.sourceKey] || {};
    let hasValue = false;
    for (const card of this.cards) {
      const v = src[card.key];
      if (v !== undefined && v !== null) hasValue = true;
      card.add(msg.t, v);
    }
    if (hasValue) this.lastSeenMs = performance.now();
  }

  updateStatus(nowMs) {
    this.statusEl.textContent = this.lastSeenMs > 0 && nowMs - this.lastSeenMs <= 2000 ? 'connected' : `${this.name} not connected`;
  }

  render() {
    for (const card of this.cards) card.render();
  }

  trimAll() {
    for (const card of this.cards) {
      if (card.points.length) card.trim(card.points[card.points.length - 1].t);
    }
  }
}

class CameraCard {
  constructor(parent, key, label) {
    this.key = key;
    this.el = document.createElement('div');
    this.el.className = 'card camera-card';
    this.el.innerHTML = `<div class="name">${label}</div><img alt="${label}" />`;
    parent.appendChild(this.el);
    this.imgEl = this.el.querySelector('img');
  }

  setFrame(dataUrl) {
    this.imgEl.src = dataUrl;
  }
}

class CameraSection {
  constructor(parent, statusEl) {
    this.parent = parent;
    this.statusEl = statusEl;
    this.labelMap = labelMapBySource.cams || {};
    this.cards = new Map();
    this.lastSeenMs = 0;
  }

  addPoint(msg) {
    const cams = msg.cams || {};
    let hasFrame = false;
    for (const [key, frame] of Object.entries(cams)) {
      if (!frame) continue;
      if (!this.cards.has(key)) this.cards.set(key, new CameraCard(this.parent, key, this.labelMap[key] || key));
      this.cards.get(key).setFrame(frame);
      hasFrame = true;
    }
    if (hasFrame) this.lastSeenMs = performance.now();
  }

  updateStatus(nowMs) {
    this.statusEl.textContent = this.lastSeenMs > 0 && nowMs - this.lastSeenMs <= 2000 ? 'connected' : 'camera not connected';
  }
}

class Dashboard {
  constructor() {
    this.follower = new ArmSection('follower', 'obs', '#0284c7', followerStatusEl);
    this.leader = new ArmSection('leader', 'act', '#ef4444', leaderStatusEl);
    this.cameras = new CameraSection(cameraGridEl, cameraStatusEl);
  }

  addPoint(msg) {
    this.follower.addPoint(msg);
    this.leader.addPoint(msg);
    this.cameras.addPoint(msg);
  }

  trimAll() {
    this.follower.trimAll();
    this.leader.trimAll();
  }

  render() {
    const nowMs = performance.now();
    this.follower.updateStatus(nowMs);
    this.leader.updateStatus(nowMs);
    this.cameras.updateStatus(nowMs);
    this.follower.render();
    this.leader.render();
  }
}

function setTrajectoryState(recording) {
  trajectoryRecording = recording;
  trajectoryToggle.textContent = recording ? 'Stop Trajectory' : 'Start Trajectory';
  trajectoryToggle.classList.toggle('recording', recording);
}

async function sendControl(payload) {
  const res = await fetch('/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
}

const dashboard = new Dashboard();
const es = new EventSource('/events');
es.onopen = () => setStreamStatus('connected', true);
es.onerror = () => setStreamStatus('reconnecting', false);
es.onmessage = (evt) => dashboard.addPoint(JSON.parse(evt.data));

historyInput.onchange = () => {
  const next = Number(historyInput.value);
  if (!Number.isFinite(next) || next <= 0) {
    historyInput.value = historyS.toFixed(0);
    return;
  }
  historyS = next;
  maxPoints = Math.max(8, Math.round(hz * historyS));
  dashboard.trimAll();
};

trajectoryToggle.onclick = async () => {
  const nextRecording = !trajectoryRecording;
  const command = nextRecording ? 'start' : 'stop';
  try {
    await sendControl({ type: 'trajectory', command });
    setTrajectoryState(nextRecording);
    controlResult.textContent = nextRecording
      ? 'trajectory recording'
      : 'trajectory saved to trajectories/';
  } catch (err) {
    controlResult.textContent = `failed to ${command} trajectory: ${err}`;
  }
};
setTrajectoryState(false);

function tick() {
  dashboard.render();
  requestAnimationFrame(tick);
}
tick();
