const cfg = JSON.parse(document.getElementById('plot-config').textContent);
const keys = cfg.keys;
const labelMapBySource = cfg.labels;
const hz = cfg.hz;
const maxBufferPoints = cfg.maxBufferPoints;
const taskGoals = cfg.taskGoals || {};
const leftKeys = keys.filter((k) => k.startsWith('left_'));
const rightKeys = keys.filter((k) => k.startsWith('right_'));

const streamStatusEl = document.getElementById('stream-status');
const historyInput = document.getElementById('history-s');
const taskSelect = document.getElementById('task-select');
const recordBtn = document.getElementById('record-btn');
const controlResult = document.getElementById('control-result');
const refreshTreeBtn = document.getElementById('refresh-tree');
const trajTreeEl = document.getElementById('traj-tree');
const followerStatusEl = document.getElementById('follower-status');
const leaderStatusEl = document.getElementById('leader-status');
const cameraStatusEl = document.getElementById('camera-status');
const cameraGridEl = document.getElementById('camera-grid');

let historyS = cfg.historyS;
let maxPoints = Math.max(8, Math.round(hz * historyS));
historyInput.value = historyS.toFixed(0);

for (const task of cfg.tasks || []) {
  const opt = document.createElement('option');
  opt.value = task;
  opt.textContent = task;
  taskSelect.appendChild(opt);
}

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

let recording = false;

recordBtn.onclick = async () => {
  recording = !recording;
  recordBtn.classList.toggle('recording', recording);
  recordBtn.textContent = recording ? '\u25A0 Stop Recording' : '\u25CF Start Recording';
  try {
    await fetch('/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: 'trajectory', action: recording ? 'start' : 'stop', task: taskSelect.value }),
    });
    if (recording) {
      controlResult.textContent = `Recording (${taskSelect.value})…`;
    } else {
      controlResult.textContent = 'Saved.';
      loadTree();
    }
  } catch (err) {
    controlResult.textContent = `failed: ${err}`;
  }
};

async function loadTree() {
  const task = taskSelect.value;
  if (!task) { trajTreeEl.innerHTML = ''; return; }
  try {
    const res = await fetch('/trajectories');
    const tree = await res.json();
    trajTreeEl.innerHTML = '';

    const eps = tree[task] || [];
    const goodCount = eps.filter(e => !e.marked_bad).length;
    const goal = taskGoals[task] ?? null;

    // Progress bar
    if (goal !== null) {
      const pct = Math.min(100, Math.round((goodCount / goal) * 100));
      const progWrap = document.createElement('div');
      progWrap.className = 'traj-progress';
      progWrap.innerHTML = `
        <div class="traj-progress-label">
          <span>${goodCount} / ${goal} episodes</span>
          <span>${pct}%</span>
        </div>
        <div class="traj-progress-bar"><div class="traj-progress-fill" style="width:${pct}%"></div></div>
      `;
      trajTreeEl.appendChild(progWrap);
    } else {
      const countEl = document.createElement('div');
      countEl.className = 'traj-count';
      countEl.textContent = `${goodCount} episode${goodCount !== 1 ? 's' : ''} collected`;
      trajTreeEl.appendChild(countEl);
    }

    if (!eps.length) {
      const empty = document.createElement('div');
      empty.className = 'traj-empty';
      empty.textContent = 'No episodes yet.';
      trajTreeEl.appendChild(empty);
      return;
    }

    const ul = document.createElement('ul');
    ul.className = 'traj-ep-list';
    for (const ep of [...eps].reverse()) {
      const li = document.createElement('li');
      li.className = 'traj-ep' + (ep.marked_bad ? ' bad' : '') + (ep.session ? ' session' : '');
      const nameSpan = document.createElement('span');
      nameSpan.className = 'ep-name';
      nameSpan.textContent = `episode ${ep.name}`;
      li.appendChild(nameSpan);
      const btn = document.createElement('button');
      btn.className = 'mark-bad-btn';
      btn.textContent = ep.marked_bad ? 'restore' : '\u2717';
      btn.onclick = async () => {
        await fetch('/mark_bad', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task, episode: ep.name, bad: !ep.marked_bad }),
        });
        loadTree();
      };
      li.appendChild(btn);
      ul.appendChild(li);
    }
    trajTreeEl.appendChild(ul);
  } catch (err) {
    trajTreeEl.textContent = `Error: ${err}`;
  }
}

refreshTreeBtn.onclick = loadTree;
taskSelect.addEventListener('change', loadTree);
loadTree();

function tick() {
  dashboard.render();
  requestAnimationFrame(tick);
}
tick();
