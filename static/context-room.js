// Context-aware behaviour room: renders the scene metadata, sends utterances to
// /api/interpret and executes the planner's step list on the shared renderer.
// Renderer features (lookAt, reachTo, sit, lie, stand, setFacing, setExpression)
// are feature-checked so an older vendored avatar.js still runs the walk/state path.
import * as THREE from '/static/vendor/three.module.js';
import {createStage} from '/static/avatar.js?v=20261006-paper5';
import {Speech} from '/static/speech.js?v=20261006-paper5';
import {setupVoiceInput} from '/static/voice-input.js?v=20261006-paper5';
import {prepareApplicationMotion, gestureSummary} from '/static/application-gesture.js?v=20261006-paper5';

const $ = selector => document.querySelector(selector);
const stage = createStage($('#room'));
const speech = new Speech(stage);
const has = name => typeof stage[name] === 'function';
const rad = degrees => degrees * Math.PI / 180;
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const room = {objects: {}, meta: {}, held: null, run: 0, lights: {}};
let current = null;

const speechStatus = $('#speech-status');
const stopRecording = setupVoiceInput(speech, $('#utterance'), $('#record'), $('#audio-file'), speechStatus);
fetch('/api/speech').then(r => r.json()).then(v => { if (!v.asr) speechStatus.textContent = 'Local transcription requires WHISPER_MODEL_DIR; typed utterances work now.'; });
window.addEventListener('pagehide', () => { stopRecording(); speech.cancel(); stage.dispose(); });

// ---------------------------------------------------------------- room geometry
function box(size, color, options = {}) {
  const material = new THREE.MeshStandardMaterial({color, roughness: .8, ...options});
  return new THREE.Mesh(new THREE.BoxGeometry(...size), material);
}

function buildRoom(scene) {
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(6.4, 5.2), new THREE.MeshStandardMaterial({color: 0x3b4b57, roughness: .95}));
  floor.rotation.x = -Math.PI / 2; floor.position.set(0, .001, -.4);
  const wall = box([6.4, 2.8, .1], 0x51606b); wall.position.set(0, 1.4, -2.65);
  const side = box([.1, 2.8, 5.2], 0x485661); side.position.set(-3.2, 1.4, -.4);
  room.lights.ceiling = new THREE.PointLight(0xfff3dd, 1.4, 12, 1.2); room.lights.ceiling.position.set(0, 2.7, -.6);
  stage.scene.add(floor, wall, side, room.lights.ceiling);
  stage.camera.position.set(0, 2.55, 6.1); stage.camera.lookAt(0, .75, -.7);
  for (const o of scene.objects) room.objects[o.id] = makeObject(o);
}

function makeObject(o) {
  const group = new THREE.Group();
  const [w, h, d] = o.size; const color = new THREE.Color(o.color || '#aaaaaa');
  const parts = {};
  if (o.class === 'Chair') {
    const seat = box([w, .08, d], color); seat.position.y = (o.seat_height || .46) - .04;
    const back = box([w, .5, .06], color); back.position.set(0, (o.seat_height || .46) + .25, -d / 2 + .03);
    const legs = box([w * .9, (o.seat_height || .46) - .08, d * .9], color.clone().multiplyScalar(.6)); legs.position.y = ((o.seat_height || .46) - .08) / 2;
    group.add(seat, back, legs);
  } else if (o.class === 'Bed') {
    const base = box([w, h * .6, d], 0x6b4f3a); base.position.y = h * .3;
    const mattress = box([w * .96, h * .4, d * .96], color); mattress.position.y = h * .8;
    const head = box([w, .9, .08], 0x6b4f3a); head.position.set(0, .45, -d / 2 - .04);
    group.add(base, mattress, head);
  } else if (o.class === 'Drawer') {
    const body = box([w, h, d], color); body.position.y = h / 2;
    const drawer = box([w * .86, h * .3, d * .9], color.clone().multiplyScalar(.85)); drawer.position.set(0, h * .7, .03);
    const knob = box([.12, .04, .04], 0xe8d7a8); knob.position.set(0, 0, d * .45 + .02); drawer.add(knob);
    group.add(body, drawer); parts.drawer = drawer;
  } else if (o.class === 'Lamp') {
    const pole = box([.05, h - .3, .05], 0x9a9184); pole.position.y = (h - .3) / 2;
    const foot = box([w, .04, d], 0x9a9184); foot.position.y = .02;
    const shade = box([w, .3, d], color, {emissive: 0x000000}); shade.position.y = h - .15;
    const light = new THREE.PointLight(0xffd27a, 0, 4.5, 1.5); light.position.y = h - .3;
    group.add(pole, foot, shade, light); parts.shade = shade; parts.light = light;
  } else if (o.class === 'Window') {
    const frame = box([w + .1, h + .1, .05], 0xe9e3d8); frame.position.y = h / 2;
    const pane = box([w * .92, h * .46, .03], color, {transparent: true, opacity: .55, emissive: 0x15303a});
    pane.position.set(0, h * .26, .03);
    const top = pane.clone(); top.position.set(0, h * .74, .015);
    group.add(frame, pane, top); parts.pane = pane;
  } else if (o.class === 'Curtain') {
    for (const sideX of [-1, 1]) {
      const panel = box([w / 2, h, d], color); panel.position.set(sideX * w / 4, h / 2, 0);
      group.add(panel); parts[sideX < 0 ? 'left' : 'right'] = panel;
    }
  } else if (o.class === 'Switch') {
    const plate = box([w, h, d], color); plate.position.y = h / 2;
    const toggle = box([w * .35, h * .4, .03], 0x7a7466); toggle.position.set(0, h / 2, d / 2 + .01);
    group.add(plate, toggle); parts.toggle = toggle;
  } else {
    const body = box([w, h, d], color); body.position.y = h / 2; group.add(body);
  }
  group.position.set(o.position[0], o.elevation || 0, o.position[1]);
  group.rotation.y = rad(o.facing || 0);
  group.userData.id = o.id;
  stage.scene.add(group);
  room.meta[o.id] = o;
  return {group, parts};
}

function tween(ms, update) {
  return new Promise(resolve => {
    const start = performance.now();
    const step = now => { const k = Math.min(1, (now - start) / ms); update(k * k * (3 - 2 * k)); if (k < 1) requestAnimationFrame(step); else resolve(); };
    requestAnimationFrame(step);
  });
}

function applyState(id, state, animate = true) {
  const o = room.meta[id], entry = room.objects[id]; if (!o || !entry) return Promise.resolve();
  const p = entry.parts, ms = animate ? 600 : 1;
  if (o.class === 'Drawer' && 'open' in state) { const from = p.drawer.position.z, to = state.open ? .38 : .03; return tween(ms, k => { p.drawer.position.z = from + (to - from) * k; }); }
  if (o.class === 'Window' && 'open' in state) { const from = p.pane.position.y, to = state.open ? o.size[1] * .7 : o.size[1] * .26; return tween(ms, k => { p.pane.position.y = from + (to - from) * k; }); }
  if (o.class === 'Curtain' && 'open' in state) {
    const w = o.size[0], to = state.open ? .3 : 1, fromScale = p.left.scale.x;
    return tween(ms, k => { const s = fromScale + (to - fromScale) * k; for (const [panel, sign] of [[p.left, -1], [p.right, 1]]) { panel.scale.x = s; panel.position.x = sign * (w / 2 - s * w / 4); } });
  }
  if (o.class === 'Lamp' && 'on' in state) { p.shade.material.emissive.set(state.on ? 0xffc864 : 0x000000); p.light.intensity = state.on ? 2.2 : 0; }
  if (o.class === 'Switch' && 'on' in state) { p.toggle.rotation.x = state.on ? -.35 : .35; room.lights.ceiling.intensity = state.on ? 1.4 : .05; }
  if (state.location && !state.held) { const g = entry.group; g.position.set(state.location[0], state.location[1] - o.size[1] / 2, state.location[2]); }
  return Promise.resolve();
}

// ---------------------------------------------------------------- avatar helpers
const root = () => stage.avatar.root;
function handPoint(side) {
  const bone = stage.avatar.rig?.bones?.get(side === 'left' ? 'lefthand' : 'righthand');
  if (bone) return bone.getWorldPosition(new THREE.Vector3());
  const r = root(); const local = new THREE.Vector3(side === 'left' ? .28 : -.28, .95, .3);
  return r.localToWorld(local);
}
function followHeld() {
  if (room.held) {
    const {id, side} = room.held, entry = room.objects[id], o = room.meta[id];
    const p = handPoint(side); entry.group.position.set(p.x, p.y - o.size[1] / 2 - .02, p.z);
    entry.group.rotation.y = root().rotation.y;
  }
  requestAnimationFrame(followHeld);
}
requestAnimationFrame(followHeld);

function faceTo(yawDegrees, ms = 350) {
  const r = root(); const from = r.rotation.y; let to = rad(yawDegrees);
  while (to - from > Math.PI) to -= 2 * Math.PI; while (to - from < -Math.PI) to += 2 * Math.PI;
  return tween(ms, k => { r.rotation.y = from + (to - from) * k; });
}
function release() { if (has('reachTo')) stage.reachTo('auto', null); if (has('clearReach')) stage.clearReach(); stage.gesture('idle'); }
function placeAgent(agent) {
  const r = root(); r.position.set(agent.position[0], 0, agent.position[1]); r.rotation.y = rad(agent.facing || 0);
  if (has('stand')) stage.stand();
}

async function execute(steps, run) {
  for (const step of steps) {
    if (run !== room.run) return;
    const o = step.object ? room.meta[step.object] : null;
    switch (step.op) {
      case 'stand': if (has('stand')) stage.stand(); else if (has('setPosture')) stage.setPosture(null); stage.gesture('idle'); await wait(500); break;
      case 'walk': {
        const r = root(), distance = Math.hypot(step.to[0] - r.position.x, step.to[1] - r.position.z);
        await faceTo(step.facing, 250);
        const seconds = Math.max(.3, distance / step.speed);
        stage.gesture(step.gait === 'run' ? 'run' : 'walk');
        stage.moveTo(step.to[0], step.to[1], seconds);
        await wait(seconds * 1000 + 60); stage.gesture('idle'); break;
      }
      case 'face': await faceTo(step.yaw); break;
      case 'look': {
        const target = o ? room.objects[o.id].group : step.point ? new THREE.Vector3(...step.point) : null;
        if (has('lookAt') && target) stage.lookAt(target, .9); break;
      }
      case 'reach':
        if (has('reachTo')) stage.reachTo(step.side || 'auto', step.point, {weight: 1}); else stage.pointAt(step.point);
        await wait(750); break;
      case 'release': release(); await wait(250); break;
      case 'state': await applyState(step.object, step.state); break;
      case 'attach': room.held = {id: step.object, side: step.side === 'left' ? 'left' : 'right'}; await wait(200); break;
      case 'place': {
        const entry = room.objects[step.object], meta = room.meta[step.object]; room.held = null;
        const from = entry.group.position.clone(), to = new THREE.Vector3(step.point[0], step.point[1] - meta.size[1] / 2, step.point[2]);
        await tween(450, k => entry.group.position.lerpVectors(from, to, k)); break;
      }
      case 'offer': if (has('reachTo')) stage.reachTo(room.held?.side || 'right', step.point, {weight: 1}); else stage.gesture('offer'); await wait(1200); break;
      case 'sit':
        if (has('sit')) stage.sit({seatHeight: step.seat_height, position: step.position, facing: rad(step.facing)});
        else { root().position.set(step.position[0], 0, step.position[1]); stage.gesture('sit'); }
        await wait(700); break;
      case 'lie':
        if (has('lie')) stage.lie({surfaceHeight: step.surface_height, position: step.position, headDirection: [step.head_direction[0], 0, step.head_direction[1]]});
        else { root().position.set(step.position[0], 0, step.position[1]); stage.gesture('lie'); }
        await wait(900); break;
      case 'idle': release(); if (has('lookAt')) stage.lookAt(null); await wait(300); break;
    }
  }
  if (has('lookAt')) stage.lookAt('camera', .6);
}

// ---------------------------------------------------------------- UI
function renderObjects(states) {
  const list = $('#objects'); list.replaceChildren();
  for (const o of current.scene.objects) {
    const s = states.objects[o.id] || {}, button = document.createElement('button');
    const label = 'open' in s ? (s.open ? 'open' : 'closed') : 'on' in s ? (s.on ? 'on' : 'off') : s.held ? 'held' : s.inside ? 'in ' + s.inside : s.resting_on ? 'on ' + s.resting_on : '';
    button.textContent = `${o.name}${label ? ' · ' + label : ''}`;
    button.onclick = () => {
      $('#inspect').textContent = `${o.name} — class ${o.class}; position (${o.position.join(', ')}) m; facing ${o.facing}°\n` +
        Object.entries(o.affordances).map(([a, ps]) => `${a}: ${ps.join(', ')}`).join('\n') + '\nWalk/Run (None, To, Left, Right) and Stand up are implicit.';
    };
    list.append(button);
  }
  const a = states.agent;
  $('#agent').textContent = `Virtual human: ${a.posture}${a.seat ? ' on ' + a.seat : ''}${a.holding ? ', holding ' + a.holding : ''}`;
}

function renderResult(v) {
  const p = v.prediction, d = v.dispatch, conf = p.confidence || {};
  const cells = ['subject', 'action', 'position', 'target'].map(h => `<td>${p[h]}${conf[h] != null ? ` <small>${Math.round(conf[h] * 100)}%</small>` : ''}</td>`).join('');
  let behaviour;
  if (d.route === 'conversation') behaviour = `Conversation → ${v.dialogue?.client || 'dialogue'}${v.dialogue?.fallback ? ' (fixed fallback)' : ''}: “${d.reply}”`;
  else if (d.accepted) behaviour = `${d.command.action} · ${d.command.position} · ${d.command.target} → <b>${d.command.behavior}</b>${d.command.object ? ' ' + d.command.object : ''}`;
  else behaviour = `Rejected: ${d.reason}`;
  $('#result').innerHTML = `<table><tr><th>Subject</th><th>Action</th><th>Position</th><th>Target</th></tr><tr>${cells}</tr></table>` +
    `<p class="behaviour ${d.accepted ? '' : 'rejected'}">${behaviour}</p>` +
    (d.notes?.length ? `<p class="notes">${d.notes.join('; ')}</p>` : '') +
    (d.steps?.length ? `<details><summary>${d.steps.length} renderer steps</summary><pre>${d.steps.map(s => JSON.stringify(s)).join('\n')}</pre></details>` : '');
}

function say(text, extra = {}) {
  return speech.speak(text, {backend: $('#speech-backend').value, ...extra}).catch(e => { speechStatus.textContent = e.message; });
}

async function converse(reply) {
  let selected = null;
  try { selected = await prepareApplicationMotion(stage, reply, {mode: 'automatic'}); }
  catch (error) { speechStatus.textContent = `Recorded co-speech unavailable: ${error.message}`; }
  await speech.speak(reply, {backend: $('#speech-backend').value,
    onStart: () => { if (selected) { selected.motion.onStart(); $('#result').insertAdjacentHTML('beforeend', `<p class="notes">Co-speech: ${gestureSummary(selected.data)}</p>`); } },
    onProgress: clock => selected?.motion.onProgress(clock),
    onEnd: () => { selected?.motion.onEnd(); stage.clearMotion(); stage.gesture('idle'); }})
    .catch(e => { speechStatus.textContent = `${e.message}. Playing motion without speech.`; selected?.motion.playSilent(); });
}

async function interpret(text) {
  const response = await fetch('/api/interpret', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})});
  const v = await response.json();
  if (v.error) { $('#result').textContent = v.error; return; }
  renderResult(v);
  const run = ++room.run, d = v.dispatch;
  if (d.route === 'conversation') { await converse(d.reply); }
  else if (d.accepted) {
    if (has('setExpression')) stage.setExpression('happy', 1);
    say('Okay, ' + d.reply + '.');
    await execute(d.steps, run);
  } else {
    if (has('setExpression')) stage.setExpression('sad', 1);
    say("Sorry, I can't do that here.");
  }
  if (run === room.run) { renderObjects(v.states); if (has('setExpression')) setTimeout(() => stage.setExpression('neutral', 1), 1500); }
}

async function load(path = '/api/scene', options = {}) {
  const v = await fetch(path, options).then(r => r.json());
  const first = !current; current = v;
  if (first) buildRoom(v.scene);
  room.run++; room.held = null; release();
  for (const o of v.scene.objects) {
    const entry = room.objects[o.id]; entry.group.position.set(o.position[0], o.elevation || 0, o.position[1]); entry.group.rotation.y = rad(o.facing || 0);
    applyState(o.id, v.states.objects[o.id] || {}, false);
  }
  placeAgent(v.states.agent);
  renderObjects(v.states);
  $('#backend').textContent = `Interpreter: ${v.backend}. Dialogue: ${v.dialogue}.`;
}

$('#send').onclick = () => interpret($('#utterance').value);
$('#utterance').addEventListener('keydown', event => { if (event.key === 'Enter') interpret($('#utterance').value); });
$('#reset').onclick = () => load('/api/reset', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
for (const chip of document.querySelectorAll('[data-example]')) chip.onclick = () => { $('#utterance').value = chip.dataset.example; interpret(chip.dataset.example); };
stage.ready?.then?.(() => placeAgent(current?.states.agent || {position: [0, 0], facing: 0}));
load();
window.contextRoom = {stage, room, interpret, execute};
