import { $, el, fmt, api } from '/core/js/util.js';
import { createViewer, THREE } from '/core/js/viewer.js';
import { mountRight } from '/core/js/panel.js';
import { mountResults } from './results.js';

const COL = { domain: 0x8492a6, fluid: 0x4cc2ff, inlet: 0x57d9a3, outlet: 0xff6b6b, sel: 0xffc857 };
let st = null;
let sel = null;
let lastGeo = '';

const V = createViewer($('gl'), $('gizmo'));
const gSym = V.group('symbols');
const gPrev = V.group('preview');

function note(msg, err) {
  const m = $('solveMsg');
  m.textContent = msg || '';
  m.style.color = err ? 'var(--err)' : '';
}

function ops(list) {
  return api('/api/ops', { json: { ops: list } })
    .then(refresh)
    .catch((e) => {
      note(e.message, true);
      refresh(st);
    });
}

const right = mountRight($('right'), {
  theory: '/theory.html',
  placeholder: '예: x 0~500, y 0~300, z 0~400 mm 에 물 채우고, (250, 150, 900) mm 에서 아래로 2 m/s 로 유입',
  note,
  onState: async (s, c) => {
    await refresh(s);
    if (c.action === 'solve') $('solve').click();
  },
});

function setNum(id, v) {
  const e = $(id);
  if (document.activeElement !== e) e.value = v === null || v === undefined ? '' : +(+v).toPrecision(6);
}

function renderControls() {
  const s = st.spec;
  const tl = st.tools || {};
  $('tools').innerHTML = `taichi <b class="${tl.taichi ? 'on' : 'off'}">${tl.taichi || '없음'}</b><br>장치 <b class="${tl.devices && tl.devices.length ? 'on' : 'off'}">${tl.devices ? tl.devices.join(', ') || '없음' : '확인 중'}</b>`;
  $('fluidPreset').value = s.fluid.preset;
  setNum('rho0', s.fluid.rho0);
  setNum('mu', s.fluid.mu);
  setNum('dx', s.particle.dx);
  $('boundary').value = s.solver.boundary || 'mdbc';
  $('viscosity').value = s.solver.viscosity || 'artificial';
  setNum('alpha', s.solver.alpha);
  $('alpha').disabled = $('viscosity').value === 'laminar';
  $('device').value = s.device;
  setNum('tEnd', s.solver.t_end);
  setNum('fps', s.solver.fps);
  setNum('cfl', s.solver.cfl);
  setNum('gz', s.gravity[2]);
  ['dx0', 'dy0', 'dz0'].forEach((id, k) => setNum(id, s.domain.min[k]));
  ['dx1', 'dy1', 'dz1'].forEach((id, k) => setNum(id, s.domain.max[k]));
  const e = st.estimate;
  $('estimate').textContent = `예상 입자 ${e.total.toLocaleString()}개 (초기 ${e.fill.toLocaleString()} + 유입 ${e.emitted.toLocaleString()}) · 입자 질량 ${fmt(e.particle_mass)} tonne · 커널 반경 ${fmt(e.support_radius)} mm`;
  $('undo').disabled = !st.can_undo;
  if (st.model) {
    const m = st.model;
    const bb = m.bbox.map((p) => p.map((v) => fmt(v)).join(', '));
    $('modelInfo').textContent = `${m.source} · ${m.kind.toUpperCase()} (파일 단위 ${m.file_unit})\n정점 ${m.counts.vertices} · 면 ${m.counts.faces} · 선 ${m.counts.lines} · 힌지 엣지 ${m.counts.hinge_edges} · 덩어리 ${m.counts.bodies}\n${m.kind === 'stl' ? (m.closed ? '닫힌 표면' : `열린 표면 (열린 엣지 ${m.open_edges})`) + '\n' : ''}범위 (${bb[0]}) ~ (${bb[1]}) mm`;
  } else $('modelInfo').textContent = '불러온 모델이 없습니다. 형상 없이 유체 블록만으로도 설정할 수 있습니다';
}

function item(list, it, title, desc, edits) {
  const d = el('div', 'item' + (sel === it.id ? ' sel' : ''));
  const top = el('div', 'top');
  const nm = el('span');
  nm.append(el('span', 'id', it.id || ''), document.createTextNode(title));
  top.append(nm);
  if (it.id) {
    const x = el('button', 'x', '✕');
    x.onclick = (e) => {
      e.stopPropagation();
      ops([{ op: 'remove', id: it.id }]);
    };
    top.append(x);
  }
  d.append(top, el('div', 'desc', desc));
  if (edits) d.append(edits);
  d.onclick = () => {
    sel = sel === it.id ? null : it.id;
    renderLists();
    buildSymbols();
  };
  list.append(d);
}

function vecEdit(id, key, vals, label) {
  const ed = el('div', 'edit');
  if (label) ed.append(el('span', 'lab', label));
  const ins = vals.map((v) => {
    const i = el('input');
    i.type = 'number';
    i.step = 'any';
    i.value = +(+v).toPrecision(6);
    i.onclick = (e) => e.stopPropagation();
    return i;
  });
  ins.forEach((i) => (i.onchange = () => ops([{ op: 'update', id, [key]: ins.map((q) => +q.value || 0) }])));
  ed.append(...ins);
  return ed;
}

function boxEdit(it) {
  const w = el('div');
  const ed0 = vecEdit(it.id, 'box', it.box.min, '최소');
  const ed1 = vecEdit(it.id, 'box', it.box.max, '최대');
  const send = () => {
    const a = [...ed0.querySelectorAll('input')].map((q) => +q.value || 0);
    const b = [...ed1.querySelectorAll('input')].map((q) => +q.value || 0);
    ops([{ op: 'update', id: it.id, box: { min: a, max: b } }]);
  };
  [...ed0.querySelectorAll('input'), ...ed1.querySelectorAll('input')].forEach((i) => (i.onchange = send));
  w.append(ed0, ed1);
  return w;
}

function renderLists() {
  const s = st.spec;
  const L = (id, cid, arr, empty) => {
    const b = $(id);
    b.innerHTML = '';
    $(cid).textContent = arr.length ? `${arr.length}개` : '';
    if (!arr.length) b.append(el('div', 'empty', empty));
    return b;
  };
  const bl = L('bdList', 'bdCount', s.bodies, st.model ? '그룹 없음' : '형상을 올리면 그룹별로 표시됩니다');
  for (const b of s.bodies) {
    const ed = el('div', 'edit');
    const sl = el('select');
    sl.innerHTML = '<option value="wall">벽 (경계 입자)</option><option value="ignore">무시</option>';
    sl.value = b.role;
    sl.onclick = (e) => e.stopPropagation();
    sl.onchange = () => ops([{ op: 'set_body', group: b.group, role: sl.value }]);
    ed.append(el('span', 'lab', '역할'), sl);
    const g = st.model && st.model.groups[b.group];
    item(bl, { id: '' }, b.group, g ? `면 ${g.faces} · 선 ${g.lines} · 고정` : '', ed);
  }
  const fl = L('fbList', 'fbCount', s.fluid_blocks, '없음. 오른쪽에서 "x 0~500, y 0~300, z 0~400 mm 에 물 채워줘"');
  for (const b of s.fluid_blocks) {
    const sz = b.box.max.map((v, k) => v - b.box.min[k]);
    item(fl, b, b.name, `${sz.map(fmt).join(' × ')} mm`, boxEdit(b));
  }
  const il = L('inList', 'inCount', s.inlets, '없음');
  for (const it of s.inlets) {
    const w = el('div');
    w.append(vecEdit(it.id, 'center', it.center, '중심'), vecEdit(it.id, 'normal', it.normal, '방향'));
    const ed = el('div', 'edit');
    const sp = el('input');
    sp.type = 'number';
    sp.step = 'any';
    sp.value = it.speed;
    sp.onclick = (e) => e.stopPropagation();
    sp.onchange = () => ops([{ op: 'update', id: it.id, speed: +sp.value || 0 }]);
    ed.append(el('span', 'lab', '속력'), sp);
    w.append(ed);
    item(il, it, it.name, `${it.shape === 'circle' ? '원형 지름 ' + fmt(it.size[0]) : '사각 ' + it.size.map(fmt).join(' × ')} mm · ${fmt(it.speed)} mm/s · ${fmt(it.t_start)}~${it.t_end === null ? '끝' : fmt(it.t_end)} s`, w);
  }
  const ol = L('outList', 'outCount', s.outlets, '없음');
  for (const b of s.outlets) item(ol, b, b.name, '이 영역에 들어간 입자는 제거', boxEdit(b));
  const ml = L('moList', 'moCount', s.monitors, '없음. 좌표로 추가하면 속도·압력 시계열이 그려집니다');
  for (const m of s.monitors) item(ml, m, m.name, `(${m.point.map(fmt).join(', ')}) mm`, vecEdit(m.id, 'point', m.point, '위치'));
  const sl = L('stList', 'stCount', s.stops, '없음');
  for (const t of s.stops) item(sl, t, `${t.quantity} ${t.cmp} ${fmt(t.value)}`, '조건을 만족하면 일시정지');
  const w = $('warnings');
  w.innerHTML = '';
  for (const x of st.warnings) w.append(el('div', x.level === 'err' ? 'err' : x.level === 'info' ? 'info' : '', (x.level === 'info' ? 'ℹ ' : '⚠ ') + x.text));
}

function patch(it) {
  const n = new THREE.Vector3(...it.normal).normalize();
  const a = Math.abs(n.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
  const u = new THREE.Vector3().crossVectors(n, a).normalize();
  const v = new THREE.Vector3().crossVectors(n, u);
  const c = new THREE.Vector3(...it.center);
  const pts = [];
  if (it.shape === 'circle') {
    for (let k = 0; k <= 48; k++) {
      const t = (k / 48) * Math.PI * 2;
      pts.push(c.clone().addScaledVector(u, (Math.cos(t) * it.size[0]) / 2).addScaledVector(v, (Math.sin(t) * it.size[0]) / 2));
    }
  } else {
    for (const [p, q] of [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) pts.push(c.clone().addScaledVector(u, (p * it.size[0]) / 2).addScaledVector(v, (q * it.size[1]) / 2));
  }
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: sel === it.id ? COL.sel : COL.inlet, depthTest: false }));
}

function buildSymbols() {
  V.clear(gSym);
  const s = st.spec;
  gSym.add(V.boxLines(s.domain.min, s.domain.max, COL.domain, true));
  for (const b of s.fluid_blocks) gSym.add(V.boxLines(b.box.min, b.box.max, sel === b.id ? COL.sel : COL.fluid));
  for (const b of s.outlets) gSym.add(V.boxLines(b.box.min, b.box.max, sel === b.id ? COL.sel : COL.outlet));
  const L = V.diag * 0.12;
  for (const m of s.monitors) gSym.add(V.points(m.point, [1, 0.78, 0.34], 9));
  for (const it of s.inlets) {
    gSym.add(patch(it));
    gSym.add(V.arrow(it.center, it.normal, L, sel === it.id ? COL.sel : COL.inlet));
  }
  applyVisibility();
}

async function buildPreview() {
  V.clear(gPrev);
  if (!$('showPreview').checked || !st.spec.fluid_blocks.length) return;
  const p = await api('/api/preview');
  if (p.points.length) gPrev.add(V.points(p.points.flat(), null, 3));
}

function applyVisibility() {
  V.group('model').visible = $('showModel').checked;
  V.group('edges').visible = $('showModel').checked && $('showEdges').checked;
  gSym.visible = $('showSymbols').checked;
  gPrev.visible = $('showPreview').checked;
}

function sceneBox() {
  const s = st.spec;
  const lo = [...s.domain.min];
  const hi = [...s.domain.max];
  if (st.model) for (let k = 0; k < 3; k++) {
    lo[k] = Math.min(lo[k], st.model.bbox[0][k]);
    hi[k] = Math.max(hi[k], st.model.bbox[1][k]);
  }
  return [lo, hi];
}

async function refresh(s) {
  const prev = st;
  st = s;
  if (st.view === undefined && prev) st.view = prev.view;
  renderControls();
  const key = st.model ? `${st.model.source}|${st.model.counts.vertices}|${st.model.counts.faces}` : '';
  if (key !== lastGeo) {
    lastGeo = key;
    V.drawGeometry(st.model ? st.view : null, { opacity: 0.35 });
    const [lo, hi] = sceneBox();
    V.fitBox(lo, hi);
  } else if (!prev) {
    const [lo, hi] = sceneBox();
    V.fitBox(lo, hi);
  }
  renderLists();
  buildSymbols();
  await buildPreview();
  applyVisibility();
}

async function loadExamples() {
  const list = await api('/api/examples');
  for (const e of list) {
    const o = new Option(e.title, e.name);
    o.dataset.note = e.note || '';
    o.dataset.prompts = JSON.stringify(e.prompts || []);
    $('example').append(o);
  }
}

async function openExample(preset) {
  const sel = $('example');
  if (!sel.value) return;
  note('불러오는 중…');
  try {
    const s = await api(`/api/examples/${sel.value}?preset=${preset}`, { method: 'POST' });
    lastGeo = '-';
    right.clear();
    const o = sel.selectedOptions[0];
    right.chips(JSON.parse(o.dataset.prompts));
    await refresh(s);
    note(o.dataset.note);
  } catch (e) {
    note(e.message, true);
  }
}

function bind() {
  $('loadPreset').onclick = () => openExample(1);
  $('loadGeo').onclick = () => openExample(0);
  $('file').onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append('file', f);
    fd.append('unit', $('fileUnit').value);
    note('불러오는 중…');
    try {
      const s = await api('/api/upload', { method: 'POST', body: fd });
      right.clear();
      right.chips(['내용에 맞춰 해석 영역 맞춰줘', '입자 간격 10 mm', '해석 시간 2초']);
      await refresh(s);
      note(`${f.name} 을(를) 불러왔습니다. 단위 ${$('fileUnit').value} → mm`);
    } catch (err) {
      note(err.message, true);
    }
    e.target.value = '';
  };
  $('fluidPreset').onchange = (e) => (e.target.value === 'custom' ? null : ops([{ op: 'set_fluid', preset: e.target.value }]));
  $('rho0').onchange = (e) => ops([{ op: 'set_fluid', rho0: +e.target.value }]);
  $('mu').onchange = (e) => ops([{ op: 'set_fluid', mu: +e.target.value }]);
  $('dx').onchange = (e) => ops([{ op: 'set_particle', dx: +e.target.value }]);
  $('boundary').onchange = (e) => ops([{ op: 'set_solver', boundary: e.target.value }]);
  $('viscosity').onchange = (e) => ops([{ op: 'set_solver', viscosity: e.target.value }]);
  $('alpha').onchange = (e) => ops([{ op: 'set_solver', alpha: +e.target.value }]);
  $('prAdd').onclick = () => {
    const p = ['prX', 'prY', 'prZ'].map((i) => +$(i).value);
    if (p.some((v) => !Number.isFinite(v))) return note('프로브 좌표를 입력해 주세요', true);
    ops([{ op: 'add_probe', point: p }]);
  };
  $('device').onchange = (e) => ops([{ op: 'set_device', device: e.target.value }]);
  $('tEnd').onchange = (e) => ops([{ op: 'set_solver', t_end: +e.target.value }]);
  $('fps').onchange = (e) => ops([{ op: 'set_solver', fps: +e.target.value }]);
  $('cfl').onchange = (e) => ops([{ op: 'set_solver', cfl: +e.target.value }]);
  $('gz').onchange = (e) => ops([{ op: 'set_gravity', vector: [st.spec.gravity[0], st.spec.gravity[1], +e.target.value || 0] }]);
  const dom = () => ops([{ op: 'set_domain', min: ['dx0', 'dy0', 'dz0'].map((i) => +$(i).value), max: ['dx1', 'dy1', 'dz1'].map((i) => +$(i).value) }]);
  ['dx0', 'dy0', 'dz0', 'dx1', 'dy1', 'dz1'].forEach((i) => ($(i).onchange = dom));
  $('fitDomain').onclick = () => ops([{ op: 'set_domain', fit: true }]);
  $('undo').onclick = async () => {
    try {
      await refresh(await api('/api/undo', { method: 'POST' }));
    } catch (e) {
      note(e.message, true);
    }
  };
  $('impSpec').onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      await refresh(await api('/api/import/spec', { json: { spec: JSON.parse(await f.text()) } }));
      note(`${f.name} 케이스를 불러왔습니다`);
    } catch (err) {
      note(err.message, true);
    }
    e.target.value = '';
  };
  $('fit').onclick = () => {
    const [lo, hi] = sceneBox();
    V.fitBox(lo, hi);
  };
  for (const id of ['showModel', 'showEdges', 'showSymbols']) $(id).onchange = applyVisibility;
  $('showPreview').onchange = () => buildPreview().then(applyVisibility);
}

async function start() {
  bind();
  mountResults(V, { note });
  V.start();
  await loadExamples();
  await refresh(await api('/api/state'));
  api('/api/engine')
    .then(() => api('/api/state'))
    .then((s) => {
      st.tools = s.tools;
      renderControls();
    })
    .catch(() => {});
}

start().catch((e) => note(e.message, true));
