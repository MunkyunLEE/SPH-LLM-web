import { $, el, fmt, api } from '/core/js/util.js';
import { THREE, setLegend } from '/core/js/viewer.js';
import { plotCard } from '/core/js/chart.js';
import * as F from '/core/js/fields.js';

const STATE_TEXT = { idle: '대기', building: '준비 중', running: '계산 중', paused: '일시정지', done: '완료', stopped: '정지됨', error: '오류' };

export function mountResults(V, opt) {
  const gPart = V.group('particles');
  const gSlice = V.group('slices');
  const gLines = V.group('reslines');
  const gIso = V.group('iso');
  const R = { run: null, frames: 0, cur: -1, fr: null, grid: null, Q: null, W: null, follow: true, playing: false, fixed: null, timer: null, loading: false };

  const view = () => ({
    field: $('field').value,
    range: $('rangeMode').value,
    part: $('showPart').checked,
    psize: +$('pscale').value || 1,
    slices: 'XYZ'.split('').filter((P) => $(`sl${P}`).checked).map((P) => [P, +$(`sl${P}v`).value / 1000]),
    vec: $('showVec').checked,
    stride: Math.max(1, +$('vecStride').value || 2),
    stream: $('showStream').checked,
    sPlane: $('streamPlane').value,
    sDens: +$('streamDens').value || 4,
    iso: $('showIso').checked,
    qf: (+$('qLevel').value || 8) / 100,
  });

  function gridOf(fr) {
    const h = fr.head.grid;
    return { n: h.n, origin: h.origin, h: h.h, u: fr.gu, p: fr.gp, occ: fr.gocc };
  }

  function vortMag(G) {
    const [nx, ny, nz] = G.n;
    const nxy = nx * ny;
    const out = new Float32Array(nx * ny * nz);
    const u = G.u;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const k = x + nx * y + nxy * z;
      if (!G.occ[k]) continue;
      const d = (off, i, n, c) => {
        const kp = i < n - 1 && G.occ[k + off] ? k + off : k;
        const km = i > 0 && G.occ[k - off] ? k - off : k;
        const den = (kp !== k) + (km !== k);
        return den ? (u[3 * kp + c] - u[3 * km + c]) / den / G.h : 0;
      };
      const wx = d(nx, y, ny, 2) - d(nxy, z, nz, 1);
      const wy = d(nxy, z, nz, 0) - d(1, x, nx, 2);
      const wz = d(1, x, nx, 1) - d(nx, y, ny, 0);
      out[k] = Math.hypot(wx, wy, wz);
    }
    return out;
  }

  function nearest(G, A, pos, n) {
    const out = new Float32Array(n);
    const [nx, ny, nz] = G.n;
    for (let i = 0; i < n; i++) {
      const a = Math.round((pos[3 * i] - G.origin[0]) / G.h);
      const b = Math.round((pos[3 * i + 1] - G.origin[1]) / G.h);
      const c = Math.round((pos[3 * i + 2] - G.origin[2]) / G.h);
      if (a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz) continue;
      out[i] = A[a + nx * (b + ny * c)];
    }
    return out;
  }

  function particleValues(fr, field) {
    const n = fr.head.n;
    const v = new Float32Array(n);
    if (field === 'p') return fr.p;
    if (field === 'vort') return nearest(R.grid, R.W || (R.W = vortMag(R.grid)), fr.pos, n);
    if (field === 'q') return nearest(R.grid, R.Q || (R.Q = F.qField(R.grid)), fr.pos, n);
    const c = { ux: 0, uy: 1, uz: 2 }[field];
    for (let i = 0; i < n; i++) v[i] = c === undefined ? Math.hypot(fr.vel[3 * i], fr.vel[3 * i + 1], fr.vel[3 * i + 2]) : fr.vel[3 * i + c];
    return v;
  }

  function lines(group, flat, color, opacity = 1) {
    if (!flat.length) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(flat), 3));
    const ls = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity, depthTest: false }));
    ls.renderOrder = 5;
    group.add(ls);
  }

  function slicePlane(G, P, s, field, lo, hi) {
    const pf = F.planeField(G, P, s, field, R.Q);
    const tex = new THREE.DataTexture(F.planeRGBA(pf, field, lo, hi), pf.w, pf.h, THREE.RGBAFormat);
    tex.magFilter = THREE.NearestFilter;
    tex.needsUpdate = true;
    const c = F.planeCorners(G, P, s);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([...c[0], ...c[1], ...c[2], ...c[3]]), 3));
    g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), 2));
    g.setIndex([0, 1, 2, 0, 2, 3]);
    return new THREE.Mesh(g, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, transparent: true, depthWrite: false }));
  }

  function draw() {
    const fr = R.fr;
    if (!fr) return;
    const v = view();
    const meta = F.FIELDS[v.field];
    [gPart, gSlice, gLines, gIso].forEach((g) => V.clear(g));
    V.group('preview').visible = false;
    document.body.classList.add('has-result');
    if ((v.field === 'q' || v.iso) && !R.Q) R.Q = F.qField(R.grid);
    const pv = particleValues(fr, v.field);
    let lo;
    let hi;
    if (v.range === 'fixed' && R.fixed && R.fixed.field === v.field) [lo, hi] = R.fixed.r;
    else {
      [lo, hi] = F.autoRange(v.field, pv);
      R.fixed = { field: v.field, r: [lo, hi] };
    }
    if (v.part) {
      const dp = (R.run && R.run.params && R.run.params.dp) || 10;
      const pts = V.points(fr.pos, F.colorize(pv, v.field, lo, hi), dp * v.psize, { world: true });
      if (v.slices.length) {
        pts.material.transparent = true;
        pts.material.opacity = 0.25;
      }
      gPart.add(pts);
    }
    const G = R.grid;
    for (const [P, f] of v.slices) gSlice.add(slicePlane(G, P, F.sliceIndex(G, P, f), v.field, lo, hi));
    if (v.vec) for (const [P, f] of v.slices) lines(gLines, F.planeVectors(G, P, F.sliceIndex(G, P, f), v.stride), 0xffffff, 0.9);
    if (v.stream) {
      const sl = v.slices.find(([P]) => P === v.sPlane);
      const f = sl ? sl[1] : 0.5;
      lines(gLines, F.planeStreams(G, v.sPlane, F.sliceIndex(G, v.sPlane, f), v.sDens), 0xe8eef6, 0.85);
    }
    if (v.iso) {
      const lev = F.qLevel(R.Q, v.qf);
      const m = F.isosurface(G, R.Q, lev);
      if (m.pos.length) {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(m.pos, 3));
        let smax = 1e-9;
        for (const s of m.spd) smax = Math.max(smax, s);
        g.setAttribute('color', new THREE.BufferAttribute(F.colorize(m.spd, 'umag', 0, smax), 3));
        g.computeVertexNormals();
        gIso.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 0.6 })));
      }
      $('isoInfo').textContent = Number.isFinite(lev) ? `Q = ${fmt(lev)} 1/s² · 삼각형 ${(m.pos.length / 9).toLocaleString()}개 (색 = 속도)` : 'Q > 0 인 곳이 없습니다';
    } else $('isoInfo').textContent = '';
    setLegend($('legend'), `${meta.label} [${meta.unit}]`, lo, hi, fmt);
    $('legendBar').style.background = F.legendGradient(v.field, lo, hi);
    $('frameLabel').textContent = `${R.cur + 1}/${R.frames} · t = ${fr.head.t.toFixed(4)} s · 입자 ${fr.head.active.toLocaleString()}`;
    opt.onDraw && opt.onDraw();
  }

  async function load(k) {
    if (R.loading || k < 0 || k >= R.frames) return;
    R.loading = true;
    try {
      const buf = await (await fetch(`/api/run/frame/${k}`)).arrayBuffer();
      R.fr = F.decodeFrame(buf);
      R.grid = gridOf(R.fr);
      R.Q = null;
      R.W = null;
      R.cur = k;
      $('frame').value = k;
      $('resultbar').hidden = false;
      draw();
    } catch (e) {
      opt.note(e.message, true);
    }
    R.loading = false;
  }

  function series(name, label, keys, names, scale = 1) {
    const s = R.run.series;
    const t = s.time || [];
    return { title: label, xlabel: 't [s]', series: keys.filter((k) => s[k] && s[k].length).map((k, i) => ({ name: names[i], x: t.slice(0, s[k].length), y: s[k].map((v) => v * scale) })) };
  }

  function plots() {
    const box = $('plots');
    box.innerHTML = '';
    if (!R.run || !(R.run.series.time || []).length || R.run.series.time.length < 2) return;
    const cards = [series('rho', '밀도 오차 |ρ/ρ₀ − 1| [%]', ['density.error.mean', 'density.error.max'], ['평균', '최대'], 100), series('e', '에너지 [mJ]', ['energy.kinetic', 'energy.potential'], ['운동', '위치'])];
    for (const g of R.run.groups || []) cards.push(series('f', `벽 하중 · ${g} [N]`, ['x', 'y', 'z'].map((c) => `force.${g}.${c}`), ['Fx', 'Fy', 'Fz']));
    for (const m of R.run.probes || []) {
      cards.push(series('ps', `${m.id} ${m.name} 속도 [mm/s]`, [`probe.${m.id}.speed`], ['|u|']));
      cards.push(series('pp', `${m.id} ${m.name} 압력 [MPa]`, [`probe.${m.id}.pressure`], ['p']));
    }
    for (const c of cards) if (c.series.length) box.append(plotCard(c));
  }

  function summary() {
    const r = R.run;
    const pct = r.t_end ? Math.min(100, (100 * r.t) / r.t_end) : 0;
    $('runBar').style.width = `${pct.toFixed(1)}%`;
    $('runState').textContent = `${STATE_TEXT[r.state] || r.state} · t = ${r.t.toFixed(4)} / ${r.t_end} s · ${r.step.toLocaleString()} 스텝 · dt ${r.dt ? r.dt.toExponential(2) : '-'} s · 경과 ${r.wall.toFixed(1)} s`;
    const busy = ['building', 'running', 'paused'].includes(r.state);
    $('solve').disabled = busy;
    $('pause').disabled = !busy || r.state === 'building';
    $('pause').textContent = r.state === 'paused' ? '재개' : '일시정지';
    $('stepOne').disabled = r.state !== 'paused';
    $('stopRun').disabled = !busy;
    $('pack').hidden = !r.frames || busy;
    const p = r.params || {};
    const s = r.series || {};
    const last = (k) => (s[k] && s[k].length ? s[k][s[k].length - 1] : null);
    const lines_ = [r.message];
    if (r.n) lines_.push(`${p.dim === 2 ? '2D' : '3D'} · 유체 ${r.n.toLocaleString()} · 경계 ${r.nbound.toLocaleString()} 입자 · ${p.boundary || ''} · ${p.scheme || ''} · 음속 계수 ${p.coefsound || ''} · ${p.viscosity || ''} · 장치 ${r.device}`);
    if (p.h) lines_.push(`입자 간격 ${fmt(p.dp)} mm · 상호작용 반경 2h ${fmt(2 * p.h)} mm (이웃 약 ${p.neighbors}개) · 음속 ${fmt(p.cs0 / 1000)} m/s · 출력 ${r.frames} 프레임`);
    if (last('density.error.max') !== null) lines_.push(`밀도 오차 평균 ${(100 * last('density.error.mean')).toFixed(2)}% · 최대 ${(100 * last('density.error.max')).toFixed(2)}%`);
    if (r.out) lines_.push(`제외된 입자 ${r.out.toLocaleString()}개 (벽을 뚫고 들어감 ${(r.out_wall || 0).toLocaleString()} · 해석 영역 밖·밀도 범위 이탈 ${(r.out - (r.out_wall || 0)).toLocaleString()})`);
    for (const w of r.warnings || []) lines_.push(`⚠ ${w}`);
    $('summary').textContent = lines_.filter(Boolean).join('\n');
  }

  async function poll() {
    try {
      const r = await api('/api/run');
      R.run = r;
      summary();
      const grew = r.frames !== R.frames;
      R.frames = r.frames;
      $('frame').max = Math.max(0, r.frames - 1);
      if (grew) plots();
      if (grew && R.follow && r.frames) await load(r.frames - 1);
      const busy = ['building', 'running', 'paused'].includes(r.state);
      clearTimeout(R.timer);
      R.timer = setTimeout(poll, busy ? 600 : 3000);
    } catch (e) {
      R.timer = setTimeout(poll, 3000);
    }
  }

  async function play() {
    if (!R.playing) return;
    const k = R.cur + 1 >= R.frames ? 0 : R.cur + 1;
    await load(k);
    setTimeout(play, 60);
  }

  async function ctl(what) {
    try {
      R.run = await api(`/api/run/${what}`, { method: 'POST' });
      summary();
    } catch (e) {
      opt.note(e.message, true);
    }
    poll();
  }

  $('solve').onclick = async () => {
    try {
      R.follow = true;
      R.frames = 0;
      R.fr = null;
      document.body.classList.remove('has-result');
      [gPart, gSlice, gLines, gIso].forEach((g) => V.clear(g));
      R.run = await api('/api/solve', { method: 'POST' });
      opt.note('해석을 시작했습니다');
      summary();
      poll();
    } catch (e) {
      opt.note(e.message, true);
    }
  };
  $('pause').onclick = () => ctl(R.run && R.run.state === 'paused' ? 'resume' : 'pause');
  $('stepOne').onclick = () => ctl('step');
  $('stopRun').onclick = () => ctl('stop');
  $('pack').onclick = () => (window.location.href = '/api/run/pack');
  $('frame').oninput = () => {
    R.follow = +$('frame').value >= R.frames - 1;
    load(+$('frame').value);
  };
  $('play').onclick = () => {
    R.playing = !R.playing;
    $('play').textContent = R.playing ? '❚❚' : '▶';
    if (R.playing) {
      R.follow = false;
      play();
    }
  };
  const redraw = () => draw();
  for (const id of ['field', 'rangeMode', 'showPart', 'showVec', 'vecStride', 'showStream', 'streamPlane', 'streamDens', 'showIso', 'qLevel', 'slX', 'slY', 'slZ']) $(id).onchange = redraw;
  for (const P of 'XYZ') $(`sl${P}v`).oninput = redraw;
  window.addEventListener('sph-pscale', redraw);
  $('field').innerHTML = Object.entries(F.FIELDS).map(([k, m]) => `<option value="${k}">${m.label}</option>`).join('');
  poll();
  return { R, draw, poll };
}
