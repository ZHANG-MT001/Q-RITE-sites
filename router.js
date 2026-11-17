/* Offline alignment search: a port of plan_route() from qanat/live.py.
 *
 * The expensive half of that function - the row-distance field and the two
 * hard masks - is precomputed by qanat/static_build.py with the same scipy
 * calls, and shipped as packed bitsets. What runs here is only the part that
 * depends on where the user clicks: crop a window, A* it, smooth it, score it.
 */
(function (global) {
  'use strict';

  const PAD = 320.0;            // halo around the clicked endpoints, metres
  const LAMBDAS = [0.5, 1.5, 4, 10, 22];
  const WIDTH_M = 26.0;         // formation width of the planned road
  const RADII = [15.0, 8.0];    // buffers tried for the "avoid every row" line
  const MAXL = 400.0;

  /* ---------- bytes ---------- */
  function b64(s) {
    const bin = atob(s), n = bin.length, out = new Uint8Array(n);
    for (let i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function unpack(s, n) {
    const bytes = b64(s), out = new Uint8Array(n);
    for (let i = 0; i < n; i++) out[i] = (bytes[i >> 3] >> (7 - (i & 7))) & 1;
    return out;
  }

  /* ---------- projection (WGS84 -> UTM north) ---------- */
  function utmForward(lon, lat, zone) {
    const a = 6378137.0, f = 1 / 298.257223563, k0 = 0.9996;
    const e2 = f * (2 - f), ep2 = e2 / (1 - e2), rad = Math.PI / 180;
    const lon0 = ((zone - 1) * 6 - 180 + 3) * rad;
    const phi = lat * rad, lam = lon * rad;
    const sp = Math.sin(phi), cp = Math.cos(phi), tp = Math.tan(phi);
    const N = a / Math.sqrt(1 - e2 * sp * sp);
    const T = tp * tp, C = ep2 * cp * cp, A = cp * (lam - lon0);
    const M = a * ((1 - e2 / 4 - 3 * e2 * e2 / 64 - 5 * e2 * e2 * e2 / 256) * phi
      - (3 * e2 / 8 + 3 * e2 * e2 / 32 + 45 * e2 * e2 * e2 / 1024) * Math.sin(2 * phi)
      + (15 * e2 * e2 / 256 + 45 * e2 * e2 * e2 / 1024) * Math.sin(4 * phi)
      - (35 * e2 * e2 * e2 / 3072) * Math.sin(6 * phi));
    const x = k0 * N * (A + (1 - T + C) * A * A * A / 6
      + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A * A * A * A * A / 120) + 500000;
    const y = k0 * (M + N * tp * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * Math.pow(A, 4) / 24
      + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * Math.pow(A, 6) / 720));
    return [x, y];
  }

  function utmInverse(x, y, zone) {
    const a = 6378137.0, f = 1 / 298.257223563, k0 = 0.9996;
    const e2 = f * (2 - f), e1 = (1 - Math.sqrt(1 - e2)) / (1 + Math.sqrt(1 - e2));
    const rad = Math.PI / 180;
    const xp = x - 500000, yp = y;
    const M = yp / k0;
    const mu = M / (a * (1 - e2 / 4 - 3 * e2 * e2 / 64 - 5 * e2 * e2 * e2 / 256));
    const phi1 = mu + (3 * e1 / 2 - 27 * Math.pow(e1, 3) / 32) * Math.sin(2 * mu)
      + (21 * e1 * e1 / 16 - 55 * Math.pow(e1, 4) / 32) * Math.sin(4 * mu)
      + (151 * Math.pow(e1, 3) / 96) * Math.sin(6 * mu)
      + (1097 * Math.pow(e1, 4) / 512) * Math.sin(8 * mu);
    const sp = Math.sin(phi1), cp = Math.cos(phi1), tp = Math.tan(phi1);
    const N1 = a / Math.sqrt(1 - e2 * sp * sp);
    const T1 = tp * tp, C1 = e2 / (1 - e2) * cp * cp;
    const R1 = a * (1 - e2) / Math.pow(1 - e2 * sp * sp, 1.5);
    const D = xp / (N1 * k0);
    const lat = phi1 - (N1 * tp / R1) * (D * D / 2
      - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * e2 / (1 - e2)) * Math.pow(D, 4) / 24
      + (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * e2 / (1 - e2) - 3 * C1 * C1)
      * Math.pow(D, 6) / 720);
    const lon = (D - (1 + 2 * T1 + C1) * Math.pow(D, 3) / 6
      + (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * e2 / (1 - e2) + 24 * T1 * T1)
      * Math.pow(D, 5) / 120) / cp;
    const lon0 = ((zone - 1) * 6 - 180 + 3);
    return [lon0 + lon / rad, lat / rad];
  }

  /* ---------- geometry ---------- */
  function segInt(a, b, c, d) {
    const d1x = b[0] - a[0], d1y = b[1] - a[1];
    const d2x = d[0] - c[0], d2y = d[1] - c[1];
    const den = d1x * d2y - d1y * d2x;
    if (Math.abs(den) < 1e-12) return null;
    const ex = c[0] - a[0], ey = c[1] - a[1];
    const t = (ex * d2y - ey * d2x) / den;
    const u = (ex * d1y - ey * d1x) / den;
    if (t < 0 || t > 1 || u < 0 || u > 1) return null;
    return { t: t, u: u, x: a[0] + t * d1x, y: a[1] + t * d1y };
  }

  function pathLen(p) {
    let L = 0;
    for (let i = 1; i < p.length; i++) L += Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]);
    return L;
  }

  function resample(p, step) {
    if (p.length < 2) return p.slice();
    const cum = [0];
    for (let i = 1; i < p.length; i++)
      cum.push(cum[i - 1] + Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]));
    const tot = cum[cum.length - 1];
    if (tot <= 0) return p.slice();
    const n = Math.max(2, Math.floor(tot / step) + 1);
    const out = [];
    for (let k = 0; k < n; k++) {
      const s = tot * k / (n - 1);
      let i = 0;
      while (i < p.length - 2 && cum[i + 1] < s) i++;
      const f = (cum[i + 1] > cum[i]) ? (s - cum[i]) / (cum[i + 1] - cum[i]) : 0;
      out.push([p[i][0] + (p[i + 1][0] - p[i][0]) * f,
                p[i][1] + (p[i + 1][1] - p[i][1]) * f]);
    }
    return out;
  }

  function chaikin(p, k) {
    let q = p;
    for (let it = 0; it < k; it++) {
      if (q.length < 3) break;
      const out = [q[0]];
      for (let i = 0; i < q.length - 1; i++) {
        out.push([0.75 * q[i][0] + 0.25 * q[i + 1][0], 0.75 * q[i][1] + 0.25 * q[i + 1][1]]);
        out.push([0.25 * q[i][0] + 0.75 * q[i + 1][0], 0.25 * q[i][1] + 0.75 * q[i + 1][1]]);
      }
      out.push(q[q.length - 1]);
      q = out;
    }
    return q;
  }

  /* ---------- scoring, identical to evaluate() in live.py ---------- */
  function evaluate(path, rows) {
    const total = pathLen(path);
    let cav = 0, mouths = 0, cross = 0, amin = null;
    for (let r = 0; r < rows.length; r++) {
      const A = rows[r].p0, B = rows[r].p1;
      const rx = B[0] - A[0], ry = B[1] - A[1], rn = Math.hypot(rx, ry) || 1;
      const spacing = rows[r].spacing || 1;
      for (let i = 0; i < path.length - 1; i++) {
        const hit = segInt(path[i], path[i + 1], A, B);
        if (!hit) continue;
        /* local tangent of the alignment, over a ~100 m window */
        const i0 = Math.max(0, i - 5), i1 = Math.min(path.length - 1, i + 5);
        let px = path[i1][0] - path[i0][0], py = path[i1][1] - path[i0][1];
        const pn = Math.hypot(px, py) || 1;
        let cosA = Math.abs((px * rx + py * ry) / (pn * rn));
        cosA = Math.min(1, Math.max(0, cosA));
        let alpha = Math.acos(cosA) * 180 / Math.PI;
        if (alpha < 5) alpha = 5;
        const L = Math.min(WIDTH_M / Math.sin(alpha * Math.PI / 180), MAXL);
        cav += L; mouths += L / spacing; cross++;
        amin = (amin === null) ? alpha : Math.min(amin, alpha);
      }
    }
    return {
      /* three decimals, like round(x, 3) in the server */
      length_km: Math.round(total) / 1000,
      cavity_km: Math.round(cav) / 1000,
      mouths: Math.round(mouths * 100) / 100,
      crossings: cross,
      alpha_min: amin === null ? null : Math.round(amin * 10) / 10
    };
  }

  function clearance(path, rows) {
    let best = Infinity;
    for (let r = 0; r < rows.length; r++) {
      const A = rows[r].p0, B = rows[r].p1;
      const abx = B[0] - A[0], aby = B[1] - A[1];
      const L2 = abx * abx + aby * aby || 1;
      for (let i = 0; i < path.length; i++) {
        let t = ((path[i][0] - A[0]) * abx + (path[i][1] - A[1]) * aby) / L2;
        t = Math.max(0, Math.min(1, t));
        const dx = path[i][0] - (A[0] + t * abx), dy = path[i][1] - (A[1] + t * aby);
        const d = Math.hypot(dx, dy);
        if (d < best) best = d;
      }
    }
    return best;
  }

  /* ---------- search ---------- */
  function astar(W, H, at, start, goal, lam, blocked) {
    const N = W * H;
    const dist = new Float32Array(N).fill(Infinity);
    const prev = new Int32Array(N).fill(-1);
    const seen = new Uint8Array(N);
    const sIdx = start[0] * W + start[1], gIdx = goal[0] * W + goal[1];
    const hf = (i) => Math.hypot((i / W | 0) - goal[0], (i % W) - goal[1]) * 12;

    const hf2 = [], hv = [];
    const push = (f, v) => {
      hf2.push(f); hv.push(v);
      let i = hf2.length - 1;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (hf2[p] <= hf2[i]) break;
        [hf2[p], hf2[i]] = [hf2[i], hf2[p]];
        [hv[p], hv[i]] = [hv[i], hv[p]];
        i = p;
      }
    };
    const pop = () => {
      const top = hv[0];
      const lf = hf2.pop(), lv = hv.pop();
      if (hf2.length) {
        hf2[0] = lf; hv[0] = lv;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1;
          let m = i;
          if (l < hf2.length && hf2[l] < hf2[m]) m = l;
          if (r < hf2.length && hf2[r] < hf2[m]) m = r;
          if (m === i) break;
          [hf2[m], hf2[i]] = [hf2[i], hf2[m]];
          [hv[m], hv[i]] = [hv[i], hv[m]];
          i = m;
        }
      }
      return top;
    };

    dist[sIdx] = 0;
    push(hf(sIdx), sIdx);
    const NB = [[-1, -1, 1.4142], [-1, 0, 1], [-1, 1, 1.4142], [0, -1, 1],
                [0, 1, 1], [1, -1, 1.4142], [1, 0, 1], [1, 1, 1.4142]];
    while (hf2.length) {
      const idx = pop();
      if (seen[idx]) continue;
      seen[idx] = 1;
      if (idx === gIdx) break;
      const y = idx / W | 0, x = idx % W;
      for (let k = 0; k < 8; k++) {
        const yy = y + NB[k][0], xx = x + NB[k][1];
        if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
        const ni = yy * W + xx;
        if (seen[ni]) continue;
        if (blocked && blocked[ni]) continue;
        if (blocked && NB[k][0] && NB[k][1]
            && (blocked[(y + NB[k][0]) * W + x] || blocked[y * W + (x + NB[k][1])])) continue;
        const ng = dist[idx] + NB[k][2] * 12 * 0.5 * (1 + lam * at(idx) + 1 + lam * at(ni));
        if (ng < dist[ni]) {
          dist[ni] = ng; prev[ni] = idx;
          push(ng + hf(ni), ni);
        }
      }
    }
    if (!isFinite(dist[gIdx])) return null;
    const path = [];
    let cur = gIdx;
    while (cur !== -1) { path.push([cur / W | 0, cur % W]); cur = prev[cur]; }
    return path.reverse();
  }

  /* ---------- one site ---------- */
  function planRouteLocal(waypoints, site) {
    const g = site.grid, cell = g.cell, zone = site.zone || 45;
    const field = b64(site.field);
    const nx = g.nx, ny = g.ny;

    const wp = waypoints.map(p => utmForward(p[0], p[1], zone));
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (const p of wp) {
      minx = Math.min(minx, p[0]); maxx = Math.max(maxx, p[0]);
      miny = Math.min(miny, p[1]); maxy = Math.max(maxy, p[1]);
    }
    const cx0 = Math.max(0, Math.floor((minx - PAD - g.x0) / cell));
    const cx1 = Math.min(nx - 1, Math.ceil((maxx + PAD - g.x0) / cell));
    const cy0 = Math.max(0, Math.floor((miny - PAD - g.y0) / cell));
    const cy1 = Math.min(ny - 1, Math.ceil((maxy + PAD - g.y0) / cell));
    const W = cx1 - cx0 + 1, H = cy1 - cy0 + 1;
    if (W < 3 || H < 3) return [];

    /* a view of the shipped field, cropped to this corridor */
    const at = (i) => {
      const y = i / W | 0, x = i % W;
      return field[((cy0 + y) * nx) + (cx0 + x)] / 255;
    };
    const cellOf = (p) => [
      Math.max(0, Math.min(H - 1, Math.round((p[1] - g.y0) / cell) - cy0)),
      Math.max(0, Math.min(W - 1, Math.round((p[0] - g.x0) / cell) - cx0))
    ];
    const ptsOf = (cells) => cells.map(c =>
      [g.x0 + (cx0 + c[1] + 0.5) * cell, g.y0 + (cy0 + c[0] + 0.5) * cell]);

    const rows = site.rows;
    const out = [];
    const straight = resample(wp, 20);
    out.push(Object.assign({ kind: 'straight', lam: null, xy: straight },
      evaluate(straight, rows)));

    const smooth = (path, keepOut) => {
      let best = resample(path, 10), bestEv = evaluate(best, rows);
      const sm = resample(chaikin(path, 2), 10);
      const ev = evaluate(sm, rows);
      if (!keepOut || clearance(sm, rows) >= keepOut) {
        if (ev.cavity_km < bestEv.cavity_km ||
           (ev.cavity_km === bestEv.cavity_km && ev.length_km < bestEv.length_km)) {
          best = sm; bestEv = ev;
        }
      }
      return [best, bestEv];
    };

    /* hard avoidance, independent of lambda */
    if (rows.length) {
      const a = wp[0], b = wp[wp.length - 1];
      const [sa, sb] = [cellOf(a), cellOf(b)];
      for (let ri = 0; ri < RADII.length; ri++) {
        const bits = unpack(site.block[ri], nx * ny);
        const blocked = new Uint8Array(W * H);
        for (let y = 0; y < H; y++)
          for (let x = 0; x < W; x++)
            blocked[y * W + x] = bits[(cy0 + y) * nx + (cx0 + x)];
        blocked[sa[0] * W + sa[1]] = 0;
        blocked[sb[0] * W + sb[1]] = 0;
        const cells = astar(W, H, at, sa, sb, 0, blocked);
        if (!cells) continue;
        const seg = ptsOf(cells);
        seg[0] = a; seg[seg.length - 1] = b;
        const [xy, ev] = smooth(seg, RADII[ri] * 0.7);
        if (ev.crossings === 0 || ri === RADII.length - 1) {
          out.push(Object.assign({ kind: 'avoid', lam: null, xy: xy, radius: RADII[ri] }, ev));
          break;
        }
      }
    }

    for (const lam of LAMBDAS) {
      /* one search per leg, so a three-point corridor is still honoured */
      const parts = [];
      let ok = true;
      for (let k = 0; k < wp.length - 1; k++) {
        const cells = astar(W, H, at, cellOf(wp[k]), cellOf(wp[k + 1]), lam, null);
        if (!cells) { ok = false; break; }
        const seg = ptsOf(cells);
        seg[0] = wp[k]; seg[seg.length - 1] = wp[k + 1];
        parts.push(k ? seg.slice(1) : seg);
      }
      if (!ok) continue;
      const joined = parts.reduce((acc, p) => acc.concat(p), []);
      const [xy, ev] = smooth(joined, 0);
      out.push(Object.assign({ kind: 'route', lam: lam, xy: xy }, ev));
    }

    /* back to lon/lat for drawing */
    for (const c of out) {
      c.lonlat = c.xy.map(p => utmInverse(p[0], p[1], zone));
      delete c.xy;
    }
    return out;
  }

  global.planRouteLocal = planRouteLocal;
})(window);
