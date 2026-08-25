/* ==========================================================================
   The whole-network view: the same knockdown, on all 2,239 genes and 8,543
   edges of the published K562 GRN.

   The detailed slide walks 34 genes so the mechanism is readable. This one
   exists to show the scale the model actually runs at, and to make the "median
   13 hops" claim something the audience watches rather than reads.

   Canvas, not SVG: 8,543 edges as DOM nodes would make stepping the animation
   visibly stutter on a conference laptop.

   The dynamics are the same Hill fixed-point iteration as the detailed slide --
   the constants are imported from IPerturbGRN.M rather than restated, so the two
   pictures cannot drift apart.
   ========================================================================== */

(function () {
  'use strict';

  /**
   * This view answers one question only: how far has the perturbation reached?
   *
   * It deliberately does not colour by direction of change. At 2,239 nodes a
   * red/blue diverging scale is a wash of pastel that says nothing legible from
   * the back of a hall, and the up/down story is already told properly on the
   * 34-gene slide. Here: untouched, reached, and reaching-now. Three states.
   */
  var PAL = {
    rest:    'rgba(10,31,92,0.035)',   // the whole network, at rest
    spent:   'rgba(169,192,220,0.55)', // edges that carried an earlier hop
    carry:   'rgba(155,58,58,0.80)',   // edge carrying the current hop
    node:    '#E8ECF4',                // not reached yet
    reached: '#A9C0DC',                // reached earlier: pale, not emphasised
    now:     '#9B3A3A',                // reached on this hop
    seed:    '#0A1F5C',
    halo:    '#FCFCFE'
  };

  /* ---------------------------------------------------------------- solver */

  /** Hill fixed point over typed arrays. Same equations, same constants. */
  function solve(D, clampIdx, clampVal, init) {
    var M = window.IPerturbGRN.M,
        N = D.ids.length, E = D.es.length,
        WM = { 1: 0.85, 2: 0.65, 3: 0.45, 4: 0.25, 5: 0.50 },
        WMr = { 1: 0.90, 2: 0.70, 3: 0.50, 4: 0.30, 5: 0.50 };

    var w = new Float64Array(E), i;
    for (i = 0; i < E; i++) {
      var s = D.sign[i];
      w[i] = s * ((s > 0 ? WM : WMr)[D.level[i]] || 0.5);
    }

    var x = init ? Float64Array.from(init) : new Float64Array(N).fill(M.x0);
    if (clampIdx != null) x[clampIdx] = clampVal;

    var traj = [Float64Array.from(x)], hops = 0, converged = false;
    var K = Math.pow(M.x0, M.n);

    for (var t = 0; t < M.maxIter; t++) {
      var h = new Float64Array(N), r = new Float64Array(N);
      for (i = 0; i < E; i++) {
        var xs = x[D.es[i]];
        if (xs < 0) xs = 0;
        var xn = Math.pow(xs, M.n),
            drive = w[i] * (xn / (K + xn + 1e-12));
        if (drive > 0) h[D.et[i]] += drive; else r[D.et[i]] += -drive;
      }
      var next = new Float64Array(N), d2 = 0;
      for (i = 0; i < N; i++) next[i] = (M.x0 * (1 + h[i]) + M.b) / (1 + M.alpha + r[i]);
      if (clampIdx != null) next[clampIdx] = clampVal;
      for (i = 0; i < N; i++) { var d = next[i] - x[i]; d2 += d * d; }
      x = next;
      traj.push(Float64Array.from(x));
      hops = t + 1;
      if (Math.sqrt(d2) < M.eps) { converged = true; break; }
    }
    return { traj: traj, hops: hops, converged: converged };
  }

  /* ---------------------------------------------------------------- render */

  function FullNet(host, D) {
    this.D = D;
    this.host = host;

    var c = document.createElement('canvas');
    c.className = 'fullnet-canvas';
    host.appendChild(c);
    this.canvas = c;
    this.ctx = c.getContext('2d');

    this.ctrl = solve(D, null, null, null).traj.slice(-1)[0];
    this.kd = this.ctrl[D.seedIdx] * window.IPerturbGRN.M.knockdown;
    this.res = solve(D, D.seedIdx, this.kd, this.ctrl);

    // the iteration at which each gene first departs from control
    var N = D.ids.length;
    this.firstMove = new Int16Array(N).fill(-1);
    for (var t = 0; t < this.res.traj.length; t++) {
      var x = this.res.traj[t];
      for (var i = 0; i < N; i++) {
        if (this.firstMove[i] < 0 && Math.abs(x[i] - this.ctrl[i]) > 1e-6) {
          this.firstMove[i] = t;
        }
      }
    }
    this.lastWave = 0;
    for (i = 0; i < N; i++) if (this.firstMove[i] > this.lastWave) this.lastWave = this.firstMove[i];

    this.cap = 0.45;
    this.hop = 0;
    this._resize();
    window.addEventListener('resize', this._resize.bind(this));
  }

  FullNet.prototype._resize = function () {
    var r = this.host.getBoundingClientRect(),
        dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (r.width < 10 || r.height < 10) return;
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.canvas.style.width = r.width + 'px';
    this.canvas.style.height = r.height + 'px';

    // fit the precomputed layout box into the canvas, preserving aspect
    var s = Math.min(r.width / this.D.canvas.w, r.height / this.D.canvas.h);
    this.scale = s * dpr;
    this.ox = (this.canvas.width - this.D.canvas.w * this.scale) / 2;
    this.oy = (this.canvas.height - this.D.canvas.h * this.scale) / 2;
    this.draw();
  };

  FullNet.prototype.X = function (i) { return this.D.x[i] * this.scale + this.ox; };
  FullNet.prototype.Y = function (i) { return this.D.y[i] * this.scale + this.oy; };

  /**
   * Painted back to front, so whatever carries the current hop ends up on top of
   * the 8,543-edge background rather than buried in it. Order matters more than
   * colour at this density: a bright edge drawn underneath a thousand grey ones
   * still reads as grey.
   */
  FullNet.prototype.draw = function () {
    var D = this.D, ctx = this.ctx, i;
    if (!this.scale) return;
    var hop = this.hop,
        u = this.scale / 0.5,
        rBase = Math.max(0.9, 1.3 * u),
        rMoved = Math.max(1.1, 1.7 * u),
        rHot = Math.max(2.6, 4.2 * u),
        live = hop >= 1 && hop <= this.lastWave;

    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.lineCap = 'round';

    // 1 — the whole network, one faint wash. Every edge is still drawn: the
    // caption claims all 8,543, so thin the ink, never the edge list.
    ctx.strokeStyle = PAL.rest;
    ctx.lineWidth = Math.max(0.4, 0.5 * u);
    ctx.beginPath();
    for (i = 0; i < D.es.length; i++) {
      ctx.moveTo(this.X(D.es[i]), this.Y(D.es[i]));
      ctx.lineTo(this.X(D.et[i]), this.Y(D.et[i]));
    }
    ctx.stroke();

    // 2 — edges that carried earlier hops, so the path travelled stays legible
    if (hop >= 2) {
      ctx.strokeStyle = PAL.spent;
      ctx.lineWidth = Math.max(0.6, 0.85 * u);
      ctx.beginPath();
      for (i = 0; i < D.es.length; i++) {
        var ps = this.firstMove[D.es[i]], pt = this.firstMove[D.et[i]];
        if (pt >= 1 && pt < hop && ps >= 0 && ps < pt) {
          ctx.moveTo(this.X(D.es[i]), this.Y(D.es[i]));
          ctx.lineTo(this.X(D.et[i]), this.Y(D.et[i]));
        }
      }
      ctx.stroke();
    }

    // 3 — genes not yet reached
    ctx.fillStyle = PAL.node;
    ctx.beginPath();
    for (i = 0; i < D.ids.length; i++) {
      var f0 = this.firstMove[i];
      if (f0 < 0 || f0 > hop) {
        ctx.moveTo(this.X(i) + rBase, this.Y(i));
        ctx.arc(this.X(i), this.Y(i), rBase, 0, 6.2832);
      }
    }
    ctx.fill();

    // 4 — genes reached on earlier hops
    ctx.fillStyle = PAL.reached;
    ctx.beginPath();
    for (i = 0; i < D.ids.length; i++) {
      var f1 = this.firstMove[i];
      if (f1 < 0 || f1 >= hop) continue;
      ctx.moveTo(this.X(i) + rMoved, this.Y(i));
      ctx.arc(this.X(i), this.Y(i), rMoved, 0, 6.2832);
    }
    ctx.fill();

    // 5 — this hop's edges, above the whole background
    if (live) {
      ctx.strokeStyle = PAL.carry;
      ctx.lineWidth = Math.max(1.1, 1.9 * u);
      ctx.beginPath();
      for (i = 0; i < D.es.length; i++) {
        var fs = this.firstMove[D.es[i]], ft = this.firstMove[D.et[i]];
        if (ft === hop && fs >= 0 && fs < hop) {
          ctx.moveTo(this.X(D.es[i]), this.Y(D.es[i]));
          ctx.lineTo(this.X(D.et[i]), this.Y(D.et[i]));
        }
      }
      ctx.stroke();
    }

    // 6 — this hop's genes, on top of everything, haloed against the mesh
    if (live) {
      for (i = 0; i < D.ids.length; i++) {
        if (this.firstMove[i] !== hop) continue;
        var cx = this.X(i), cy = this.Y(i);
        ctx.beginPath();
        ctx.arc(cx, cy, rHot, 0, 6.2832);
        ctx.fillStyle = PAL.now;
        ctx.fill();
        ctx.lineWidth = Math.max(1.4, 2.4 * u);
        ctx.strokeStyle = PAL.halo;
        ctx.stroke();
      }
    }

    // 7 — the knocked-down gene, always visible
    var si = D.seedIdx;
    ctx.beginPath();
    ctx.arc(this.X(si), this.Y(si), rHot * 0.72, 0, 6.2832);
    ctx.fillStyle = PAL.seed;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(this.X(si), this.Y(si), rHot * 1.5, 0, 6.2832);
    ctx.strokeStyle = PAL.seed;
    ctx.lineWidth = Math.max(1.2, 1.8 * u);
    ctx.setLineDash([Math.max(2, 3 * u), Math.max(2, 3 * u)]);
    ctx.stroke();
    ctx.setLineDash([]);
  };

  FullNet.prototype.reached = function () {
    var n = 0;
    for (var i = 0; i < this.firstMove.length; i++) {
      if (this.firstMove[i] >= 0 && this.firstMove[i] <= this.hop) n++;
    }
    return n;
  };

  FullNet.prototype.go = function (hop) {
    this.hop = Math.max(0, Math.min(hop, this.lastWave + 1));
    this.draw();
    if (this.onHop) this.onHop(this.hop, this);
  };
  FullNet.prototype.next = function () { this.go(this.hop + 1); };
  FullNet.prototype.prev = function () { this.go(this.hop - 1); };
  FullNet.prototype.play = function () {
    var self = this;
    this.stop();
    if (this.hop >= this.lastWave + 1) this.go(0);
    this._timer = setInterval(function () {
      if (self.hop >= self.lastWave + 1) { self.stop(); return; }
      self.next();
    }, 500);
  };
  FullNet.prototype.stop = function () {
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
  };
  FullNet.prototype.playing = function () { return !!this._timer; };

  window.IPerturbFullNet = { View: FullNet, solve: solve };
})();
