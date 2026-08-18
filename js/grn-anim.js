/* ==========================================================================
   Animation 2 — a CRISPRi knockdown propagating on the real K562 GRN.

   The dynamics on screen are the paper's equations, evaluated in the browser:

       phi_ki = x_k^n / (K^n + x_k^n)
       h_i    = sum_k max(0,  w_ki * phi_ki)
       r_i    = sum_k max(0, -w_ki * phi_ki)
       x_i    = (V_i (1 + h_i) + b_i) / (1 + alpha_i + r_i)

   iterated with the perturbed gene re-clamped every hop until
   ||x(t+1) - x(t)||_2 < eps, exactly as GRNN.forward does.

   Scalar constants come from the model's initialisation (iperturb.py):
       V = x0,  alpha = exp(-2.3),  b = softplus(-4),  K = x0(source),  n = 2.5
   Edge magnitudes come from the same level-keyed table the code initialises
   w_raw from. This is not a fitted model, and the slide says so.

   Layout is one column per hop, so the wavefront reads left to right. Each step
   bolds exactly the edges and nodes that carry that hop and dims the rest.

   Colour is always the change against the model's OWN unperturbed fixed point,
   never against x0 — otherwise an unfitted model's drift would be mistaken for
   a perturbation effect.
   ========================================================================== */

(function () {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';

  /* ------------------------------------------------------- model constants */
  var M = {
    x0: 1.0,                       // normalised baseline expression
    alpha: Math.exp(-2.3),         // ~0.1003   (log_alpha init)
    b: Math.log1p(Math.exp(-4.0)), // ~0.0181   (softplus(b_raw) init)
    n: 2.5,                        // 1 + 3*sigmoid(0)
    eps: 1e-5,
    maxIter: 100,
    knockdown: 0.25,               // CRISPRi holds the target at ~25% of baseline
    rwrAlpha: 0.6
  };

  // |w| by (sign, level) — the w_raw initialisation table in iperturb.py.
  var W_MAG = {
    act: { 1: 0.85, 2: 0.65, 3: 0.45, 4: 0.25, 5: 0.50 },
    rep: { 1: 0.90, 2: 0.70, 3: 0.50, 4: 0.30, 5: 0.50 }
  };

  function edgeWeight(e) {
    var table = e.sign > 0 ? W_MAG.act : W_MAG.rep;
    return e.sign * (table[e.level] || 0.5);
  }

  /* --------------------------------------------------------------- solver */

  function makeSolver(data) {
    var ids = data.nodes.map(function (n) { return n.id; });
    var idx = {};
    ids.forEach(function (id, i) { idx[id] = i; });

    var N = ids.length;
    var edges = data.edges.map(function (e) {
      return { s: idx[e.source], t: idx[e.target], w: edgeWeight(e), ref: e };
    });

    /**
     * Fixed-point iteration. clamp = {i, value} or null.
     *
     * `init` seeds the state. For the perturbed run we seed from the model's own
     * unperturbed fixed point, so hop 0 shows the clamped gene moving and
     * nothing else.
     */
    function run(clamp, init) {
      var x = init ? Float64Array.from(init) : new Float64Array(N).fill(M.x0);
      if (clamp) x[clamp.i] = clamp.value;

      var traj = [Float64Array.from(x)];
      var hops = 0, converged = false;

      for (var t = 0; t < M.maxIter; t++) {
        var h = new Float64Array(N), r = new Float64Array(N);
        for (var e = 0; e < edges.length; e++) {
          var ed = edges[e],
              xs = Math.max(x[ed.s], 0),
              xn = Math.pow(xs, M.n),
              phi = xn / (Math.pow(M.x0, M.n) + xn + 1e-12),  // K = x0(source)
              drive = ed.w * phi;
          if (drive > 0) h[ed.t] += drive; else r[ed.t] += -drive;
        }

        var next = new Float64Array(N), d2 = 0, i;
        for (i = 0; i < N; i++) {
          next[i] = (M.x0 * (1 + h[i]) + M.b) / (1 + M.alpha + r[i]);
        }
        if (clamp) next[clamp.i] = clamp.value;
        for (i = 0; i < N; i++) { var d = next[i] - x[i]; d2 += d * d; }

        x = next;
        traj.push(Float64Array.from(x));
        hops = t + 1;
        if (Math.sqrt(d2) < M.eps) { converged = true; break; }
      }
      return { traj: traj, hops: hops, converged: converged };
    }

    /** Linear signed random walk with restart — the ablation, kept for questions. */
    function runRWR(clamp, init) {
      var deg = new Float64Array(N).fill(0), i;
      edges.forEach(function (e) { deg[e.t] += 1; });
      for (i = 0; i < N; i++) deg[i] = Math.max(deg[i], 1);

      var x0v = new Float64Array(N).fill(M.x0);
      var x = init ? Float64Array.from(init) : Float64Array.from(x0v);
      if (clamp) x[clamp.i] = clamp.value;

      var traj = [Float64Array.from(x)], hops = 0, converged = false;
      for (var t = 0; t < M.maxIter; t++) {
        var msg = new Float64Array(N);
        edges.forEach(function (e) {
          var s = e.ref.dbSign === 0 ? 1 : e.ref.sign;   // sign-blind fallback
          msg[e.t] += s * x[e.s];
        });
        var next = new Float64Array(N), d2 = 0;
        for (i = 0; i < N; i++) {
          next[i] = M.rwrAlpha * (msg[i] / deg[i]) + (1 - M.rwrAlpha) * x0v[i];
        }
        if (clamp) next[clamp.i] = clamp.value;
        for (i = 0; i < N; i++) { var dd = next[i] - x[i]; d2 += dd * dd; }
        x = next;
        traj.push(Float64Array.from(x));
        hops = t + 1;
        if (Math.sqrt(d2) < M.eps) { converged = true; break; }
      }
      return { traj: traj, hops: hops, converged: converged };
    }

    return { ids: ids, idx: idx, N: N, edges: edges, run: run, runRWR: runRWR };
  }

  /* ---------------------------------------------------------------- colour */

  function lerp(a, b, u) { return a + (b - a) * u; }

  /** Diverging red(up) / blue(down) around neutral, saturating at |d| = cap. */
  function shade(d, cap) {
    var u = Math.max(-1, Math.min(1, d / cap));
    var neu = [230, 234, 242], up = [192, 57, 43], dn = [33, 102, 172];
    var to = u >= 0 ? up : dn, m = Math.abs(u);
    m = Math.pow(m, 0.62);   // ease so small changes stay visible from the back
    return 'rgb(' + [0, 1, 2].map(function (k) {
      return Math.round(lerp(neu[k], to[k], m));
    }).join(',') + ')';
  }

  /* ---------------------------------------------------------------- render */

  function el(tag, attrs, cls) {
    var e = document.createElementNS(NS, tag);
    if (cls) e.setAttribute('class', cls);
    if (attrs) Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    return e;
  }

  function GRNView(host, data) {
    this.data = data;
    this.solver = makeSolver(data);
    this.R = data.nodeRadius;
    this.byId = {};
    data.nodes.forEach(function (n) { this.byId[n.id] = n; }, this);

    var PAD = 34;
    this.svg = el('svg', {
      viewBox: (-PAD) + ' 0 ' + (data.canvas.w + PAD * 2) + ' ' + data.canvas.h,
      preserveAspectRatio: 'xMidYMid meet'
    }, 'grn-svg');

    this.gCols = el('g', null, 'net-cols');
    this.gEdges = el('g', null, 'net-edges');
    this.gNodes = el('g', null, 'net-nodes');
    // Whatever carries the current hop is moved here. SVG paints in document
    // order, so bolding alone is not enough — a thick edge still renders
    // underneath every node drawn after it.
    this.gFront = el('g', null, 'net-front');
    this.gChrome = el('g', null, 'net-chrome');
    [this.gCols, this.gEdges, this.gNodes, this.gFront, this.gChrome].forEach(function (g) {
      this.svg.appendChild(g);
    }, this);

    this._drawColumns();
    this._drawEdges();
    this._drawNodes();
    this._drawChrome();

    host.appendChild(this.svg);
  }

  GRNView.prototype._drawColumns = function () {
    var self = this, H = this.data.canvas.h;
    this.colEls = this.data.columns.map(function (c) {
      var g = el('g', null, 'net-col');
      var lab = el('text', { x: c.x, y: 26 }, 'net-col-label');
      lab.textContent = c.label;
      g.appendChild(lab);
      g.appendChild(el('line', { x1: c.x, x2: c.x, y1: 38, y2: H - 10 }, 'net-col-rule'));
      self.gCols.appendChild(g);
      return { c: c, g: g, label: lab };
    });
  };

  /** Trim a segment so it starts and ends on the node circles, not their centres. */
  function trim(a, b, rA, rB) {
    var dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1;
    var ux = dx / L, uy = dy / L;
    return {
      x1: a.x + ux * rA, y1: a.y + uy * rA,
      x2: b.x - ux * rB, y2: b.y - uy * rB,
      ux: ux, uy: uy
    };
  }

  GRNView.prototype._drawEdges = function () {
    var self = this, R = this.R;
    this.edgeEls = this.data.edges.map(function (e) {
      var a = self.byId[e.source], b = self.byId[e.target];
      var g = el('g', null, 'net-edge-g');
      var seg = trim(a, b, R + 2, R + 9);

      var line = el('line', {
        x1: seg.x1, y1: seg.y1, x2: seg.x2, y2: seg.y2
      }, 'net-edge');
      g.appendChild(line);

      var glyph;
      if (e.sign > 0) {
        var s = 7.5,
            px = -seg.uy, py = seg.ux,
            tipX = seg.x2 + seg.ux * 6.5, tipY = seg.y2 + seg.uy * 6.5;
        glyph = el('polygon', {
          points: [
            tipX + ',' + tipY,
            (seg.x2 - px * s * 0.6) + ',' + (seg.y2 - py * s * 0.6),
            (seg.x2 + px * s * 0.6) + ',' + (seg.y2 + py * s * 0.6)
          ].join(' ')
        }, 'net-glyph act');
      } else {
        var bx = -seg.uy * 7.5, by = seg.ux * 7.5;
        glyph = el('line', {
          x1: seg.x2 - bx, y1: seg.y2 - by, x2: seg.x2 + bx, y2: seg.y2 + by
        }, 'net-glyph rep');
      }
      g.appendChild(glyph);
      self.gEdges.appendChild(g);
      return { e: e, g: g, line: line, glyph: glyph, home: self.gEdges };
    });
  };

  GRNView.prototype._drawNodes = function () {
    var self = this, R = this.R;
    this.nodeEls = this.data.nodes.map(function (n) {
      var g = el('g', { transform: 'translate(' + n.x + ',' + n.y + ')' },
                 'net-node' + (n.seed ? ' seed' : ''));
      if (n.seed) g.appendChild(el('circle', { r: R + 7 }, 'ring'));
      g.appendChild(el('circle', { r: R + 7 }, 'flipring'));
      var body = el('circle', { r: R, fill: '#E6EAF2' }, 'body');
      g.appendChild(body);

      var label = el('text', { y: R + 15 }, 'name');
      label.textContent = n.id;
      g.appendChild(label);

      g.style.cursor = 'pointer';
      g.addEventListener('click', function () { self._onPick && self._onPick(n.id); });

      self.gNodes.appendChild(g);
      return { n: n, g: g, body: body, label: label, home: self.gNodes };
    });
  };

  GRNView.prototype._drawChrome = function () {
    this.readout = el('text', { x: this.data.canvas.w, y: 26, 'text-anchor': 'end' },
                      'net-readout');
    this.gChrome.appendChild(this.readout);
    this.tag = el('text', { x: 0, y: 0, opacity: 0 }, 'net-tag');
    this.tag.textContent = 'CRISPRi';
    this.gChrome.appendChild(this.tag);
  };

  /* ------------------------------------------------------------- painting */

  /** Show edges as the databases give them: unsigned ones grey and dashed. */
  GRNView.prototype.paintEdgesRaw = function () {
    this.edgeEls.forEach(function (E) {
      var uns = E.e.dbSign === 0;
      E.line.setAttribute('class', 'net-edge ' + (uns ? 'uns' : (E.e.sign > 0 ? 'act' : 'rep')));
      E.glyph.setAttribute('class', 'net-glyph ' + (uns ? 'uns' : (E.e.sign > 0 ? 'act' : 'rep')));
    });
  };

  /** After the reveal: every edge carries a direction. */
  GRNView.prototype.paintEdgesResolved = function () {
    this.edgeEls.forEach(function (E) {
      var k = E.e.sign > 0 ? 'act' : 'rep';
      E.line.setAttribute('class', 'net-edge ' + k);
      E.glyph.setAttribute('class', 'net-glyph ' + k);
    });
  };

  /**
   * Bold exactly what carries this hop, mute everything else, and lift the
   * carrying elements into the front layer so nothing paints over them.
   * `liveEdge(e)` and `liveNode(id)` decide membership; passing null clears.
   */
  GRNView.prototype.highlightHop = function (liveEdge, liveNode) {
    var on = !!liveEdge, front = this.gFront;

    this.edgeEls.forEach(function (E) {
      var hot = on && liveEdge(E.e);
      E.g.classList.toggle('carrying', hot);
      E.g.classList.toggle('muted', on && !hot);
      (hot ? front : E.home).appendChild(E.g);
    });
    // nodes after edges inside the front layer, so a label is never crossed by
    // the very edge that is highlighting it
    this.nodeEls.forEach(function (NE) {
      var hot = on && liveNode(NE.n.id);
      NE.g.classList.toggle('carrying', hot);
      NE.g.classList.toggle('muted', on && !hot);
      (hot ? front : NE.home).appendChild(NE.g);
    });
    this.colEls.forEach(function (C) { C.g.classList.remove('active'); });
  };

  GRNView.prototype.markColumn = function (hop) {
    this.colEls.forEach(function (C) {
      C.g.classList.toggle('active', C.c.hop === hop);
    });
  };

  /**
   * Colour every node by its change, and ring the ones that move *against* the
   * prevailing direction.
   *
   * The ring matters: derepressed genes shift by ~+0.07 against a -0.69
   * knockdown, so on a shared colour scale they are a pale wash invisible from
   * the back of a hall. Rescaling the positive half would overstate a small
   * change, so the scale stays honest and the ring carries the emphasis.
   */
  GRNView.prototype.paintNodes = function (delta, cap, opts) {
    var o = opts || {}, against = o.against || 0;
    this.nodeEls.forEach(function (NE, i) {
      var d = delta[i];
      NE.body.setAttribute('fill', shade(d, cap));
      NE.g.classList.toggle('flip', against !== 0 && Math.abs(d) > 5e-4 &&
                                    Math.sign(d) !== against && !NE.n.seed);
    });
  };

  GRNView.prototype.showTag = function (nodeId) {
    var n = this.byId[nodeId];
    if (!n) { this.tag.setAttribute('opacity', 0); return; }
    this.tag.setAttribute('x', n.x);
    this.tag.setAttribute('y', n.y - this.R - 15);
    this.tag.setAttribute('opacity', 1);
  };

  GRNView.prototype.setReadout = function (parts) {
    while (this.readout.firstChild) this.readout.removeChild(this.readout.firstChild);
    parts.forEach(function (p) {
      var t = document.createElementNS(NS, 'tspan');
      if (p.k) t.setAttribute('class', 'k');
      t.textContent = p.text;
      this.readout.appendChild(t);
    }, this);
  };

  GRNView.prototype.onPick = function (fn) { this._onPick = fn; };

  /* ------------------------------------------------------------- playback */

  /**
   * Steps the story. Every phase is idempotent and addressable, so the deck can
   * be driven forwards, backwards, or jumped into from the speaker view.
   *
   * Note on the linear ablation: runRWR is kept for questions, but there is no
   * side-by-side replay. On a subgraph this shallow, linear propagation recovers
   * similar directions at reduced magnitude, so a visual comparison would
   * understate the real gap. The published failure (directional accuracy
   * 0.68 -> 0.53, MSE ~18x) comes from the full 2,000-gene network and is
   * reported there as numbers rather than faked here as a picture.
   */
  function Player(view, opts) {
    this.view = view;
    this.opts = opts || {};
    this.solver = view.solver;

    var S = this.solver;
    this.ctrl = S.run(null).traj.slice(-1)[0];
    this.seedIdx = S.idx[view.data.meta.seed];
    this.kdValue = this.ctrl[this.seedIdx] * M.knockdown;
    this.result = S.run({ i: this.seedIdx, value: this.kdValue }, this.ctrl);

    this.zero = new Float64Array(S.N);
    this.cap = 0.45;
    this.phase = -1;

    // the iteration at which each gene first departs from control
    this.firstMove = {};
    var self = this;
    this.result.traj.forEach(function (x, t) {
      S.ids.forEach(function (id, i) {
        if (self.firstMove[id] === undefined && Math.abs(x[i] - self.ctrl[i]) > 1e-6) {
          self.firstMove[id] = t;
        }
      });
    });

    // walk one column per step until the wavefront stops reaching new genes,
    // then a final step for the remaining settling iterations
    this.lastWave = Math.max.apply(null, Object.keys(this.firstMove)
      .map(function (k) { return self.firstMove[k]; }));
    this.nHops = this.result.hops;
    // phase 0 rest, 1 sign reveal, 2 clamp, 3..(2+lastWave) hops, +1 steady state
    this.lastPhase = 2 + this.lastWave + 1;
  }

  Player.prototype.deltaAt = function (t) {
    var traj = this.result.traj,
        x = traj[Math.max(0, Math.min(t, traj.length - 1))],
        d = new Float64Array(x.length);
    for (var i = 0; i < x.length; i++) d[i] = x[i] - this.ctrl[i];
    return d;
  };

  Player.prototype.caption = function (p) {
    var seed = this.view.data.meta.seed;
    if (p <= 0) return {
      lead: 'The network at rest.',
      note: 'Grey dashed edges are the ones the databases do not sign.'
    };
    if (p === 1) return {
      lead: 'The databases know these edges exist — not which way they push.',
      note: 'iPerturb fits a sign for every one of them.'
    };
    if (p === 2) return {
      lead: 'We hold ' + seed + ' at its measured knockdown level.',
      note: 'Every other gene is free to recompute from its regulators.'
    };
    var hop = p - 2;
    if (hop === 1) return {
      lead: 'Hop 1 — the direct targets fall.',
      note: 'The erythroid programme: haem synthesis, globins, blood-group ' +
            'membrane proteins.'
    };
    if (hop === 2) return {
      lead: 'Hop 2 — the second regulatory layer, and the first genes go <em>up</em>.',
      note: 'HES1 and GAS6 are repressed by regulators that just fell. ' +
            'Lose the repressor, and the target is released.'
    };
    if (hop === 3) return {
      lead: 'Hop 3 — stress and cell-cycle programmes.',
      note: 'CCND1, SERPINE1, ARID5B. The signal is no longer erythroid-specific.'
    };
    if (hop === 4) return {
      lead: 'Hop 4 — the integrated stress response.',
      note: 'ATF4 and its serine-synthesis targets, four regulatory steps from ' +
            'the gene we silenced.'
    };
    if (hop <= this.lastWave) return {
      lead: 'Hop ' + hop + ' — the wavefront reaches the far edge.',
      note: 'Every gene now recomputes from regulators that have themselves moved.'
    };
    return {
      lead: 'Steady state after ' + this.nHops + ' hops.',
      note: 'Median 13 hops on the full 2,000-gene network, with no ' +
            'non-convergent case on either cell line.'
    };
  };

  Player.prototype.go = function (p) {
    p = Math.max(0, Math.min(p, this.lastPhase));
    this.phase = p;
    var V = this.view, S = this.solver, self = this;

    if (p === 0) {
      V.paintEdgesRaw();
      V.paintNodes(this.zero, this.cap);
      V.showTag(null);
      V.highlightHop(null, null);
      V.markColumn(-1);
    } else {
      V.paintEdgesResolved();
    }

    if (p === 1) {
      V.paintNodes(this.zero, this.cap);
      V.showTag(null);
      V.highlightHop(null, null);
      V.markColumn(-1);
    }

    if (p >= 2) {
      V.showTag(V.data.meta.seed);
      var t = Math.min(p - 2, this.result.traj.length - 1);
      var dNow = this.deltaAt(t);
      V.paintNodes(dNow, this.cap, { against: Math.sign(dNow[this.seedIdx]) });

      var hop = p - 2;
      if (hop >= 1 && hop <= this.lastWave) {
        // this hop is carried by edges from genes that had already moved into
        // genes arriving now
        V.highlightHop(
          function (e) {
            return self.firstMove[e.target] === hop &&
                   self.firstMove[e.source] !== undefined &&
                   self.firstMove[e.source] < hop;
          },
          function (id) {
            return self.firstMove[id] !== undefined && self.firstMove[id] <= hop &&
                   (self.firstMove[id] === hop || self.firstMove[id] === hop - 1);
          }
        );
        V.markColumn(hop);
      } else {
        V.highlightHop(null, null);
        V.markColumn(hop === 0 ? 0 : -1);
      }

      var moved = 0;
      for (var i = 0; i < dNow.length; i++) if (Math.abs(dNow[i]) > 1e-6) moved++;
      V.setReadout([
        { text: 'hop ' }, { text: String(Math.max(0, hop)), k: true },
        { text: '  ·  ' + moved + ' of ' + dNow.length + ' genes moved' },
        { text: hop > this.lastWave ? '  ·  converged in ' + this.nHops : '' }
      ]);
    } else {
      V.setReadout([{ text: '' }]);
    }

    if (this.opts.onPhase) this.opts.onPhase(p, this.caption(p), this);
    void S;
  };

  Player.prototype.next = function () { this.go(this.phase + 1); };
  Player.prototype.prev = function () { this.go(this.phase - 1); };
  Player.prototype.reset = function () { this.stop(); this.go(0); };

  Player.prototype.play = function () {
    var self = this;
    this.stop();
    if (this.phase >= this.lastPhase) this.go(0);
    this._timer = setInterval(function () {
      if (self.phase >= self.lastPhase) { self.stop(); return; }
      self.next();
    }, 900);
  };
  Player.prototype.stop = function () {
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
  };
  Player.prototype.playing = function () { return !!this._timer; };

  /** Which incoming edges moved this gene, ranked — the interpretability click. */
  Player.prototype.contributions = function (geneId) {
    var S = this.solver, ti = S.idx[geneId], self = this;
    if (ti === undefined) return [];
    var xs = this.result.traj[this.result.traj.length - 1];
    return S.edges.filter(function (e) { return e.t === ti; }).map(function (e) {
      var xsrc = Math.max(xs[e.s], 0),
          xn = Math.pow(xsrc, M.n),
          phi = xn / (Math.pow(M.x0, M.n) + xn + 1e-12);
      return {
        source: S.ids[e.s], target: geneId, w: e.w, drive: e.w * phi,
        level: e.ref.level, learned: e.ref.learned, db: e.ref.db,
        delta: xs[e.s] - self.ctrl[e.s]
      };
    }).sort(function (a, b) { return Math.abs(b.drive) - Math.abs(a.drive); });
  };

  window.IPerturbGRN = {
    View: GRNView, Player: Player,
    makeSolver: makeSolver, M: M, edgeWeight: edgeWeight
  };
})();
