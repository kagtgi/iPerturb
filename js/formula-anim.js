/* ==========================================================================
   Animation 1 — the Hill equation, term by term.

   Two pieces:
     1. Fragment-driven highlighting. Each caption fragment declares which
        symbol it is talking about (data-focus) and in which semantic colour
        (data-tone); every .tok carrying that term lights up while the rest of
        the equation dims. Captions are stacked in a fixed-height box so the
        equation never moves.
     2. A live Hill curve with K and n sliders, started at the model's own
        initialisation (n = 2.5, K = x0) where occupancy is exactly 0.5.
   ========================================================================== */

(function () {
  'use strict';

  /* ------------------------------------------------------ highlighting ---- */

  function formulaeIn(slide) {
    return Array.from(slide.querySelectorAll('.formula'));
  }

  var TONES = ['c-navy', 'c-act', 'c-rep', 'c-brick'];

  /**
   * Apply a focus state to every formula on the slide.
   *
   * MathJax's \class{}{} puts our marker classes straight onto the rendered SVG
   * groups, so a term is addressed as `.tok-V` regardless of how many glyphs
   * LaTeX decided it needs. Colour travels through `currentColor`, which MathJax
   * uses for every glyph fill.
   */
  function applyFocus(slide, terms, tone) {
    var list = terms ? terms.split(/\s+/).filter(Boolean) : [];
    formulaeIn(slide).forEach(function (f) {
      f.classList.toggle('focusing', list.length > 0);
      f.querySelectorAll('.tok').forEach(function (tok) {
        tok.classList.remove.apply(tok.classList, ['lit'].concat(TONES));
      });
      list.forEach(function (term) {
        f.querySelectorAll('.tok-' + term).forEach(function (tok) {
          tok.classList.add('lit', tone || 'c-navy');
        });
      });
    });
  }

  /**
   * Recompute focus from whichever caption is currently reached, and paint that
   * caption ourselves.
   *
   * The captions are stacked absolutely in a fixed-height box, so exactly one
   * may be visible at a time. We set that inline rather than leaning on
   * reveal's `current-visible` styling: the two interact badly once the
   * fragments are taken out of flow, and an inline style is not something a
   * future reveal upgrade can quietly change underneath us.
   *
   * Driven off the DOM rather than off event order, so stepping backwards,
   * jumping by hash, or landing mid-slide from the speaker view all behave
   * identically.
   */
  function syncSlide(slide) {
    if (!slide || !slide.querySelector('.formula')) return;
    var caps = Array.from(slide.querySelectorAll('.formula-caption .cap'));
    if (!caps.length) return;

    var active = null;
    caps.forEach(function (c) {
      if (c.classList.contains('visible')) active = c;   // last reached wins
    });
    if (!active) active = caps[0];        // landing on the slide shows the first

    caps.forEach(function (c) {
      var on = c === active;
      c.classList.toggle('on', on);
      c.style.opacity = on ? '1' : '0';
      c.style.visibility = on ? 'visible' : 'hidden';
    });

    applyFocus(slide, active.dataset.focus, active.dataset.tone);
    driveHill(slide, (active.dataset.focus || '').trim(), active.dataset.tone);
  }

  /* -------------------------------------------------------- hill curve ---- */

  var HILL = {
    W: 520, H: 320,
    pad: { l: 54, r: 18, t: 16, b: 46 },
    xMax: 2.0,
    // model initialisation: n_raw = 0 -> n = 1 + 3*sigmoid(0) = 2.5
    //                       softplus(Kd_raw) = x0(source) -> occupancy(x0) = 0.5
    n0: 2.5, K0: 1.0
  };

  /**
   * What the curve does on each step.
   *
   * There are no sliders: nobody in an audience can reach them, and a presenter
   * dragging one mid-sentence loses the room. Each caption instead declares a
   * short demo, and the curve plays it automatically — the parameter sweeps to
   * its extremes and settles back, so the point is made by watching rather than
   * by fiddling.
   */
  var DEMO = {
    // term      colour      K keyframes            n keyframes
    h: { tone: '#1B7F4B' },
    r: { tone: '#B4531F' },
    w: { tone: '#9B3A3A', split: true },
    K: { tone: '#9B3A3A', sweep: 'K', keys: [1.0, 0.45, 1.6, 1.0] },
    n: { tone: '#9B3A3A', sweep: 'n', keys: [2.5, 1.05, 4.0, 2.5] }
  };

  function hillPath(K, n) {
    var p = HILL.pad,
        w = HILL.W - p.l - p.r,
        h = HILL.H - p.t - p.b,
        d = '', N = 160;
    for (var i = 0; i <= N; i++) {
      var x = HILL.xMax * i / N,
          y = Math.pow(x, n) / (Math.pow(K, n) + Math.pow(x, n) + 1e-12),
          px = p.l + (x / HILL.xMax) * w,
          py = p.t + (1 - y) * h;
      d += (i ? 'L' : 'M') + px.toFixed(2) + ' ' + py.toFixed(2);
    }
    return d;
  }

  function buildHill(host) {
    var p = HILL.pad,
        w = HILL.W - p.l - p.r,
        h = HILL.H - p.t - p.b,
        NS = 'http://www.w3.org/2000/svg';

    var svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + HILL.W + ' ' + HILL.H);
    svg.setAttribute('class', 'hill-svg');
    svg.style.width = '100%';
    svg.style.height = 'auto';

    function el(tag, attrs) {
      var e = document.createElementNS(NS, tag);
      Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
      return e;
    }

    // half-occupancy guide
    svg.appendChild(el('line', {
      x1: p.l, x2: p.l + w, y1: p.t + h / 2, y2: p.t + h / 2,
      stroke: '#DDE2ED', 'stroke-width': 1, 'stroke-dasharray': '4 4'
    }));
    // axes
    svg.appendChild(el('line', { x1: p.l, x2: p.l + w, y1: p.t + h, y2: p.t + h, stroke: '#98A2B3', 'stroke-width': 1.4 }));
    svg.appendChild(el('line', { x1: p.l, x2: p.l, y1: p.t, y2: p.t + h, stroke: '#98A2B3', 'stroke-width': 1.4 }));

    function label(x, y, text, anchor, size, fill) {
      var t = el('text', { x: x, y: y, 'text-anchor': anchor || 'middle',
                           'font-size': size || 13, fill: fill || '#5F6980' });
      t.textContent = text;
      return t;
    }
    svg.appendChild(label(p.l - 10, p.t + 5, '1', 'end'));
    svg.appendChild(label(p.l - 10, p.t + h / 2 + 5, '0.5', 'end'));
    svg.appendChild(label(p.l - 10, p.t + h + 4, '0', 'end'));
    svg.appendChild(label(p.l + w / 2, HILL.H - 8, 'regulator level  xₖ', 'middle'));

    var yTitle = label(15, p.t + h / 2, 'occupancy  φ', 'middle');
    yTitle.setAttribute('transform', 'rotate(-90 15 ' + (p.t + h / 2) + ')');
    svg.appendChild(yTitle);

    // K marker + curve
    var kLine = el('line', { y1: p.t, y2: p.t + h, stroke: '#9B3A3A',
                             'stroke-width': 1.4, 'stroke-dasharray': '5 4', opacity: 0.75 });
    svg.appendChild(kLine);

    // ghost of the resting shape, so a sweep reads as a change from something
    var ghost = el('path', { fill: 'none', stroke: '#C6CDDC', 'stroke-width': 2,
                             'stroke-dasharray': '5 5', opacity: 0 });
    svg.appendChild(ghost);

    // second curve, used only when the two signs are being contrasted
    var curve2 = el('path', { fill: 'none', stroke: '#B4531F', 'stroke-width': 3,
                              'stroke-linejoin': 'round', 'stroke-linecap': 'round',
                              opacity: 0 });
    svg.appendChild(curve2);

    var curve = el('path', { fill: 'none', stroke: '#0A1F5C', 'stroke-width': 3.4,
                             'stroke-linejoin': 'round', 'stroke-linecap': 'round' });
    svg.appendChild(curve);

    var dot = el('circle', { r: 6, fill: '#9B3A3A', stroke: '#fff', 'stroke-width': 2 });
    svg.appendChild(dot);

    var kTag = label(0, p.t + h + 20, 'K', 'middle', 13, '#9B3A3A');
    svg.appendChild(kTag);

    // live parameter readout — replaces the sliders
    var readK = label(p.l + w, p.t + 4, '', 'end', 15, '#9B3A3A');
    readK.setAttribute('font-weight', '700');
    svg.appendChild(readK);

    host.appendChild(svg);

    return {
      svg: svg,
      render: function (K, n, opts) {
        var o = opts || {};
        curve.setAttribute('d', hillPath(K, n));
        curve.setAttribute('stroke', o.tone || '#0A1F5C');
        var kx = p.l + Math.min(K / HILL.xMax, 1) * w;
        kLine.setAttribute('x1', kx); kLine.setAttribute('x2', kx);
        dot.setAttribute('cx', kx);
        dot.setAttribute('cy', p.t + h / 2);       // occupancy is ½ at x = K, always
        dot.setAttribute('fill', o.tone || '#9B3A3A');
        kTag.setAttribute('x', kx);
        readK.textContent = o.readout || '';
      },
      ghost: function (on) {
        ghost.setAttribute('d', hillPath(HILL.K0, HILL.n0));
        ghost.setAttribute('opacity', on ? 0.9 : 0);
      },
      second: function (on, K, n) {
        curve2.setAttribute('opacity', on ? 1 : 0);
        if (on) curve2.setAttribute('d', hillPath(K, n));
      }
    };
  }

  function ease(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }

  /** Walk a parameter through keyframes and settle, driving the curve as it goes. */
  function sweep(panel, which, keys, tone, done) {
    var st = panel._hill;
    cancelAnimationFrame(st.raf);
    var legs = keys.length - 1, LEG = 850, total = legs * LEG, t0 = null;

    // Paint the settled state up front. requestAnimationFrame is throttled when
    // a window is backgrounded or a projector mirrors oddly; if it never fires,
    // the slide must still show the right curve rather than a stale one.
    st.api.render(HILL.K0, HILL.n0, {
      tone: tone,
      readout: which === 'K' ? 'K = ' + HILL.K0.toFixed(2) : 'n = ' + HILL.n0.toFixed(1)
    });

    st.api.ghost(true);
    function frame(ts) {
      if (t0 === null) t0 = ts;
      var el = Math.min(ts - t0, total), i = Math.min(Math.floor(el / LEG), legs - 1),
          u = ease((el - i * LEG) / LEG),
          v = keys[i] + (keys[i + 1] - keys[i]) * u;
      var K = which === 'K' ? v : HILL.K0,
          n = which === 'n' ? v : HILL.n0;
      st.api.render(K, n, {
        tone: tone,
        readout: which === 'K' ? 'K = ' + K.toFixed(2) : 'n = ' + n.toFixed(1)
      });
      if (el < total) st.raf = requestAnimationFrame(frame);
      else { st.api.ghost(false); if (done) done(); }
    }
    st.raf = requestAnimationFrame(frame);
  }

  /** Put the curve in the state the current caption is talking about. */
  function driveHill(slide, term, tone) {
    var panel = slide.querySelector('.hill-panel');
    if (!panel || !panel._hill) return;
    var st = panel._hill, d = DEMO[term];
    cancelAnimationFrame(st.raf);
    st.api.ghost(false);
    st.api.second(false);

    if (!d) {
      st.api.render(HILL.K0, HILL.n0, { readout: '' });
      st.caption.textContent = st.captionDefault;
      return;
    }
    st.caption.textContent = st.captions[term] || st.captionDefault;

    if (d.split) {                       // one edge, either sign
      st.api.render(HILL.K0, HILL.n0, { tone: '#1B7F4B', readout: '' });
      st.api.second(true, HILL.K0, HILL.n0);
      return;
    }
    if (d.sweep) { sweep(panel, d.sweep, d.keys, d.tone); return; }
    st.api.render(HILL.K0, HILL.n0, { tone: d.tone, readout: '' });
  }

  function initHill(panel) {
    if (panel.dataset.ready) return;
    panel.dataset.ready = '1';
    var api = buildHill(panel.querySelector('.hill-plot'));
    var cap = panel.querySelector('.hill-caption');
    panel._hill = {
      api: api, raf: 0, caption: cap,
      captionDefault: cap.textContent.trim(),
      captions: {
        h: 'An activating edge: more regulator, more drive — until it saturates.',
        r: 'A repressing edge has the same shape; only its sign differs.',
        w: 'Same curve, two signs — green pushes up, rust pushes down.',
        K: 'Watch the threshold slide: K sets where half-effect happens.',
        n: 'Watch it sharpen: n turns a graded dial into an on/off switch.'
      }
    };
    api.render(HILL.K0, HILL.n0, { readout: '' });
  }

  /* ------------------------------------------------------------- wiring --- */

  window.IPerturbFormula = {
    register: function (deck) {
      function refresh() {
        var slide = deck.getCurrentSlide();
        if (!slide) return;
        slide.querySelectorAll('.hill-panel').forEach(initHill);
        syncSlide(slide);
      }
      ['ready', 'slidechanged', 'fragmentshown', 'fragmenthidden'].forEach(function (ev) {
        deck.on(ev, refresh);
      });
    }
  };
})();
