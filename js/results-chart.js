/* ==========================================================================
   Benchmark charts, drawn natively from figs/fig3_values.csv.

   Rebuilt rather than rasterised so the panels can be revealed one metric at a
   time, stay crisp on a projector, and use the deck's own palette. The numbers
   below are transcribed verbatim from that file (means; sd in SD).
   ========================================================================== */

(function () {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';

  var METHODS = ['CPA', 'GEARS', 'scGPT', 'STATE', 'Cell2Sentence', 'iPerturb'];
  var SHORT   = { CPA: 'CPA', GEARS: 'GEARS', scGPT: 'scGPT', STATE: 'STATE',
                  Cell2Sentence: 'C2S', iPerturb: 'iPerturb' };

  var DATA = {
    directional: {
      K562: [0.561, 0.651, 0.681, 0.701, 0.631, 0.721],
      RPE1: [0.561, 0.671, 0.681, 0.731, 0.641, 0.881]
    },
    mse: {
      K562: [0.711, 0.571, 0.521, 0.481, 0.601, 0.201],
      RPE1: [0.741, 0.521, 0.461, 0.411, 0.551, 0.231]
    },
    centroid: {
      K562: [0.452, 0.532, 0.552, 0.572, 0.522, 0.612],
      RPE1: [0.432, 0.572, 0.582, 0.612, 0.552, 0.622]
    },
    pearson_d20: {
      K562: [0.07, 0.31, 0.46, 0.50, 0.29, 0.61],
      RPE1: [0.21, 0.55, 0.66, 0.74, 0.53, 0.72]
    },
    pearson_d: {
      K562: [0.06, 0.22, 0.29, 0.34, 0.18, 0.34],
      RPE1: [0.10, 0.48, 0.52, 0.62, 0.42, 0.71]
    }
  };

  var SD = {
    directional: { K562: [0.03, 0.02, 0.02, 0.01, 0.02, 0.01],
                   RPE1: [0.03, 0.02, 0.02, 0.01, 0.02, 0.02] },
    mse:         { K562: [0.05, 0.03, 0.02, 0.02, 0.03, 0.03],
                   RPE1: [0.05, 0.03, 0.02, 0.02, 0.03, 0.01] },
    centroid:    { K562: [0.03, 0.02, 0.02, 0.00, 0.02, 0.03],
                   RPE1: [0.03, 0.02, 0.02, 0.00, 0.02, 0.03] }
  };

  function el(tag, attrs, cls) {
    var e = document.createElementNS(NS, tag);
    if (cls) e.setAttribute('class', cls);
    if (attrs) Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    return e;
  }
  function text(x, y, s, opts) {
    var o = opts || {};
    var t = el('text', {
      x: x, y: y,
      'text-anchor': o.anchor || 'middle',
      'font-size': o.size || 14,
      'font-weight': o.weight || 400,
      fill: o.fill || '#5F6980'
    });
    t.textContent = s;
    return t;
  }

  /**
   * Grouped bars: one group per cell line, one bar per method.
   * `lowerIsBetter` flips which bar gets the win highlight.
   */
  function barChart(host, metric, opts) {
    var o = opts || {},
        W = o.width || 660, H = o.height || 330,
        pad = { l: 52, r: 14, t: 18, b: 62 },
        cells = ['K562', 'RPE1'],
        lower = !!o.lowerIsBetter,
        vmax = o.vmax || 1.0;

    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H });
    svg.style.width = '100%'; svg.style.height = 'auto';

    var pw = W - pad.l - pad.r, ph = H - pad.t - pad.b,
        groupW = pw / cells.length,
        barW = Math.min(38, (groupW - 40) / METHODS.length);

    // y grid
    [0, 0.25, 0.5, 0.75, 1].forEach(function (f) {
      var y = pad.t + ph * (1 - f);
      svg.appendChild(el('line', {
        x1: pad.l, x2: pad.l + pw, y1: y, y2: y,
        stroke: '#EEF1F7', 'stroke-width': 1
      }));
      svg.appendChild(text(pad.l - 9, y + 5, (f * vmax).toFixed(2).replace(/0$/, ''),
                           { anchor: 'end', size: 12.5, fill: '#98A2B3' }));
    });
    svg.appendChild(el('line', {
      x1: pad.l, x2: pad.l + pw, y1: pad.t + ph, y2: pad.t + ph,
      stroke: '#98A2B3', 'stroke-width': 1.4
    }));

    cells.forEach(function (cell, ci) {
      var vals = DATA[metric][cell],
          sds = (SD[metric] || {})[cell],
          best = lower ? Math.min.apply(null, vals) : Math.max.apply(null, vals),
          gx = pad.l + ci * groupW,
          span = METHODS.length * (barW + 7) - 7,
          x0 = gx + (groupW - span) / 2;

      METHODS.forEach(function (m, mi) {
        var v = vals[mi],
            h = Math.max(2, (v / vmax) * ph),
            x = x0 + mi * (barW + 7),
            y = pad.t + ph - h,
            ours = m === 'iPerturb',
            win = Math.abs(v - best) < 1e-9;

        var bar = el('rect', {
          x: x, y: y, width: barW, height: h, rx: 3,
          fill: ours ? '#9B3A3A' : '#C6CDDC'
        });
        bar.setAttribute('class', 'bench-bar' + (ours ? ' ours' : ''));
        svg.appendChild(bar);

        if (sds && sds[mi] > 0) {
          var e2 = (sds[mi] / vmax) * ph, cx = x + barW / 2;
          svg.appendChild(el('line', {
            x1: cx, x2: cx, y1: y - e2, y2: y + e2,
            stroke: ours ? '#6E2626' : '#98A2B3', 'stroke-width': 1.4
          }));
        }
        if (win) {
          svg.appendChild(text(x + barW / 2, y - 9, v.toFixed(2),
                               { size: 14, weight: 700, fill: '#9B3A3A' }));
        }
        // method label, rotated so neighbouring labels can never collide
        var lb = text(0, 0, SHORT[m], { size: 12.5, anchor: 'end',
                                        fill: ours ? '#9B3A3A' : '#5F6980',
                                        weight: ours ? 700 : 400 });
        lb.setAttribute('transform',
          'translate(' + (x + barW / 2 + 4) + ',' + (pad.t + ph + 12) + ') rotate(-42)');
        svg.appendChild(lb);
      });

      svg.appendChild(text(gx + groupW / 2, H - 8, cell,
                           { size: 15, weight: 700, fill: '#0A1F5C' }));
    });

    host.appendChild(svg);
    return svg;
  }

  window.IPerturbResults = {
    render: function (root) {
      (root || document).querySelectorAll('[data-chart]').forEach(function (host) {
        if (host.dataset.ready) return;
        host.dataset.ready = '1';
        barChart(host, host.dataset.chart, {
          lowerIsBetter: host.dataset.lower === 'true',
          vmax: parseFloat(host.dataset.vmax || '1')
        });
      });
    },
    DATA: DATA, METHODS: METHODS
  };
})();
