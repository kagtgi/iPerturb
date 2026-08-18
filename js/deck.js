/* ==========================================================================
   Deck wiring: reveal setup, the two animations, the benchmark charts, and a
   rehearsal timer that reads the per-slide budgets off data-t attributes.
   ========================================================================== */

(function () {
  'use strict';

  var deck = new Reveal({
    width: 1600,
    height: 900,
    margin: 0.04,
    minScale: 0.2,
    maxScale: 1.6,
    hash: true,
    slideNumber: 'c/t',
    transition: 'slide',
    transitionSpeed: 'fast',
    backgroundTransition: 'none',
    controls: true,
    controlsLayout: 'edges',
    progress: true,
    center: false,
    // reveal 6 silently switches to a scrolling page below 435px wide. On a
    // narrow window or an odd projector mode that would stop this being a
    // slideshow in the middle of the talk, so auto-activation is disabled and
    // the slide view is pinned.
    view: null,
    scrollActivationWidth: null,
    // fragments in the caption stacks must not be skipped over in speaker view
    fragmentInURL: true,
    plugins: [RevealNotes]
  });

  /* ---------------------------------------------------- the GRN animation */

  function initPropagation() {
    var host = document.getElementById('grn-stage');
    if (!host || host.dataset.ready) return;

    var data = window.K562_GATA1;
    if (!data) {
      document.getElementById('grn-lead').textContent =
        'Could not load data/k562-gata1.js — run tools/build_subgraph.py.';
      return;
    }
    host.dataset.ready = '1';

    var lead = document.getElementById('grn-lead'),
        note = document.getElementById('grn-note'),
        hops = document.getElementById('grn-hops'),
        playBtn = document.getElementById('grn-play');

    var view = new IPerturbGRN.View(host, data);
    var player = new IPerturbGRN.Player(view, {
      onPhase: function (p, cap, self) {
        lead.innerHTML = cap.lead;
        note.innerHTML = cap.note;
        hops.textContent = 'phase ' + p + ' / ' + self.lastPhase;
        playBtn.textContent = self.playing() ? '❚❚ Pause' : '▶ Play';
      }
    });
    window.__grnPlayer = player;

    // clicking a settled gene shows which edges moved it
    view.onPick(function (geneId) {
      if (player.phase < 2) return;
      var rows = player.contributions(geneId);
      if (!rows.length) {
        note.innerHTML = '<strong>' + geneId + '</strong> has no regulator in this sub-network.';
        return;
      }
      note.innerHTML =
        '<strong>' + geneId + '</strong> was moved by: ' +
        rows.map(function (r) {
          return '<span style="color:' + (r.w > 0 ? 'var(--act)' : 'var(--rep)') + '">' +
                 r.source + (r.w > 0 ? ' →' : ' ⊣') + '</span> ' +
                 '(w = ' + (r.w > 0 ? '+' : '') + r.w.toFixed(2) +
                 ', L' + r.level + (r.learned ? ', sign learned' : '') + ')';
        }).join(' &nbsp;·&nbsp; ');
    });

    playBtn.addEventListener('click', function () {
      if (player.playing()) { player.stop(); playBtn.textContent = '▶ Play'; }
      else player.play();
    });
    document.getElementById('grn-step').addEventListener('click', function () {
      player.stop(); player.next();
    });
    document.getElementById('grn-back').addEventListener('click', function () {
      player.stop(); player.prev();
    });
    document.getElementById('grn-reset').addEventListener('click', function () {
      player.reset();
    });

    player.go(0);
  }

  /* ------------------------------------------- the whole-network overview */

  function initFullNet() {
    var host = document.getElementById('fullnet-stage');
    if (!host || host.dataset.ready) return;

    var D = window.K562_FULL;
    if (!D) {
      document.getElementById('fullnet-lead').textContent =
        'Could not load data/k562-full.js — run tools/build_fullnet.py.';
      return;
    }
    host.dataset.ready = '1';

    var lead = document.getElementById('fullnet-lead'),
        note = document.getElementById('fullnet-note'),
        hops = document.getElementById('fullnet-hops'),
        playBtn = document.getElementById('fullnet-play');

    var view = new IPerturbFullNet.View(host, D);
    window.__fullNet = view;

    view.onHop = function (hop, self) {
      var reached = self.reached(), total = D.ids.length;
      if (hop === 0) {
        lead.textContent = 'GATA1 held at its knockdown level. Nothing else has moved yet.';
        note.textContent = '';
      } else if (hop <= self.lastWave) {
        lead.innerHTML = 'Hop <strong>' + hop + '</strong> — the wavefront spreads.';
        note.textContent = reached.toLocaleString() + ' of ' +
          total.toLocaleString() + ' genes reached.';
      } else {
        lead.innerHTML = 'Steady state — <strong>' + self.res.hops + ' hops</strong>, ' +
          reached.toLocaleString() + ' of ' + total.toLocaleString() + ' genes moved.';
        note.textContent = 'Median 13 hops across all perturbations, ' +
          'with no non-convergent case on either cell line.';
      }
      hops.textContent = 'hop ' + hop + ' / ' + self.lastWave;
      playBtn.textContent = self.playing() ? '❚❚ Pause' : '▶ Play';
    };

    playBtn.addEventListener('click', function () {
      if (view.playing()) { view.stop(); playBtn.textContent = '▶ Play'; }
      else view.play();
    });
    document.getElementById('fullnet-step').addEventListener('click', function () {
      view.stop(); view.next();
    });
    document.getElementById('fullnet-back').addEventListener('click', function () {
      view.stop(); view.prev();
    });
    document.getElementById('fullnet-reset').addEventListener('click', function () {
      view.stop(); view.go(0);
    });

    view.go(0);
  }

  /* -------- drive the animation from the spacebar while on its slide ------ */

  function onPropagationSlide() {
    var s = deck.getCurrentSlide();
    return s && s.id === 'slide-propagation';
  }

  function onFullNetSlide() {
    var s = deck.getCurrentSlide();
    return s && s.id === 'slide-fullnet';
  }

  /**
   * Both animated slides advance on the normal presentation keys, so the talk is
   * driven with one clicker throughout. We only swallow the key while the
   * animation still has steps left; after that it falls through to reveal and
   * moves to the next slide.
   */
  document.addEventListener('keydown', function (ev) {
    var anim = null, at = 0, last = 0;
    if (onPropagationSlide() && window.__grnPlayer) {
      anim = window.__grnPlayer; at = anim.phase; last = anim.lastPhase;
    } else if (onFullNetSlide() && window.__fullNet) {
      anim = window.__fullNet; at = anim.hop; last = anim.lastWave + 1;
    }
    if (!anim) return;

    if (ev.key === ' ' || ev.key === 'ArrowRight' || ev.key === 'PageDown') {
      if (at < last) {
        ev.preventDefault(); ev.stopPropagation();
        anim.stop(); anim.next();
      }
    } else if (ev.key === 'ArrowLeft' || ev.key === 'PageUp') {
      if (at > 0) {
        ev.preventDefault(); ev.stopPropagation();
        anim.stop(); anim.prev();
      }
    }
  }, true);

  /* ------------------------------------------------ rehearsal time budget */

  function budget(skipCut) {
    var secs = 0;
    document.querySelectorAll('.slides section[data-t]').forEach(function (s) {
      if (s.dataset.optional === 'true' || s.dataset.backup === 'true') return;
      if (skipCut && s.dataset.cut === '15') return;
      var m = /(\d+):(\d+)/.exec(s.dataset.t);
      if (m) secs += (+m[1]) * 60 + (+m[2]);
    });
    return secs;
  }
  function mmss(s) { return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); }

  /* ------------------------------------------------------------- lifecycle */

  /**
   * Typeset the LaTeX before anything measures or highlights it. MathJax is
   * configured with startup.typeset = false so this happens once, after reveal
   * has laid the slides out — otherwise the equations get sized against a
   * pre-layout container and land at the wrong scale.
   */
  function typesetMath() {
    if (!window.MathJax || !window.MathJax.typesetPromise) return Promise.resolve();
    // Typeset the whole deck rather than tagged blocks only: MathJax converts
    // just what sits between delimiters, so this costs nothing extra and means
    // inline \( ... \) works anywhere without remembering to tag its container.
    return window.MathJax.typesetPromise(
      Array.from(document.querySelectorAll('.reveal .slides'))
    ).catch(function (err) {
      console.error('MathJax failed to typeset — equations will show as raw LaTeX', err);
    });
  }

  deck.initialize().then(function () {
    return typesetMath();
  }).then(function () {
    IPerturbFormula.register(deck);

    function onSlide() {
      IPerturbResults.render(document);
      if (onPropagationSlide()) initPropagation();
      if (onFullNetSlide()) { initFullNet(); if (window.__fullNet) window.__fullNet._resize(); }
    }
    deck.on('slidechanged', onSlide);
    onSlide();

    var full = budget(false), lean = budget(true),
        nFull = document.querySelectorAll('.slides section[data-t]:not([data-optional]):not([data-backup])').length,
        nCut = document.querySelectorAll('.slides section[data-cut="15"]').length;
    console.info(
      'iPerturb deck\n' +
      '  full talk : ' + mmss(full) + ' over ' + nFull + ' slides\n' +
      '  15-min cut: ' + mmss(lean) + ' — skip the ' + nCut +
      ' slides marked data-cut="15", then trim the two animation slides\n' +
      '  press S for speaker notes');
  });

  window.__deck = deck;
})();
