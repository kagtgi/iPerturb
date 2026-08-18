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

  /**
   * The two animated slides (GATA1 propagation, whole-network) are driven by a
   * bespoke Player object, not reveal's native fragments. Route Up/Down to
   * whichever mechanism the current slide actually uses, and do nothing at
   * either end rather than falling through to a slide change.
   */
  function currentAnim() {
    var s = deck.getCurrentSlide();
    if (!s) return null;
    if (s.id === 'slide-propagation' && window.__grnPlayer) {
      var p = window.__grnPlayer;
      return { obj: p, at: p.phase, last: p.lastPhase };
    }
    if (s.id === 'slide-fullnet' && window.__fullNet) {
      var f = window.__fullNet;
      return { obj: f, at: f.hop, last: f.lastWave + 1 };
    }
    return null;
  }

  function stepAnimForward() {
    var a = currentAnim();
    if (a) { if (a.at < a.last) { a.obj.stop(); a.obj.next(); } return; }
    deck.nextFragment();
  }
  function stepAnimBackward() {
    var a = currentAnim();
    if (a) { if (a.at > 0) { a.obj.stop(); a.obj.prev(); } return; }
    deck.prevFragment();
  }

  /**
   * Left/Right/PageUp/PageDown/Space always change the slide, full stop —
   * they never get intercepted to step a fragment or an animation first.
   * Up/Down do the opposite: they only ever step whatever is animated on the
   * current slide (a custom Player, or a native reveal fragment) and never
   * change the slide themselves.
   *
   * `addKeyBinding` (populating reveal's internal `this.bindings`) is the
   * mechanism that actually overrides a key's default behaviour in this
   * version of reveal — the documented-looking `keyboard: {37: fn, ...}`
   * constructor option is checked against `config.keyboard`, which reveal's
   * own config merge collapses back down to its default boolean `true`
   * whenever a plain object is supplied instead of `false`/`true`. That
   * silently drops the whole override with no error, so verify with
   * `deck.getConfig().keyboard` after any reveal upgrade — if it isn't the
   * object you passed, bindings need to move (back) to `addKeyBinding`.
   *
   * `left`/`right` are fragment-aware by default in this reveal version —
   * bare `deck.right()` on a slide with hidden fragments reveals the next
   * fragment instead of changing slide. `{skipFragments: true}` forces pure
   * slide navigation; it is the same option reveal's own default Alt+Arrow
   * handling passes internally. `nextFragment`/`prevFragment` are the
   * fragment-only primitives that no-op at the start/end of a slide instead
   * of spilling into slide navigation.
   */
  function bindKeys() {
    deck.addKeyBinding(37, function () { deck.left({ skipFragments: true }); });   // Left
    deck.addKeyBinding(39, function () { deck.right({ skipFragments: true }); });  // Right
    deck.addKeyBinding(33, function () { deck.left({ skipFragments: true }); });   // Page Up
    deck.addKeyBinding(34, function () { deck.right({ skipFragments: true }); });  // Page Down
    deck.addKeyBinding(32, function () { deck.right({ skipFragments: true }); });  // Space
    deck.addKeyBinding(38, function () { stepAnimBackward(); });                   // Up
    deck.addKeyBinding(40, function () { stepAnimForward(); });                    // Down
  }

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

  /* -------------------- which slide is currently showing ------------------ */

  function onPropagationSlide() {
    var s = deck.getCurrentSlide();
    return s && s.id === 'slide-propagation';
  }

  function onFullNetSlide() {
    var s = deck.getCurrentSlide();
    return s && s.id === 'slide-fullnet';
  }

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
    bindKeys();
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
