(function () {
  'use strict';

  let puzzle = null;
  let timerInterval = null;
  let timerSeconds = 0;
  let completed = false;

  const PD = window.PUZZLE_DATA;
  if (!PD) {
    document.getElementById('puzzle-title').textContent = 'No puzzle data';
    return;
  }

  function initPuzzle() {
    const container = document.getElementById('puzzle-container');
    puzzle = new pzpr.Puzzle(container, {
      type: 'player',
      callback: function () {
        puzzle.setMode('play');
        const cellsize = Math.min(
          Math.floor((container.clientWidth - 40) / PD.width),
          Math.floor(480 / PD.height),
          50
        );
        puzzle.setCanvasSize(cellsize);
      }
    }).open(PD.pzpr_url);

    setInterval(function () {
      if (!completed) checkCompletion();
    }, 300);
  }

  function loadDetailedRules(puzzleType) {
    let capturedRules = null;
    let capturedData = null;

    window.ui = window.ui || {};
    window.ui.debug = window.ui.debug || {};
    window.ui.debug.addRules = function (pid, rules) { capturedRules = rules; };
    window.ui.debug.addDebugData = function (pid, data) { capturedData = data; };

    const pid = pzpr.variety.toPID(puzzleType) || puzzleType;
    const script = document.createElement('script');
    script.src = PD.base_path + 'vendor/pzprjs/dist/js/pzpr-samples/' + pid + '.js';
    script.onload = function () {
      if (capturedRules && capturedData) renderDetailedRules(capturedRules, capturedData, pid);
    };
    script.onerror = function () {};
    document.head.appendChild(script);
  }

  function renderDetailedRules(rulesInfo, debugData, pid) {
    const detailedRules = rulesInfo[0].rules;
    if (detailedRules) {
      const rulesEl = document.getElementById('rules-text');
      const lines = detailedRules.split('\n').filter(l => l.trim());
      rulesEl.innerHTML = lines.map(l => '<p>' + escapeHtml(l) + '</p>').join('');
    }
    if (!debugData.failcheck || debugData.failcheck.length === 0) return;

    document.getElementById('examples-section').style.display = '';
    const container = document.getElementById('rules-examples');
    debugData.failcheck.forEach(function (entry) {
      if (entry.length > 2 && entry[2] && entry[2].skiprules) return;
      const boardState = entry[1];
      const example = document.createElement('div');
      example.className = 'rule-example';
      const canvasDiv = document.createElement('div');
      canvasDiv.className = 'rule-example-canvas';
      canvasDiv.style.width = '150px';
      canvasDiv.style.height = '150px';
      example.appendChild(canvasDiv);
      const labelDiv = document.createElement('div');
      labelDiv.className = 'rule-example-label';
      example.appendChild(labelDiv);
      container.appendChild(example);
      var p = new pzpr.Puzzle(canvasDiv, { type: 'viewer' });
      p.setConfig('autocmp', false);
      p.setConfig('forceallcell', true);
      p.once('ready', function () {
        var result = p.check(true);
        var isOK = result.complete;
        var msg = result.gettext('en');
        labelDiv.className = 'rule-example-label ' + (isOK ? 'ok' : 'ng');
        labelDiv.textContent = (isOK ? 'OK' : 'NG') + (msg ? ': ' + msg : '');
        p.setCanvasSize(20);
      });
      p.open(boardState);
    });
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function startTimer() {
    if (timerInterval) return;
    timerInterval = setInterval(function () {
      timerSeconds++;
      const min = Math.floor(timerSeconds / 60);
      const sec = timerSeconds % 60;
      document.getElementById('timer').textContent = min + ':' + (sec < 10 ? '0' : '') + sec;
    }, 1000);
  }

  function formatTimerStyle(seconds) {
    const min = Math.floor(seconds / 60);
    const sec = seconds % 60;
    return min + ':' + (sec < 10 ? '0' : '') + sec;
  }

  function showComplete() {
    var msgEl = document.getElementById('check-message');
    msgEl.textContent = 'Complete';
    msgEl.className = 'check-message complete';
  }

  function checkCompletion() {
    if (!puzzle) return;
    const result = puzzle.check(false);
    if (result.complete) {
      completed = true;
      clearInterval(timerInterval);
      timerInterval = null;
      showComplete();
      showCompletionModal();
    }
  }

  function obfuscateShare(idx, seconds) {
    var key = 'sillysecretpencilpuzzle';
    var plain = idx + '|' + seconds;
    var out = [];
    for (var i = 0; i < plain.length; i++) {
      out.push(String.fromCharCode(plain.charCodeAt(i) ^ key.charCodeAt(i % key.length)));
    }
    return btoa(out.join('')).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }

  function buildLeaderboard(yourTime) {
    var solvers = (PD.ai_times || []).slice();
    var youIdx = 0;
    while (youIdx < solvers.length && solvers[youIdx].time <= yourTime) youIdx++;
    solvers.splice(youIdx, 0, { model: 'YOU', time: yourTime, isYou: true });
    var failedCount = PD.failed_count || 0;
    var shown = new Set();
    for (var k = 0; k < Math.min(3, solvers.length); k++) shown.add(k);
    for (var k = Math.max(0, youIdx - 2); k <= Math.min(solvers.length - 1, youIdx + 2); k++) shown.add(k);
    if (solvers.length > 0) shown.add(solvers.length - 1);
    var indices = Array.from(shown).sort(function (a, b) { return a - b; });
    var html = '<ol class="modal-leaderboard-list">';
    for (var n = 0; n < indices.length; n++) {
      var idx = indices[n];
      var s = solvers[idx];
      var cls = s.isYou ? ' lb-you' : '';
      var label = s.isYou ? 'YOU' : s.model;
      if (n > 0 && indices[n] > indices[n - 1] + 1) html += '<li class="lb-separator">\u22ee</li>';
      html += '<li class="' + cls + '"><span class="lb-rank">' + (idx + 1) + '.</span><span class="lb-name">' + label + '</span><span class="lb-time">' + formatTimerStyle(s.time) + '</span></li>';
    }
    if (failedCount > 0) {
      html += '<li class="lb-separator">\u22ee</li>';
      html += '<li class="lb-failed"><span class="lb-rank"></span><span class="lb-name">' + failedCount + ' AI models failed</span><span class="lb-time">\u2717</span></li>';
    }
    html += '</ol>';
    return html;
  }

  function showCompletionModal() {
    document.getElementById('modal-time').textContent = formatTimerStyle(timerSeconds);
    let statsText = '';
    if (PD.total_count > 0) {
      statsText = PD.solved_count + ' of ' + PD.total_count + ' AI models solved this puzzle';
    }
    document.getElementById('modal-stats').textContent = statsText;
    document.getElementById('modal-leaderboard').innerHTML = buildLeaderboard(timerSeconds);
    var token = obfuscateShare(PD.idx, timerSeconds);
    var shareUrl = window.location.origin + '/share/' + token;
    document.getElementById('share-url').value = shareUrl;
    document.getElementById('completion-modal').style.display = 'flex';
  }

  function closeModal() {
    document.getElementById('completion-modal').style.display = 'none';
  }

  document.getElementById('btn-check').addEventListener('click', () => {
    if (!puzzle) return;
    var result = puzzle.check(true);
    var msgEl = document.getElementById('check-message');
    if (result.complete) {
      showComplete();
      if (!completed) {
        completed = true;
        clearInterval(timerInterval);
        timerInterval = null;
        showCompletionModal();
      }
    } else {
      var text = result.gettext('en') || 'Errors found';
      msgEl.textContent = text;
      msgEl.className = 'check-message error';
    }
  });

  document.getElementById('btn-undo').addEventListener('click', () => {
    if (puzzle) puzzle.undo();
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    if (puzzle) {
      puzzle.ansclear();
      completed = false;
    }
  });

  document.getElementById('btn-modal-close').addEventListener('click', closeModal);
  document.getElementById('btn-modal-close-2').addEventListener('click', closeModal);
  document.getElementById('completion-modal').addEventListener('click', function (e) {
    if (e.target === this) closeModal();
  });
  document.getElementById('btn-copy-share').addEventListener('click', function () {
    var input = document.getElementById('share-url');
    input.select();
    navigator.clipboard.writeText(input.value).then(function () {
      var btn = document.getElementById('btn-copy-share');
      btn.textContent = 'Copied!';
      setTimeout(function () { btn.textContent = 'Copy'; }, 1500);
    });
  });
  document.getElementById('btn-share-x').addEventListener('click', function () {
    var text = 'I solved this ' + PD.type + ' puzzle in ' + formatTimerStyle(timerSeconds) + ' on Pencil Puzzle Bench! Can you beat me?';
    window.open('https://x.com/intent/tweet?text=' + encodeURIComponent(text) +
      '&url=' + encodeURIComponent(document.getElementById('share-url').value), '_blank');
  });

  initPuzzle();
  startTimer();
  loadDetailedRules(PD.type);
})();
