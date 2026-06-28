(function () {
  "use strict";
  // Флешкарты + интервальное повторение (SM-2-lite).
  // Два режима в одном файле:
  //   A) на странице урока — тихо регистрирует карточки из пар
  //      !!! question "Проверь себя"  <->  ??? success "Ответы"
  //      Контент урока не редактируется, только читается.
  //   B) на странице /review/ — строит тренажёр повторения по контейнерам
  //      #dsa-fc-stats / #dsa-fc-start / #dsa-fc-session.
  //
  // Хранилище (namespace dsa:):
  //   dsa:fc:card:<id> = JSON {id,q,a,path,title}      — контент карточки
  //   dsa:fc:<id>      = JSON {ef,interval,due,reps,lapses} — состояние SM-2

  var CARD_PREFIX = "dsa:fc:card:";
  var STATE_PREFIX = "dsa:fc:";
  var LESSON_RE = /\/(phase\d+|practicum)\//;
  var DAY = 86400000;

  // ----------------------------------------------------------- утилиты
  function djb2(str) {
    var h = 5381;
    for (var i = 0; i < str.length; i++) {
      h = (h * 33) ^ str.charCodeAt(i);
    }
    return (h >>> 0).toString(16);
  }
  function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function readJSON(k) { var v = lsGet(k); if (!v) return null; try { return JSON.parse(v); } catch (e) { return null; } }
  function writeJSON(k, o) { lsSet(k, JSON.stringify(o)); }
  function pad2(n) { return n < 10 ? "0" + n : "" + n; }
  function todayISO() { var d = new Date(); return d.getFullYear() + "-" + pad2(d.getMonth() + 1) + "-" + pad2(d.getDate()); }

  function typeset(el) {
    if (window.MathJax && MathJax.typesetPromise && el.querySelector && el.querySelector(".arithmatex")) {
      try { if (MathJax.typesetClear) MathJax.typesetClear([el]); } catch (e) {}
      try { MathJax.typesetPromise([el]); } catch (e) {}
    }
  }

  function plural(n, one, few, many) {
    var n10 = n % 10, n100 = n % 100;
    if (n10 === 1 && n100 !== 11) return one;
    if (n10 >= 2 && n10 <= 4 && (n100 < 10 || n100 >= 20)) return few;
    return many;
  }

  // ======================================================= A) РЕГИСТРАЦИЯ
  function lessonTitle() {
    var h = document.querySelector(".md-content__inner h1") || document.querySelector("article h1") || document.querySelector("h1");
    if (h) return (h.textContent || "").replace(/[¶\s]+$/, "").trim();
    return (document.title || "").split(/[·|—-]/)[0].trim();
  }

  function topItems(container) {
    // элементы <li> верхнего уровня внутри первого <ol> контейнера
    var ol = container.querySelector(":scope > ol") || container.querySelector("ol");
    if (!ol) return [];
    return Array.prototype.filter.call(ol.children, function (n) { return n.tagName === "LI"; });
  }

  function hasText(el, sel, needle) {
    var t = el.querySelector(sel);
    return t && (t.textContent || "").indexOf(needle) >= 0;
  }

  function findAnswerAfter(qadm, answers, used) {
    for (var i = 0; i < answers.length; i++) {
      if (used[i]) continue;
      if (qadm.compareDocumentPosition(answers[i]) & Node.DOCUMENT_POSITION_FOLLOWING) {
        used[i] = true;
        return answers[i];
      }
    }
    return null;
  }

  function markRegistered(qadm) {
    if (qadm.dataset.fcReg) return;
    qadm.dataset.fcReg = "1";
    var title = qadm.querySelector(":scope > .admonition-title");
    if (title && !title.querySelector(".dsa-fc-tag")) {
      var tag = document.createElement("span");
      tag.className = "dsa-fc-tag";
      tag.textContent = "в повторении";
      title.appendChild(tag);
    }
  }

  function registerCards() {
    if (!LESSON_RE.test(location.pathname)) return;
    var article = document.querySelector(".md-content__inner") || document;

    var qAdms = Array.prototype.slice.call(article.querySelectorAll(".admonition.question"))
      .filter(function (a) { return hasText(a, ":scope > .admonition-title", "Проверь себя"); });
    if (!qAdms.length) return;

    var answers = Array.prototype.slice.call(article.querySelectorAll("details.success"))
      .filter(function (d) { return hasText(d, ":scope > summary", "Ответы"); });
    var used = [];

    var path = location.pathname;
    var title = lessonTitle();
    var gi = 0;

    qAdms.forEach(function (qadm) {
      var ans = findAnswerAfter(qadm, answers, used);
      if (!ans) return;
      var qItems = topItems(qadm);
      var aItems = topItems(ans);
      var n = Math.min(qItems.length, aItems.length);
      if (!n) return;

      for (var i = 0; i < n; i++) {
        var q = (qItems[i].innerHTML || "").trim();
        var a = (aItems[i].innerHTML || "").trim();
        if (!q || !a) { gi++; continue; }
        var id = djb2(path + "#" + gi);
        gi++;
        var key = CARD_PREFIX + id;
        var prev = readJSON(key);
        if (!prev || prev.q !== q || prev.a !== a || prev.title !== title) {
          writeJSON(key, { id: id, q: q, a: a, path: path, title: title });
        }
      }
      markRegistered(qadm);
    });
  }

  // ===================================================== SM-2-lite
  function freshState() { return { ef: 2.5, interval: 0, due: Date.now(), reps: 0, lapses: 0 }; }

  function grade(state, g) {
    state = state || freshState();
    if (g < 3) {
      state.reps = 0;
      state.interval = 1;
      state.lapses = (state.lapses || 0) + 1;
      state.ef = Math.max(1.3, state.ef - 0.2);
    } else {
      state.reps = (state.reps || 0) + 1;
      state.ef = Math.max(1.3, state.ef + (0.1 - (5 - g) * (0.08 + (5 - g) * 0.02)));
      if (state.reps === 1) state.interval = 1;
      else if (state.reps === 2) state.interval = 6;
      else state.interval = Math.round((state.interval || 1) * state.ef);
    }
    state.due = Date.now() + state.interval * DAY;
    return state;
  }

  // ===================================================== B) ТРЕНАЖЁР
  function allCards() {
    var out = [];
    var n;
    try { n = localStorage.length; } catch (e) { return out; }
    for (var i = 0; i < n; i++) {
      var k;
      try { k = localStorage.key(i); } catch (e) { continue; }
      if (!k || k.indexOf(CARD_PREFIX) !== 0) continue;
      var c = readJSON(k);
      if (c && c.q && c.a) out.push(c);
    }
    return out;
  }
  function stateOf(id) { return readJSON(STATE_PREFIX + id); }
  function isDue(c, now) {
    var s = stateOf(c.id);
    return !s || (s.due || 0) <= now;
  }
  function isLearned(c) {
    var s = stateOf(c.id);
    return s && (s.interval || 0) >= 21;
  }
  function shuffle(arr) {
    for (var i = arr.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  function statCard(value, label) {
    var el = document.createElement("div");
    el.className = "dsa-fc-stat";
    var b = document.createElement("b");
    b.textContent = String(value);
    var s = document.createElement("span");
    s.textContent = label;
    el.appendChild(b);
    el.appendChild(s);
    return el;
  }

  function renderStats(host) {
    if (!host) return;
    var cards = allCards();
    var now = Date.now();
    var total = cards.length;
    var due = 0, learned = 0;
    cards.forEach(function (c) {
      if (isDue(c, now)) due++;
      if (isLearned(c)) learned++;
    });
    host.innerHTML = "";
    if (!total) return;
    host.appendChild(statCard(total, "всего карточек"));
    host.appendChild(statCard(due, "к повторению сегодня"));
    host.appendChild(statCard(learned, "выучено"));
  }

  function emptyState(host) {
    host.innerHTML = "";
    var box = document.createElement("div");
    box.className = "dsa-fc-empty dsa-card";
    var h = document.createElement("div");
    h.className = "dsa-fc-empty__title";
    h.textContent = "Карточек пока нет";
    var p = document.createElement("p");
    p.textContent = "Пройди уроки с блоком «Проверь себя», и карточки появятся здесь. Каждая пара вопрос/ответ автоматически попадает в повторение.";
    box.appendChild(h);
    box.appendChild(p);
    host.appendChild(box);
  }

  function buildReview() {
    var session = document.getElementById("dsa-fc-session");
    if (!session || session.dataset.fcDone) return;
    session.dataset.fcDone = "1";

    var statsHost = document.getElementById("dsa-fc-stats");
    var startBtn = document.getElementById("dsa-fc-start");

    function refreshStats() { renderStats(statsHost); }
    refreshStats();

    var cards = allCards();
    if (!cards.length) {
      if (startBtn) startBtn.style.display = "none";
      emptyState(session);
      return;
    }

    function startSession(pool, modeLabel) {
      var queue = shuffle(pool.slice());
      var total = queue.length;
      var idx = 0;
      if (startBtn) startBtn.disabled = true;

      function finish() {
        if (window.DSA && DSA.award) DSA.award(5, "fc:session:" + todayISO());
        if (window.DSA && DSA.touchStreak) DSA.touchStreak();
        session.innerHTML = "";
        var done = document.createElement("div");
        done.className = "dsa-fc-card dsa-fc-done";
        var t = document.createElement("div");
        t.className = "dsa-fc-done__title";
        t.textContent = "Сессия завершена";
        var p = document.createElement("p");
        p.textContent = "Повторено карточек: " + total + ". +5 XP за сессию. Возвращайся завтра — интервалы рассчитаны автоматически.";
        var again = document.createElement("button");
        again.type = "button";
        again.className = "dsa-btn dsa-btn--ghost";
        again.textContent = "К списку";
        again.addEventListener("click", function () {
          session.innerHTML = "";
          if (startBtn) { startBtn.disabled = false; startBtn.style.display = ""; }
          refreshStats();
          setupStart();
        });
        done.appendChild(t);
        done.appendChild(p);
        done.appendChild(again);
        session.appendChild(done);
        refreshStats();
      }

      function showCard() {
        if (idx >= total) { finish(); return; }
        var card = queue[idx];
        session.innerHTML = "";

        var bar = document.createElement("div");
        bar.className = "dsa-fc-bar";
        var fill = document.createElement("div");
        fill.className = "dsa-fc-bar__fill";
        fill.style.width = Math.round((idx / total) * 100) + "%";
        bar.appendChild(fill);
        session.appendChild(bar);

        var counter = document.createElement("div");
        counter.className = "dsa-fc-counter";
        var left = total - idx;
        counter.textContent = "осталось " + left + " " + plural(left, "карточка", "карточки", "карточек");
        if (modeLabel) counter.textContent += " · " + modeLabel;
        session.appendChild(counter);

        var box = document.createElement("div");
        box.className = "dsa-fc-card";

        var src = document.createElement("div");
        src.className = "dsa-fc-card__src";
        if (card.path) {
          var a = document.createElement("a");
          a.href = card.path;
          a.textContent = card.title || card.path;
          src.appendChild(document.createTextNode("Источник: "));
          src.appendChild(a);
        } else if (card.title) {
          src.textContent = "Источник: " + card.title;
        }
        box.appendChild(src);

        var q = document.createElement("div");
        q.className = "dsa-fc-card__q";
        q.innerHTML = card.q;
        box.appendChild(q);

        var ans = document.createElement("div");
        ans.className = "dsa-fc-card__a";
        ans.innerHTML = card.a;
        ans.style.display = "none";
        box.appendChild(ans);

        var actions = document.createElement("div");
        actions.className = "dsa-fc-card__actions";
        box.appendChild(actions);

        var reveal = document.createElement("button");
        reveal.type = "button";
        reveal.className = "dsa-btn dsa-btn--primary dsa-fc-reveal";
        reveal.textContent = "Показать ответ";
        actions.appendChild(reveal);

        session.appendChild(box);
        typeset(q);

        function rate(g) {
          var st = grade(stateOf(card.id), g);
          writeJSON(STATE_PREFIX + card.id, st);
          idx++;
          showCard();
        }

        reveal.addEventListener("click", function () {
          ans.style.display = "";
          typeset(ans);
          actions.innerHTML = "";
          var grades = [
            { g: 0, label: "Не помню", cls: "again" },
            { g: 3, label: "Трудно", cls: "hard" },
            { g: 4, label: "Хорошо", cls: "good" },
            { g: 5, label: "Легко", cls: "easy" }
          ];
          grades.forEach(function (gr) {
            var b = document.createElement("button");
            b.type = "button";
            b.className = "dsa-btn dsa-fc-grade dsa-fc-grade--" + gr.cls;
            b.textContent = gr.label;
            b.addEventListener("click", function () { rate(gr.g); });
            actions.appendChild(b);
          });
        });
      }

      session.innerHTML = "";
      showCard();
    }

    function setupStart() {
      if (!startBtn) return;
      var now = Date.now();
      var due = cards.filter(function (c) { return isDue(c, now); });
      // переустанавливаем обработчик чистой кнопкой (защита от дублей)
      var fresh = startBtn.cloneNode(true);
      startBtn.parentNode.replaceChild(fresh, startBtn);
      startBtn = fresh;

      if (due.length) {
        startBtn.disabled = false;
        startBtn.textContent = "Начать повторение (" + due.length + ")";
        startBtn.addEventListener("click", function () { startSession(due, null); });
      } else {
        startBtn.disabled = false;
        startBtn.textContent = "Повторить всё равно";
        var note = document.createElement("div");
        note.className = "dsa-fc-note";
        note.textContent = "На сегодня всё повторено. Можно потренировать карточки заранее.";
        session.appendChild(note);
        startBtn.addEventListener("click", function () {
          if (note.parentNode) note.parentNode.removeChild(note);
          startSession(cards, "тренировка");
        });
      }
    }

    setupStart();
  }

  // ============================================================ запуск
  function run() {
    registerCards();
    buildReview();
  }

  if (typeof document$ !== "undefined" && document$.subscribe) document$.subscribe(run);
  else document.addEventListener("DOMContentLoaded", run);
})();
