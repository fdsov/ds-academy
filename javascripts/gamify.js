/* DS Academy — геймификация и глобальный API window.DSA.
   Грузится ПЕРВЫМ из модулей: определяет window.DSA синхронно в теле IIFE,
   чтобы остальные модули (quiz/task/pyodide/flashcards/lesson-deck) могли
   звать DSA.* сразу. DOM-зависимая инициализация (HUD, кольца, дашборд) —
   через document$ / DOMContentLoaded. Все обработчики идемпотентны. */
(function () {
  "use strict";

  /* ----------------------------------------------------- localStorage */
  var LS = window.localStorage;
  function get(k) { try { return LS.getItem(k); } catch (e) { return null; } }
  function set(k, v) { try { LS.setItem(k, v); } catch (e) {} }
  function del(k) { try { LS.removeItem(k); } catch (e) {} }

  var K = {
    done: function (p) { return "dsa:done:" + p; },
    xp: "dsa:xp",
    awarded: function (key) { return "dsa:xp:awarded:" + key; },
    streakCount: "dsa:streak:count",
    streakLast: "dsa:streak:last",
    badges: "dsa:badges"
  };

  function getXP() { return parseInt(get(K.xp) || "0", 10) || 0; }
  function getStreak() { return parseInt(get(K.streakCount) || "0", 10) || 0; }
  function getBadges() { try { return JSON.parse(get(K.badges) || "[]"); } catch (e) { return []; } }
  function setBadges(arr) { set(K.badges, JSON.stringify(arr)); }

  /* ------------------------------------------------------------- утилиты */
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function pathOf(a) {
    try { return new URL(a.href, location.href).pathname; } catch (e) { return null; }
  }
  // new Date() допустим в браузерном рантайме (ограничение только в workflow-скриптах)
  function isoDate(d) {
    var m = d.getMonth() + 1, day = d.getDate();
    return d.getFullYear() + "-" + (m < 10 ? "0" + m : m) + "-" + (day < 10 ? "0" + day : day);
  }
  function todayISO() { return isoDate(new Date()); }
  function yesterdayISO() {
    var d = new Date();
    d.setDate(d.getDate() - 1);
    return isoDate(d);
  }

  /* --------------------------------------------------------- иконки SVG */
  var ICON_BOLT = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M13 2 4.5 13.5H11l-1 8.5 8.5-12H12l1-8z"/></svg>';
  var ICON_FIRE = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2c1 3-1 4-2 6-1 2 0 4 2 4 1.4 0 2-1 2-2 1 1 2 2.5 2 4a6 6 0 0 1-12 0c0-3 2-5 3-7 .5 2 2 2 2 0 0-2-1-3 1-5z"/></svg>';
  var ICON_BADGE = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2a5 5 0 1 0 0 10 5 5 0 0 0 0-10zm0 2.4.95 2.1 2.3.18-1.74 1.5.55 2.24L12 9.2 9.94 10.4l.55-2.24L8.75 6.68l2.3-.18L12 4.4zM7.7 13.1 6.2 22 12 18.9 17.8 22l-1.5-8.9a7 7 0 0 1-8.6 0z"/></svg>';

  /* --------------------------------------------------------- шина событий */
  var listeners = {};
  function on(ev, cb) { (listeners[ev] = listeners[ev] || []).push(cb); }
  function emit(ev, data) {
    (listeners[ev] || []).forEach(function (cb) { try { cb(data); } catch (e) {} });
  }

  /* ------------------------------------------------------- сбор путей nav */
  var LESSON_RE = /\/(phase\d+|practicum|workshops)\//;
  function lessonPaths() {
    var seen = {}, out = [];
    document.querySelectorAll(".md-nav__link[href]").forEach(function (a) {
      var p = pathOf(a);
      if (!p || !LESSON_RE.test(p) || seen[p]) return;
      seen[p] = true; out.push(p);
    });
    return out;
  }
  function groupOf(p) {
    var m = p.match(LESSON_RE);
    return m ? m[1] : null;
  }

  /* ----------------------------------------------------------- прогресс */
  function isDone(path) { return get(K.done(path || location.pathname)) === "1"; }

  function progress() {
    var paths = lessonPaths(), done = 0;
    paths.forEach(function (p) { if (isDone(p)) done += 1; });
    var total = paths.length;
    return { done: done, total: total, pct: total ? Math.round((done / total) * 100) : 0 };
  }

  function phaseProgress() {
    var groups = {}, order = [];
    document.querySelectorAll(".md-nav__link[href]").forEach(function (a) {
      var p = pathOf(a);
      if (!p) return;
      var g = groupOf(p);
      if (!g) return;
      if (!groups[g]) { groups[g] = {}; order.push(g); }
      groups[g][p] = true;
    });
    return order.map(function (g) {
      var total = 0, done = 0;
      for (var p in groups[g]) { total += 1; if (isDone(p)) done += 1; }
      return { phase: g, done: done, total: total, pct: total ? Math.round((done / total) * 100) : 0 };
    });
  }

  /* ----------------------------------------------------------- стрик/XP */
  function touchStreak() {
    var today = todayISO();
    var last = get(K.streakLast);
    if (last === today) return;
    var count = getStreak();
    if (last && last === yesterdayISO()) count = count + 1;
    else count = 1;
    set(K.streakCount, String(count));
    set(K.streakLast, today);
    renderHUD();
  }

  function award(amount, dedupeKey) {
    amount = parseInt(amount, 10) || 0;
    if (dedupeKey) {
      if (get(K.awarded(dedupeKey)) === "1") return false;
      set(K.awarded(dedupeKey), "1");
    }
    var total = getXP() + amount;
    set(K.xp, String(total));
    emit("xp", { total: total, amount: amount });
    touchStreak();
    renderHUD();
    checkBadges();
    return true;
  }

  function markDone(path) {
    path = path || location.pathname;
    var was = get(K.done(path)) === "1";
    if (!was) {
      set(K.done(path), "1");
      award(10, "done:" + path);
    }
    emit("done", { path: path });
    touchStreak();
    renderHUD();
    markSidebar();
    checkBadges();
    return true;
  }

  /* ------------------------------------------------------------- бейджи */
  var BADGES = [
    { id: "first-step", title: "Первый шаг", desc: "Пройден первый шаг курса", test: function (s) { return s.done >= 1; } },
    { id: "ten-steps", title: "Десятка", desc: "Пройдено 10 шагов", test: function (s) { return s.done >= 10; } },
    { id: "fifty-steps", title: "Полста", desc: "Пройдено 50 шагов", test: function (s) { return s.done >= 50; } },
    { id: "streak-3", title: "Серия 3 дня", desc: "Занимался 3 дня подряд", test: function (s) { return s.streak >= 3; } },
    { id: "streak-7", title: "Серия 7 дней", desc: "Занимался 7 дней подряд", test: function (s) { return s.streak >= 7; } },
    { id: "first-phase", title: "Фаза закрыта", desc: "Первая фаза пройдена на 100%", test: function (s) { return s.phaseDone; } }
  ];

  function computeState() {
    var pr = progress();
    var phaseDone = phaseProgress().some(function (x) { return x.total > 0 && x.pct === 100; });
    return { done: pr.done, total: pr.total, streak: getStreak(), phaseDone: phaseDone };
  }

  function checkBadges() {
    var s = computeState();
    var have = getBadges();
    var changed = false;
    BADGES.forEach(function (b) {
      if (have.indexOf(b.id) === -1 && b.test(s)) {
        have.push(b.id);
        changed = true;
        emit("badge", { id: b.id, title: b.title });
        toast(b.title);
      }
    });
    if (changed) { setBadges(have); renderBadges(); }
  }

  /* -------------------------------------------------------------- toast */
  function toast(title) {
    var wrap = document.querySelector(".dsa-toasts");
    if (!wrap) {
      wrap = document.createElement("div");
      wrap.className = "dsa-toasts";
      document.body.appendChild(wrap);
    }
    var t = document.createElement("div");
    t.className = "dsa-toast";
    t.innerHTML =
      '<span class="dsa-toast__icon">' + ICON_BADGE + "</span>" +
      '<span class="dsa-toast__body"><span class="dsa-toast__kicker">Новый бейдж</span>' +
      '<span class="dsa-toast__title">' + esc(title) + "</span></span>";
    wrap.appendChild(t);
    requestAnimationFrame(function () { t.classList.add("is-in"); });
    setTimeout(function () {
      t.classList.remove("is-in");
      setTimeout(function () { if (t.parentNode) t.parentNode.removeChild(t); }, 350);
    }, 4200);
  }

  /* ---------------------------------------------------------------- HUD */
  function hudContainer() {
    var c = document.querySelector("#dsa-hud");
    if (c) return c;
    // защитный монтаж, если overrides/main.html не подставил контейнер
    var inner = document.querySelector(".md-header__inner");
    if (!inner) return null;
    c = document.createElement("span");
    c.id = "dsa-hud";
    c.className = "dsa-hud";
    var opt = inner.querySelector(".md-header__option");
    if (opt) inner.insertBefore(c, opt);
    else inner.appendChild(c);
    return c;
  }
  function renderHUD() {
    var c = hudContainer();
    if (!c) return;
    if (!c.classList.contains("dsa-hud")) c.classList.add("dsa-hud");
    c.innerHTML =
      '<span class="dsa-hud__item" title="Очки опыта">' + ICON_BOLT + "<span>" + getXP() + "</span></span>" +
      '<span class="dsa-hud__item" title="Серия: дней подряд">' + ICON_FIRE + "<span>" + getStreak() + "</span></span>";
  }

  /* ----------------------------------------------------------- sidebar */
  // Только добавляем класс (снятие отметки — забота progress.js). Идемпотентно.
  function markSidebar() {
    document.querySelectorAll(".md-nav__link[href]").forEach(function (a) {
      var p = pathOf(a);
      if (p && isDone(p) && !a.classList.contains("dsa-done")) a.classList.add("dsa-done");
    });
  }

  /* -------------------------------------------------- кольца и статы */
  function fillRings() {
    var prAll = null, phasesCache = null;
    document.querySelectorAll("[data-dsa-ring]").forEach(function (el) {
      var key = el.getAttribute("data-dsa-ring");
      var pct = 0;
      if (key === "all") {
        prAll = prAll || progress();
        pct = prAll.pct;
      } else {
        phasesCache = phasesCache || phaseProgress();
        var found = null;
        phasesCache.forEach(function (x) { if (x.phase === key) found = x; });
        pct = found ? found.pct : 0;
      }
      if (!el.classList.contains("dsa-ring")) el.classList.add("dsa-ring");
      el.style.setProperty("--p", pct);
      el.innerHTML = '<span class="dsa-ring__val">' + pct + "%</span>";
    });
  }

  function fillStats() {
    var nodes = document.querySelectorAll("[data-dsa-stat]");
    if (!nodes.length) return;
    var pr = progress();
    nodes.forEach(function (el) {
      var k = el.getAttribute("data-dsa-stat");
      var v = "";
      if (k === "xp") v = getXP();
      else if (k === "streak") v = getStreak();
      else if (k === "done") v = pr.done;
      else if (k === "total") v = pr.total;
      else if (k === "pct") v = pr.pct + "%";
      el.textContent = v;
    });
  }

  /* ----------------------------------------------- дашборд: бейджи/reset */
  function renderBadges() {
    var box = document.querySelector("#dsa-badges");
    if (!box) return;
    var have = getBadges();
    box.innerHTML = "";
    BADGES.forEach(function (b) {
      var got = have.indexOf(b.id) !== -1;
      var el = document.createElement("div");
      el.className = "dsa-badge" + (got ? "" : " dsa-badge--locked");
      el.innerHTML =
        '<span class="dsa-badge__icon">' + ICON_BADGE + "</span>" +
        '<span class="dsa-badge__title">' + esc(b.title) + "</span>" +
        '<span class="dsa-badge__desc">' + esc(b.desc) + "</span>" +
        '<span class="dsa-badge__state">' + (got ? "Получен" : "Закрыт") + "</span>";
      box.appendChild(el);
    });
  }

  function wireReset() {
    var btn = document.querySelector("#dsa-reset");
    if (!btn || btn.dataset.dsaWired) return;
    btn.dataset.dsaWired = "1";
    btn.addEventListener("click", function () {
      if (!window.confirm(
        "Сбросить весь прогресс? Будут удалены XP, серия, бейджи, отметки пройденного и состояние флешкарт. Действие необратимо."
      )) return;
      var keys = [];
      for (var i = 0; i < LS.length; i++) {
        var k = LS.key(i);
        if (k && k.indexOf("dsa:") === 0) keys.push(k);
      }
      keys.forEach(function (k) { del(k); });
      location.reload();
    });
  }

  /* ------------------------------------------------------------- запуск */
  function run() {
    renderHUD();
    markSidebar();
    fillRings();
    fillStats();
    renderBadges();
    wireReset();
  }

  // держим HUD и динамические элементы свежими на событиях
  on("xp", function () { renderHUD(); fillRings(); fillStats(); });
  on("done", function () { renderHUD(); markSidebar(); fillRings(); fillStats(); });
  on("badge", function () { renderBadges(); });

  /* ------------------------------- определяем window.DSA СИНХРОННО */
  window.DSA = {
    award: award,
    markDone: markDone,
    isDone: isDone,
    touchStreak: touchStreak,
    progress: progress,
    phaseProgress: phaseProgress,
    on: on,
    emit: emit
  };

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(run);
  } else {
    document.addEventListener("DOMContentLoaded", run);
  }
})();
