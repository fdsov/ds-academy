(function () {
  "use strict";
  // Степпер-деск: показывает урок по одной карточке-экрану за раз.
  // Заменяет step-nav.js. Должен грузиться ПОСЛЕДНИМ — после того как
  // quiz.js/task.js/viz.js/pyodide-run.js уже заменили свои fenced-блоки на виджеты.

  var DONE_PREFIX = "dsa:done:";
  var DECK_PREFIX = "dsa:deck:";
  var SHOW_RE = /\/(phase\d+|practicum|workshops)\//;

  function pathOf(href) { try { return new URL(href, location.href).pathname; } catch (e) { return null; } }
  function isWorkshop(p) { return /-workshop\/?$/.test(p) || /\/workshops\//.test(p); }

  // ---- done state (через window.DSA если есть, иначе localStorage) ----
  function setDone(p) {
    if (window.DSA && DSA.markDone) { DSA.markDone(p); return; }
    try { localStorage.setItem(DONE_PREFIX + p, "1"); } catch (e) {}
  }
  function isDone(p) {
    if (window.DSA && DSA.isDone) return DSA.isDone(p);
    try { return localStorage.getItem(DONE_PREFIX + p) === "1"; } catch (e) { return false; }
  }

  // ====================================================== ЛЕНТА ПОД-ШАГОВ
  function moduleKey(path) {
    var m = path.match(/\/((?:phase\d+|practicum)\/m\d+)(?:-workshop)?\//);
    if (m) return { type: "module", base: m[1] };
    if (/\/workshops\//.test(path)) return { type: "workshops", base: "workshops" };
    return null;
  }
  function memberRe(key) {
    if (key.type === "workshops") return /\/workshops\/[^/]+\/?$/;
    var b = key.base.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return new RegExp("/" + b + "(?:/(?:\\d+)?|-workshop)/?$");
  }
  function collectMembers(key) {
    var re = memberRe(key), seen = {}, members = [];
    document.querySelectorAll(".md-nav__link[href]").forEach(function (a) {
      var p = pathOf(a.href);
      if (!p || seen[p] || !re.test(p)) return;
      seen[p] = true;
      members.push({ path: p, url: a.href, text: (a.textContent || "").trim() });
    });
    return members;
  }
  function classify(m, key) {
    if (key.type === "workshops") {
      var mw = m.text.match(/W\s*(\d+)/i) || m.text.match(/(\d+)/);
      return { kind: "step", label: mw ? mw[1] : "•" };
    }
    if (/-workshop\/?$/.test(m.path)) return { kind: "workshop", label: "W" };
    var ms = m.text.match(/M\d+\.(\d+)/);
    if (ms) return { kind: "step", label: ms[1] };
    return { kind: "index", label: null };
  }
  function buildRail(host) {
    var key = moduleKey(location.pathname);
    if (!key) return;
    var members = collectMembers(key);
    if (members.length < 2) return;
    var indexMember = null, pills = [];
    members.forEach(function (m) {
      var c = classify(m, key);
      if (c.kind === "index") { indexMember = m; return; }
      pills.push({ m: m, c: c });
    });
    if (!pills.length) return;

    var rail = document.createElement("nav");
    rail.className = "dsa-rail";
    if (indexMember && indexMember.text) {
      var h = document.createElement("a");
      h.className = "dsa-rail__title";
      h.href = indexMember.url;
      h.textContent = indexMember.text;
      rail.appendChild(h);
    }
    var track = document.createElement("div");
    track.className = "dsa-rail__track";
    var curPill = null;
    pills.forEach(function (pp) {
      var a = document.createElement("a");
      a.className = "dsa-pillx" + (pp.c.kind === "workshop" ? " dsa-pillx--w" : "");
      a.href = pp.m.url;
      a.title = pp.m.text;
      a.textContent = pp.c.label;
      if (pp.m.path === location.pathname) { a.classList.add("is-current"); curPill = a; }
      else if (isDone(pp.m.path)) { a.classList.add("is-done"); }
      track.appendChild(a);
    });
    rail.appendChild(track);
    host.appendChild(rail);
    if (curPill) {
      requestAnimationFrame(function () {
        track.scrollLeft = Math.max(0, curPill.offsetLeft - track.clientWidth / 2 + curPill.clientWidth / 2);
      });
    }
  }

  // ====================================================== НАРЕЗКА НА КАРТОЧКИ
  function countChildren(article, tag) {
    var n = 0;
    Array.prototype.forEach.call(article.children, function (el) { if (el.tagName === tag) n++; });
    return n;
  }
  // Адаптивный уровень нарезки: воркшопы — по H2. Уроки — по H3, если их >=2
  // (как раньше, без регрессии); иначе по H2 (часть уроков секционируется на ##).
  function splitTag(article) {
    if (isWorkshop(location.pathname)) return "H2";
    if (countChildren(article, "H3") >= 2) return "H3";
    if (countChildren(article, "H2") >= 2) return "H2";
    return "H3";
  }

  function partition(article) {
    var tag = splitTag(article);
    var kids = Array.prototype.slice.call(article.children);
    var head = [], cards = [], cur = null;
    var started = false;
    kids.forEach(function (el) {
      if (el.tagName === "H1") { head.push(el); return; }
      if (el.tagName === tag) { started = true; cur = [el]; cards.push(cur); return; }
      if (!started) { head.push(el); return; }
      cur.push(el);
    });
    return { head: head, cards: cards };
  }

  function typesetCard(card) {
    if (window.MathJax && MathJax.typesetPromise && card.querySelector(".arithmatex")) {
      try { if (MathJax.typesetClear) MathJax.typesetClear([card]); } catch (e) {}
      try { MathJax.typesetPromise([card]); } catch (e) {}
    }
  }

  function build() {
    if (!SHOW_RE.test(location.pathname)) return;
    var article = document.querySelector(".md-content__inner");
    if (!article || article.dataset.deckDone) return;

    var part = partition(article);
    // Шапка: лента под-шагов всегда, если модуль найден
    var head = document.createElement("div");
    head.className = "dsa-deckhead";
    part.head.forEach(function (el) { head.appendChild(el); });
    buildRail(head);

    // Меньше 2 карточек — деск не нужен, оставляем обычный скролл (но с шапкой/лентой)
    if (part.cards.length < 2) {
      article.insertBefore(head, article.firstChild);
      article.dataset.deckDone = "1";
      // короткая страница тоже считается шагом по достижении
      setDone(location.pathname);
      return;
    }

    article.dataset.deckDone = "1";
    document.body.classList.add("dsa-deck-page");

    // Точки прогресса
    var dots = document.createElement("div");
    dots.className = "dsa-dots";
    var dlabel = document.createElement("span");
    dlabel.className = "dsa-dots__label";
    dots.appendChild(dlabel);
    var dotEls = [];

    // Контейнер карточек
    var deck = document.createElement("div");
    deck.className = "dsa-deck";
    var cardEls = part.cards.map(function (nodes) {
      var c = document.createElement("section");
      c.className = "dsa-deck__card";
      nodes.forEach(function (n) { c.appendChild(n); });
      deck.appendChild(c);
      return c;
    });
    var total = cardEls.length;

    part.cards.forEach(function (_, i) {
      var d = document.createElement("button");
      d.type = "button";
      d.className = "dsa-dot";
      d.title = "Карточка " + (i + 1);
      d.addEventListener("click", function () { go(i); });
      dots.appendChild(d);
      dotEls.push(d);
    });

    // Футер навигации
    var foot = document.createElement("div");
    foot.className = "dsa-deckfoot";
    var prevBtn = document.createElement("button");
    prevBtn.type = "button";
    prevBtn.className = "dsa-btn dsa-btn--ghost dsa-btn--prev";
    prevBtn.innerHTML = "&larr; Назад";
    var hint = document.createElement("span");
    hint.className = "dsa-deckfoot__hint";
    var nextBtn = document.createElement("button");
    nextBtn.type = "button";
    nextBtn.className = "dsa-btn dsa-btn--primary dsa-btn--next";
    foot.appendChild(prevBtn);
    foot.appendChild(hint);
    foot.appendChild(nextBtn);

    // Ссылки на соседние под-шаги (footer Material)
    var nextLink = document.querySelector(".md-footer__link--next");
    var prevLink = document.querySelector(".md-footer__link--prev");
    var nextHref = nextLink ? nextLink.href : null;
    var prevHref = prevLink ? prevLink.href : null;

    // Собираем в статью
    article.insertBefore(head, article.firstChild);
    head.appendChild(dots);
    article.appendChild(deck);
    article.appendChild(foot);

    var cur = -1;
    var maxReached = 0;

    function persist() { try { localStorage.setItem(DECK_PREFIX + location.pathname, String(cur)); } catch (e) {} }

    function go(i) {
      i = Math.max(0, Math.min(total - 1, i));
      if (i === cur) return;
      cur = i;
      maxReached = Math.max(maxReached, i);
      cardEls.forEach(function (c, k) { c.classList.toggle("is-active", k === i); });
      dotEls.forEach(function (d, k) {
        d.classList.toggle("is-current", k === i);
        d.classList.toggle("is-done", k < maxReached || (k === total - 1 && i === total - 1));
      });
      dlabel.textContent = "Шаг " + (i + 1) + " / " + total;
      // навигация
      var atLast = i === total - 1;
      var atFirst = i === 0;
      nextBtn.innerHTML = atLast ? "Следующий шаг &rarr;" : "Далее &rarr;";
      if (atLast && !nextHref) { nextBtn.style.display = "none"; }
      else { nextBtn.style.display = ""; }
      if (atFirst && !prevHref) { prevBtn.style.visibility = "hidden"; }
      else { prevBtn.style.visibility = ""; prevBtn.innerHTML = atFirst ? "&larr; Предыдущий шаг" : "&larr; Назад"; }
      typesetCard(cardEls[i]);
      // достигли конца — отмечаем шаг пройденным
      if (atLast) setDone(location.pathname);
      persist();
      // скролл к началу контента
      var top = article.getBoundingClientRect().top + window.pageYOffset - 90;
      if (window.pageYOffset > top + 4) window.scrollTo({ top: top, behavior: "smooth" });
    }

    prevBtn.addEventListener("click", function () {
      if (cur > 0) go(cur - 1);
      else if (prevHref) location.href = prevHref;
    });
    nextBtn.addEventListener("click", function () {
      if (cur < total - 1) go(cur + 1);
      else if (nextHref) { setDone(location.pathname); location.href = nextHref; }
    });
    document.addEventListener("keydown", function (e) {
      if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight" && cur < total - 1) { e.preventDefault(); go(cur + 1); }
      else if (e.key === "ArrowLeft" && cur > 0) { e.preventDefault(); go(cur - 1); }
    });

    // resume
    var saved = parseInt(localStorage.getItem(DECK_PREFIX + location.pathname) || "0", 10);
    if (isNaN(saved) || saved < 0 || saved >= total) saved = 0;
    maxReached = saved;
    go(saved);
    if (saved === 0) typesetCard(cardEls[0]);
  }

  function rememberLast() {
    if (!SHOW_RE.test(location.pathname)) return;
    var h1 = document.querySelector(".md-content__inner h1");
    var title = h1 ? h1.textContent.replace(/[¶\s]+$/, "").trim() : document.title;
    try {
      localStorage.setItem("dsa:last", location.pathname);
      localStorage.setItem("dsa:last:title", title);
    } catch (e) {}
  }

  function wireContinue() {
    var btn = document.getElementById("dsa-continue");
    if (!btn) return;
    var last = null, title = null;
    try { last = localStorage.getItem("dsa:last"); title = localStorage.getItem("dsa:last:title"); } catch (e) {}
    if (!last) return;
    btn.href = last;
    btn.textContent = "Продолжить: " + (title || "последний шаг");
    btn.style.display = "";
  }

  function run() { build(); rememberLast(); wireContinue(); }

  if (typeof document$ !== "undefined" && document$.subscribe) document$.subscribe(run);
  else document.addEventListener("DOMContentLoaded", run);
})();
