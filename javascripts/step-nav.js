(function () {
  "use strict";
  // Источник статуса прохождения общий с progress.js
  var DONE_PREFIX = "dsa:done:";
  // Лента и кнопки только на страницах-шагах
  var SHOW_RE = /\/(phase\d+|practicum|workshops)\//;

  function pathOf(href) { try { return new URL(href).pathname; } catch (e) { return null; } }
  function isDone(p) { return localStorage.getItem(DONE_PREFIX + p) === "1"; }

  // Определяем модуль текущей страницы по URL (надёжнее, чем по дереву DOM)
  function moduleKey(path) {
    var m = path.match(/\/((?:phase\d+|practicum)\/m\d+)(?:-workshop)?\//);
    if (m) return { type: "module", base: m[1] };
    if (/\/workshops\//.test(path)) return { type: "workshops", base: "workshops" };
    return null;
  }
  // Какие ссылки навигации принадлежат этому модулю
  function memberRe(key) {
    if (key.type === "workshops") return /\/workshops\/[^/]+\/?$/;
    var b = key.base.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    // /base/ (обзор), /base/NN/ (шаг), /base-workshop/ (воркшоп)
    return new RegExp("/" + b + "(?:/(?:\\d+)?|-workshop)/?$");
  }

  function collectMembers(key) {
    var re = memberRe(key);
    var seen = {};
    var members = [];
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

  function buildRail() {
    if (!SHOW_RE.test(location.pathname)) return;
    var article = document.querySelector(".md-content__inner");
    if (!article || article.querySelector(".dsa-steprail")) return;
    var key = moduleKey(location.pathname);
    if (!key) return;
    var members = collectMembers(key);
    if (members.length < 2) return;

    var indexMember = null;
    var pills = [];
    members.forEach(function (m) {
      var c = classify(m, key);
      if (c.kind === "index") { indexMember = m; return; }
      pills.push({ m: m, c: c });
    });
    if (!pills.length) return;

    var rail = document.createElement("nav");
    rail.className = "dsa-steprail";

    if (indexMember && indexMember.text) {
      var h = document.createElement("a");
      h.className = "dsa-steprail__title";
      h.href = indexMember.url;
      h.textContent = indexMember.text;
      rail.appendChild(h);
    }

    var track = document.createElement("div");
    track.className = "dsa-steprail__track";
    var curPill = null;
    pills.forEach(function (pp) {
      var a = document.createElement("a");
      a.className = "dsa-pill" + (pp.c.kind === "workshop" ? " dsa-pill--w" : "");
      a.href = pp.m.url;
      a.title = pp.m.text;
      a.textContent = pp.c.label;
      if (pp.m.path === location.pathname) { a.classList.add("is-current"); curPill = a; }
      else if (isDone(pp.m.path)) { a.classList.add("is-done"); }
      track.appendChild(a);
    });
    rail.appendChild(track);

    var h1 = article.querySelector("h1");
    if (h1) article.insertBefore(rail, h1);
    else article.insertBefore(rail, article.firstChild);

    if (curPill) {
      var off = curPill.offsetLeft - track.clientWidth / 2 + curPill.clientWidth / 2;
      track.scrollLeft = Math.max(0, off);
    }
    document.body.classList.add("dsa-step-page");
  }

  function buildNavButtons() {
    if (!SHOW_RE.test(location.pathname)) return;
    var article = document.querySelector(".md-content__inner");
    if (!article || article.querySelector(".dsa-nav")) return;

    var nextA = document.querySelector(".md-footer__link--next");
    var prevA = document.querySelector(".md-footer__link--prev");
    var nextHref = nextA ? nextA.href : null;
    var prevHref = prevA ? prevA.href : null;
    if (!nextHref && !prevHref) return;

    var box = document.createElement("div");
    box.className = "dsa-nav";

    if (prevHref) {
      var b = document.createElement("a");
      b.className = "dsa-nav__btn dsa-nav__btn--prev";
      b.href = prevHref;
      b.textContent = "← Назад";
      box.appendChild(b);
    }
    if (nextHref) {
      var n = document.createElement("a");
      n.className = "dsa-nav__btn dsa-nav__btn--next";
      n.href = nextHref;
      n.textContent = "Далее →";
      // «Далее» как в Stepik — отмечает текущий шаг пройденным и ведёт дальше
      n.addEventListener("click", function () {
        localStorage.setItem(DONE_PREFIX + location.pathname, "1");
      });
      box.appendChild(n);
    }
    article.appendChild(box);
  }

  function run() { buildRail(); buildNavButtons(); }

  if (typeof document$ !== "undefined" && document$.subscribe) document$.subscribe(run);
  else document.addEventListener("DOMContentLoaded", run);
})();
