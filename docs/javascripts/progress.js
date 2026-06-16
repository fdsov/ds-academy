(function () {
  "use strict";
  var PREFIX = "dsa:done:";

  function curKey() { return PREFIX + location.pathname; }
  function isDone(path) { return localStorage.getItem(PREFIX + path) === "1"; }

  function markSidebar() {
    var links = document.querySelectorAll(".md-nav__link");
    links.forEach(function (a) {
      var path;
      try { path = new URL(a.href).pathname; } catch (e) { return; }
      if (isDone(path)) a.classList.add("dsa-done");
      else a.classList.remove("dsa-done");
    });
  }

  function renderCounter() {
    // Общий прогресс по всем урокам, ссылки на которые есть в навигации
    var seen = {};
    var total = 0, done = 0;
    document.querySelectorAll(".md-nav__link").forEach(function (a) {
      var path;
      try { path = new URL(a.href).pathname; } catch (e) { return; }
      if (seen[path]) return;
      // считаем только страницы-уроки (внутри фаз/практикума)
      if (!/\/(phase\d|practicum)\//.test(path)) return;
      seen[path] = true;
      total += 1;
      if (isDone(path)) done += 1;
    });
    var box = document.querySelector(".dsa-counter");
    if (!box) {
      var primary = document.querySelector(".md-nav--primary > .md-nav__title");
      if (!primary) return;
      box = document.createElement("div");
      box.className = "dsa-counter";
      primary.insertAdjacentElement("afterend", box);
    }
    var pct = total ? Math.round((done / total) * 100) : 0;
    box.innerHTML =
      '<div class="dsa-counter__label">Прогресс курса: ' + done + " / " + total + " (" + pct + "%)</div>" +
      '<div class="dsa-counter__bar"><span style="width:' + pct + '%"></span></div>';
  }

  function injectToggle() {
    var article = document.querySelector(".md-content__inner");
    if (!article) return;
    var h1 = article.querySelector("h1");
    if (!h1) return;
    // только на страницах-уроках
    if (!/\/(phase\d|practicum)\//.test(location.pathname)) return;
    if (article.querySelector(".lesson-progress")) return;

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "lesson-progress";

    function refresh() {
      var done = localStorage.getItem(curKey()) === "1";
      btn.classList.toggle("done", done);
      btn.textContent = done ? "Пройдено — снять отметку" : "Отметить как пройденное";
    }
    btn.addEventListener("click", function () {
      var done = localStorage.getItem(curKey()) === "1";
      if (done) localStorage.removeItem(curKey());
      else localStorage.setItem(curKey(), "1");
      refresh();
      markSidebar();
      renderCounter();
    });
    refresh();
    h1.insertAdjacentElement("afterend", btn);
  }

  function run() {
    injectToggle();
    markSidebar();
    renderCounter();
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(run);
  } else {
    document.addEventListener("DOMContentLoaded", run);
  }
})();
