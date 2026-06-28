(function () {
  "use strict";
  // Виджет задачи с числовым ответом и авто-проверкой (Stepik-style).
  // Источник — fenced ```text блок с полями:
  //   TASK: текст задачи (может быть многострочным)
  //   ANSWER: эталонное число
  //   TOL: абсолютный допуск (опц.)
  //   RTOL: относительный допуск, доля (опц.)
  //   UNIT: подпись после поля ввода (опц., напр. %)
  //   PLACEHOLDER: подсказка в поле (опц.)
  //   EXPLAIN: разбор, показывается после верного/после "Показать ответ" (опц.)
  // Несколько задач в одном блоке разделяются строкой ---.

  var FIELD_RE = /^([A-Z][A-Z_]*):\s?([\s\S]*)$/;

  function num(v) {
    var s = ("" + v).trim().replace(/\s+/g, "").replace("%", "").replace(",", ".");
    if (s === "") return NaN;
    return parseFloat(s);
  }

  function parse(raw) {
    var blocks = raw.replace(/\r/g, "").split(/^\s*-{3,}\s*$/m);
    var tasks = [];
    blocks.forEach(function (block) {
      var t = { task: "", answer: NaN, tol: null, rtol: null, unit: "", placeholder: "", explain: "" };
      var cur = null;
      block.split("\n").forEach(function (line) {
        var m = line.match(FIELD_RE);
        if (m) {
          var key = m[1].toUpperCase();
          var val = m[2];
          cur = null;
          if (key === "TASK") { t.task = val; cur = "task"; }
          else if (key === "ANSWER") { t.answer = num(val); }
          else if (key === "TOL") { t.tol = num(val); }
          else if (key === "RTOL") { t.rtol = num(val); }
          else if (key === "UNIT") { t.unit = val.trim(); }
          else if (key === "PLACEHOLDER") { t.placeholder = val.trim(); }
          else if (key === "EXPLAIN") { t.explain = val; cur = "explain"; }
          else if (cur === null && line.trim()) { /* неизвестный ключ — игнор */ }
          return;
        }
        var s = line.trim();
        if (cur === "task") { if (s) t.task += (t.task ? " " : "") + s; }
        else if (cur === "explain") { t.explain += (s ? (t.explain ? " " : "") + s : ""); }
      });
      if (t.task && !isNaN(t.answer)) tasks.push(t);
    });
    return tasks;
  }

  function judge(t, x) {
    if (isNaN(x)) return "empty";
    var diff = Math.abs(x - t.answer);
    var tol = t.tol != null ? t.tol : 0;
    var rtol = t.rtol != null && Math.abs(t.answer) > 0 ? t.rtol * Math.abs(t.answer) : 0;
    var band = Math.max(tol, rtol);
    if (diff <= band) return "ok";
    if (band > 0 && diff <= band * 3) return "near";
    return "no";
  }

  function renderTask(t) {
    var wrap = document.createElement("div");
    wrap.className = "task";

    var label = document.createElement("div");
    label.className = "task__label";
    label.textContent = "Задача";
    wrap.appendChild(label);

    var q = document.createElement("div");
    q.className = "task-q__title";
    q.textContent = t.task;
    wrap.appendChild(q);

    var row = document.createElement("div");
    row.className = "task-row";
    var input = document.createElement("input");
    input.type = "text";
    input.inputMode = "decimal";
    input.className = "task-input";
    input.placeholder = t.placeholder || "ответ числом";
    row.appendChild(input);
    if (t.unit) {
      var u = document.createElement("span");
      u.className = "task-unit";
      u.textContent = t.unit;
      row.appendChild(u);
    }
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "task-btn";
    btn.textContent = "Проверить";
    row.appendChild(btn);
    wrap.appendChild(row);

    var verdict = document.createElement("div");
    verdict.className = "task-verdict";
    wrap.appendChild(verdict);

    var exp = document.createElement("div");
    exp.className = "task-exp";
    exp.style.display = "none";
    if (t.explain) exp.textContent = t.explain.trim();
    wrap.appendChild(exp);

    var attempts = 0, solved = false, revealBtn = null;

    function showExp() { if (t.explain) exp.style.display = ""; }

    function reveal() {
      solved = true;
      verdict.className = "task-verdict ok";
      verdict.textContent = "Ответ: " + t.answer + (t.unit ? " " + t.unit : "");
      input.value = t.answer;
      input.disabled = true;
      btn.disabled = true;
      if (revealBtn) revealBtn.remove();
      showExp();
    }

    function check() {
      if (solved) return;
      var res = judge(t, num(input.value));
      if (res === "empty") {
        verdict.className = "task-verdict no";
        verdict.textContent = "Введи число";
        return;
      }
      attempts++;
      input.classList.remove("is-ok", "is-no");
      if (res === "ok") {
        verdict.className = "task-verdict ok";
        verdict.textContent = "Верно";
        input.classList.add("is-ok");
        input.disabled = true;
        btn.disabled = true;
        solved = true;
        if (revealBtn) revealBtn.remove();
        showExp();
      } else if (res === "near") {
        verdict.className = "task-verdict near";
        verdict.textContent = "Близко — проверь округление";
        input.classList.add("is-no");
      } else {
        verdict.className = "task-verdict no";
        verdict.textContent = "Неверно";
        input.classList.add("is-no");
      }
      if (!solved && attempts >= 2 && !revealBtn) {
        revealBtn = document.createElement("button");
        revealBtn.type = "button";
        revealBtn.className = "task-reveal";
        revealBtn.textContent = "Показать ответ";
        revealBtn.addEventListener("click", reveal);
        wrap.appendChild(revealBtn);
      }
    }

    btn.addEventListener("click", check);
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); check(); }
    });
    return wrap;
  }

  function build(raw) {
    var tasks = parse(raw);
    if (!tasks.length) return null;
    if (tasks.length === 1) return renderTask(tasks[0]);
    var box = document.createElement("div");
    box.className = "task-group";
    tasks.forEach(function (t) { box.appendChild(renderTask(t)); });
    return box;
  }

  function isTask(raw) {
    return /^[ \t]*TASK:/m.test(raw) && /^[ \t]*ANSWER:/m.test(raw);
  }

  function initTasks() {
    document.querySelectorAll("div.language-text").forEach(function (container) {
      if (container.dataset.taskDone) return;
      var code = container.querySelector("code");
      if (!code) return;
      var raw = code.textContent || "";
      if (!isTask(raw)) return;
      var widget = build(raw);
      if (!widget) return;
      container.dataset.taskDone = "1";
      container.replaceWith(widget);
    });
  }

  if (typeof document$ !== "undefined" && document$.subscribe) document$.subscribe(initTasks);
  else document.addEventListener("DOMContentLoaded", initTasks);
})();
