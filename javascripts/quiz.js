(function () {
  "use strict";

  function parse(raw) {
    var lines = raw.replace(/\r/g, "").split("\n");
    var questions = [];
    var cur = null;
    function push() { if (cur && cur.opts.length) questions.push(cur); cur = null; }
    lines.forEach(function (line) {
      var t = line.trim();
      if (t === "") return;
      if (/^-{3,}$/.test(t)) { push(); return; }
      var opt = t.match(/^\[([ xX])\]\s*(.*)$/);
      if (opt) {
        if (!cur) cur = { q: "", opts: [], exp: "" };
        cur.opts.push({ correct: opt[1].toLowerCase() === "x", text: opt[2] });
        return;
      }
      if (/^>\s?/.test(t)) {
        if (cur) cur.exp += (cur.exp ? " " : "") + t.replace(/^>\s?/, "");
        return;
      }
      var qt = t.replace(/^Q:\s*/, "");
      if (!cur) { cur = { q: qt, opts: [], exp: "" }; }
      else if (cur.opts.length === 0) { cur.q += (cur.q ? " " : "") + qt; }
      else { push(); cur = { q: qt, opts: [], exp: "" }; }
    });
    push();
    return questions;
  }

  function renderQuestion(qd, idx, total) {
    var wrap = document.createElement("div");
    wrap.className = "quiz-q";

    var head = document.createElement("div");
    head.className = "quiz-q__title";
    head.textContent = (total > 1 ? (idx + 1) + ". " : "") + qd.q;
    wrap.appendChild(head);

    var opts = document.createElement("div");
    opts.className = "quiz-opts";
    var answered = false;

    qd.opts.forEach(function (o) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "quiz-opt";
      b.textContent = o.text;
      b.addEventListener("click", function () {
        if (answered) return;
        answered = true;
        opts.classList.add("answered");
        if (o.correct) b.classList.add("correct");
        else {
          b.classList.add("wrong");
          Array.prototype.forEach.call(opts.children, function (c, i) {
            if (qd.opts[i].correct) c.classList.add("correct");
          });
        }
        var verdict = document.createElement("div");
        verdict.className = "quiz-verdict " + (o.correct ? "ok" : "no");
        verdict.textContent = o.correct ? "Верно" : "Неверно";
        wrap.appendChild(verdict);
        if (qd.exp) {
          var e = document.createElement("div");
          e.className = "quiz-exp";
          e.textContent = qd.exp;
          wrap.appendChild(e);
        }
      });
      opts.appendChild(b);
    });
    wrap.appendChild(opts);
    return wrap;
  }

  function build(raw) {
    var questions = parse(raw);
    if (!questions.length) return null;
    var box = document.createElement("div");
    box.className = "quiz";
    var label = document.createElement("div");
    label.className = "quiz__label";
    label.textContent = "Квиз";
    box.appendChild(label);
    questions.forEach(function (qd, i) {
      box.appendChild(renderQuestion(qd, i, questions.length));
    });
    return box;
  }

  function isQuiz(raw) {
    return /^[ \t]*\[[ xX]\]\s/m.test(raw) && /^[ \t]*\[[xX]\]\s/m.test(raw);
  }

  function initQuizzes() {
    document.querySelectorAll("div.language-text").forEach(function (container) {
      if (container.dataset.quizDone) return;
      var code = container.querySelector("code");
      if (!code) return;
      var raw = code.textContent || "";
      if (!isQuiz(raw)) return;
      var widget = build(raw);
      if (!widget) return;
      container.dataset.quizDone = "1";
      container.replaceWith(widget);
    });
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(initQuizzes);
  } else {
    document.addEventListener("DOMContentLoaded", initQuizzes);
  }
})();
