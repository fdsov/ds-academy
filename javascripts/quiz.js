(function () {
  "use strict";
  // Квизы из fenced ```text блока. Синтаксис:
  //   Q: текст вопроса (необязательный префикс Q:)
  //   [ ] неверный вариант
  //   [x] верный вариант
  //   > пояснение (опц.)
  //   --- разделитель вопросов
  // Один [x] -> single-choice (мгновенная подсветка по клику).
  // Несколько [x] -> multi-select (чекбоксы + кнопка «Проверить»).
  // Верный ответ начисляет XP через window.DSA (опционально).

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

  function award(dkIdx) {
    if (window.DSA && DSA.award) {
      try { DSA.award(5, "quiz:" + location.pathname + ":" + dkIdx); } catch (e) {}
    }
  }

  function countCorrect(qd) {
    var n = 0;
    qd.opts.forEach(function (o) { if (o.correct) n++; });
    return n;
  }

  function makeShell(qd, dispIdx, total) {
    var wrap = document.createElement("div");
    wrap.className = "quiz-q";
    var head = document.createElement("div");
    head.className = "quiz-q__title";
    head.textContent = (total > 1 ? (dispIdx + 1) + ". " : "") + qd.q;
    wrap.appendChild(head);
    return wrap;
  }
  function appendExp(wrap, qd) {
    if (!qd.exp) return;
    var e = document.createElement("div");
    e.className = "quiz-exp";
    e.textContent = qd.exp;
    wrap.appendChild(e);
  }

  // ----- single-choice: мгновенная подсветка по клику -----
  function renderSingle(qd, dispIdx, total, dkIdx) {
    var wrap = makeShell(qd, dispIdx, total);
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
        if (o.correct) {
          b.classList.add("correct");
          award(dkIdx);
        } else {
          b.classList.add("wrong");
          Array.prototype.forEach.call(opts.children, function (c, i) {
            if (qd.opts[i].correct) c.classList.add("correct");
          });
        }
        var verdict = document.createElement("div");
        verdict.className = "quiz-verdict " + (o.correct ? "ok" : "no");
        verdict.textContent = o.correct ? "Верно" : "Неверно";
        wrap.appendChild(verdict);
        appendExp(wrap, qd);
      });
      opts.appendChild(b);
    });
    wrap.appendChild(opts);
    return wrap;
  }

  // ----- multi-select: чекбоксы + «Проверить» -----
  function renderMulti(qd, dispIdx, total, dkIdx) {
    var wrap = makeShell(qd, dispIdx, total);
    var opts = document.createElement("div");
    opts.className = "quiz-opts quiz-opts--multi";
    var solved = false;
    var buttons = [];

    qd.opts.forEach(function (o) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "quiz-opt quiz-opt--multi";
      b.textContent = o.text;
      b.addEventListener("click", function () {
        if (solved) return;
        b.classList.toggle("is-selected");
      });
      buttons.push(b);
      opts.appendChild(b);
    });
    wrap.appendChild(opts);

    var verdict = document.createElement("div");
    verdict.className = "quiz-verdict";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "quiz-check dsa-btn dsa-btn--primary";
    btn.textContent = "Проверить";

    btn.addEventListener("click", function () {
      if (solved) return;
      var anySelected = false;
      buttons.forEach(function (b) {
        b.classList.remove("correct", "wrong");
        if (b.classList.contains("is-selected")) anySelected = true;
      });
      if (!anySelected) {
        verdict.className = "quiz-verdict no";
        verdict.textContent = "Выбери хотя бы один вариант";
        return;
      }
      var exact = true;
      qd.opts.forEach(function (o, i) {
        var sel = buttons[i].classList.contains("is-selected");
        if (sel && o.correct) buttons[i].classList.add("correct");
        else if (sel && !o.correct) { buttons[i].classList.add("wrong"); exact = false; }
        else if (!sel && o.correct) { exact = false; }
      });
      if (exact) {
        solved = true;
        opts.classList.add("answered");
        verdict.className = "quiz-verdict ok";
        verdict.textContent = "Верно";
        btn.disabled = true;
        appendExp(wrap, qd);
        award(dkIdx);
      } else {
        verdict.className = "quiz-verdict no";
        verdict.textContent = "Пока не точно — поправь выбор";
      }
    });

    wrap.appendChild(btn);
    wrap.appendChild(verdict);
    return wrap;
  }

  function renderQuestion(qd, dispIdx, total, dkIdx) {
    return countCorrect(qd) > 1
      ? renderMulti(qd, dispIdx, total, dkIdx)
      : renderSingle(qd, dispIdx, total, dkIdx);
  }

  function build(raw, base) {
    var questions = parse(raw);
    if (!questions.length) return null;
    var box = document.createElement("div");
    box.className = "quiz";
    var label = document.createElement("div");
    label.className = "quiz__label";
    label.textContent = "Квиз";
    box.appendChild(label);
    questions.forEach(function (qd, i) {
      box.appendChild(renderQuestion(qd, i, questions.length, base + i));
    });
    box._qcount = questions.length;
    return box;
  }

  function isQuiz(raw) {
    return /^[ \t]*\[[ xX]\]\s/m.test(raw) && /^[ \t]*\[[xX]\]\s/m.test(raw);
  }

  function initQuizzes() {
    var base = 0;
    document.querySelectorAll("div.language-text").forEach(function (container) {
      if (container.dataset.quizDone) return;
      var code = container.querySelector("code");
      if (!code) return;
      var raw = code.textContent || "";
      if (!isQuiz(raw)) return;
      var widget = build(raw, base);
      if (!widget) return;
      base += widget._qcount || 0;
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
