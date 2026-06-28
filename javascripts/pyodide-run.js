(function () {
  "use strict";
  // Python в браузере через Pyodide.
  //   A) Кнопка «Запустить» под каждым отрендеренным блоком ```python.
  //   B) Авто-проверяемые задания ```pyexercise (solution + скрытые tests).
  // Pyodide грузится лениво один раз на страницу при первом клике Run/Проверить.
  // Должен грузиться ДО lesson-deck.js, чтобы тот нарезал на карточки уже
  // с готовыми панелями (панель — это sibling блока кода, едет вместе с ним).

  var PYO_VER = "0.26.2";
  var PYO_BASE = "https://cdn.jsdelivr.net/pyodide/v" + PYO_VER + "/full/";

  // ---- ленивая инициализация Pyodide (singleton-промис на всю страницу) ----
  var pyodidePromise = null;

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement("script");
      s.src = src;
      s.onload = function () { resolve(); };
      s.onerror = function () { reject(new Error("Не удалось загрузить " + src)); };
      document.head.appendChild(s);
    });
  }

  function ensurePyodide() {
    if (pyodidePromise) return pyodidePromise;
    pyodidePromise = (async function () {
      if (typeof loadPyodide === "undefined") {
        await loadScript(PYO_BASE + "pyodide.js");
      }
      var py = await loadPyodide({ indexURL: PYO_BASE });
      // Headless-бэкенд matplotlib, чтобы savefig работал без дисплея.
      try { py.runPython("import os; os.environ['MPLBACKEND']='AGG'"); } catch (e) {}
      return py;
    })();
    return pyodidePromise;
  }

  // ============================================================ ВЫВОД
  function clearOut(outEl) { outEl.textContent = ""; }

  function writeChunk(outEl, text, cls) {
    var span = document.createElement("span");
    if (cls) span.className = cls;
    span.appendChild(document.createTextNode(text));
    outEl.appendChild(span);
  }
  function writeInfo(outEl, text) { writeChunk(outEl, text, "dsa-run-out__info"); }
  function writeErr(outEl, text) { writeChunk(outEl, text, "dsa-run-out__err"); }

  function attachIO(py, outEl) {
    py.setStdout({ batched: function (s) { writeChunk(outEl, s); } });
    py.setStderr({ batched: function (s) { writeChunk(outEl, s, "dsa-run-out__err"); } });
  }

  // Перехват импортируемых тяжёлых пакетов — показать прогресс пользователю.
  var HEAVY_RE = /\b(?:import|from)\s+(numpy|pandas|scipy|matplotlib|sklearn|sympy|statsmodels)\b/;

  async function renderFigures(py, outEl) {
    var proxy;
    try {
      proxy = py.runPython(
        "def _dsa_figs():\n" +
        "    import sys\n" +
        "    if 'matplotlib' not in sys.modules:\n" +
        "        return []\n" +
        "    import matplotlib.pyplot as _plt, io as _io, base64 as _b64\n" +
        "    _out = []\n" +
        "    for _n in _plt.get_fignums():\n" +
        "        _f = _plt.figure(_n)\n" +
        "        _buf = _io.BytesIO()\n" +
        "        _f.savefig(_buf, format='png', bbox_inches='tight', dpi=110)\n" +
        "        _buf.seek(0)\n" +
        "        _out.append(_b64.b64encode(_buf.read()).decode())\n" +
        "    _plt.close('all')\n" +
        "    return _out\n" +
        "_dsa_figs()"
      );
    } catch (e) { return false; }
    if (!proxy) return false;
    var arr = proxy.toJs ? proxy.toJs() : proxy;
    var drew = false;
    arr.forEach(function (b64) {
      var img = document.createElement("img");
      img.className = "dsa-run-fig";
      img.alt = "График matplotlib";
      img.src = "data:image/png;base64," + b64;
      outEl.appendChild(img);
      drew = true;
    });
    if (proxy.destroy) proxy.destroy();
    return drew;
  }

  // ============================================================ A) RUN
  function getSource(block) {
    var code = block.querySelector("pre > code") || block.querySelector("code");
    return code ? (code.textContent || "") : "";
  }

  function buildRunPanel(block) {
    var panel = document.createElement("div");
    panel.className = "dsa-run";

    var bar = document.createElement("div");
    bar.className = "dsa-run__bar";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "dsa-btn dsa-btn--primary dsa-run-btn";
    btn.textContent = "Запустить";
    bar.appendChild(btn);
    panel.appendChild(bar);

    var out = document.createElement("div");
    out.className = "dsa-run-out";
    out.hidden = true;
    panel.appendChild(out);

    var busy = false;
    btn.addEventListener("click", async function () {
      if (busy) return;
      busy = true;
      btn.disabled = true;
      out.hidden = false;
      clearOut(out);
      var code = getSource(block);
      var firstRun = !pyodidePromise;
      if (firstRun) writeInfo(out, "Загружаю Python (первый запуск, ~несколько секунд)...\n");
      try {
        var py = await ensurePyodide();
        if (firstRun) clearOut(out);
        attachIO(py, out);
        var hasImport = /^\s*(?:import|from)\s+\w/m.test(code);
        if (hasImport) {
          writeInfo(out, "Подгружаю библиотеки...\n");
          try { await py.loadPackagesFromImports(code); } catch (e) {}
          clearOut(out);
          attachIO(py, out);
        }
        await py.runPythonAsync(code);
        await renderFigures(py, out);
        if (!out.childNodes.length) writeInfo(out, "Готово (без вывода)");
      } catch (e) {
        var msg = String(e && e.message ? e.message : e);
        var mm = msg.match(/ModuleNotFoundError: No module named '([^']+)'/);
        if (mm) {
          writeErr(out, "Пакет «" + mm[1] + "» недоступен в браузерном Python (Pyodide). " +
            "В браузере работают stdlib, numpy, pandas, scipy, matplotlib, scikit-learn, sympy, statsmodels. " +
            "Этот пример запусти локально.");
        } else {
          writeErr(out, msg);
        }
      } finally {
        btn.disabled = false;
        btn.textContent = "Запустить снова";
        busy = false;
      }
    });

    return panel;
  }

  function collectPythonBlocks(root) {
    var seen = (typeof WeakSet !== "undefined") ? new WeakSet() : null;
    var list = [];
    function add(d) {
      if (!d) return;
      if (seen) { if (seen.has(d)) return; seen.add(d); }
      else if (list.indexOf(d) !== -1) return;
      list.push(d);
    }
    root.querySelectorAll("div.language-python").forEach(add);
    root.querySelectorAll("div.highlight").forEach(function (d) {
      if (d.querySelector('code[class*="language-python"]')) add(d);
    });
    return list;
  }

  function initRun(root) {
    collectPythonBlocks(root).forEach(function (block) {
      if (block.dataset.dsaRun) return;
      block.dataset.dsaRun = "1";
      var panel = buildRunPanel(block);
      block.parentNode.insertBefore(panel, block.nextSibling);
    });
  }

  // ============================================================ B) PYEXERCISE
  var SOL_RE = /^#\s*-+\s*solution\s*-+\s*$/i;
  var TESTS_RE = /^#\s*-+\s*tests?\s*-+\s*$/i;

  function trimBlankEdges(lines) {
    var a = 0, b = lines.length;
    while (a < b && lines[a].trim() === "") a++;
    while (b > a && lines[b - 1].trim() === "") b--;
    return lines.slice(a, b);
  }

  function parsePyex(raw) {
    var lines = raw.replace(/\r/g, "").split("\n");
    var sol = [], tests = [], mode = null;
    lines.forEach(function (line) {
      if (SOL_RE.test(line)) { mode = "sol"; return; }
      if (TESTS_RE.test(line)) { mode = "tests"; return; }
      if (mode === "sol") sol.push(line);
      else if (mode === "tests") tests.push(line);
    });
    return {
      solution: trimBlankEdges(sol).join("\n"),
      tests: trimBlankEdges(tests).join("\n")
    };
  }

  // Прогон tests изолированно: вердикт читаем из переменной, traceback
  // (он содержит исходник tests) ученику НЕ показываем.
  var TEST_RUNNER = [
    "_dsa_result = 'OK'",
    "try:",
    "    exec(_dsa_tests_src, globals())",
    "except AssertionError as _e:",
    "    _dsa_result = 'FAIL: ' + (str(_e) or 'проверка не прошла')",
    "except Exception as _e:",
    "    _dsa_result = 'ERR: ' + type(_e).__name__ + ': ' + str(_e)",
    "_dsa_result"
  ].join("\n");

  function buildPyex(block, data, idx) {
    var card = document.createElement("div");
    card.className = "dsa-card dsa-pyex";

    var label = document.createElement("div");
    label.className = "dsa-pyex__label";
    label.textContent = "Задание";
    card.appendChild(label);

    var ta = document.createElement("textarea");
    ta.className = "dsa-pyex-code";
    ta.spellcheck = false;
    ta.setAttribute("autocomplete", "off");
    ta.setAttribute("autocorrect", "off");
    ta.setAttribute("autocapitalize", "off");
    ta.value = data.solution;
    var lineCount = (data.solution.match(/\n/g) || []).length + 1;
    ta.rows = Math.max(6, Math.min(24, lineCount + 1));
    card.appendChild(ta);

    var bar = document.createElement("div");
    bar.className = "dsa-pyex__bar";
    var checkBtn = document.createElement("button");
    checkBtn.type = "button";
    checkBtn.className = "dsa-btn dsa-btn--primary dsa-pyex__check";
    checkBtn.textContent = "Проверить";
    var resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.className = "dsa-btn dsa-btn--ghost dsa-pyex__reset";
    resetBtn.textContent = "Сбросить";
    bar.appendChild(checkBtn);
    bar.appendChild(resetBtn);
    card.appendChild(bar);

    var verdict = document.createElement("div");
    verdict.className = "dsa-pyex__verdict";
    verdict.hidden = true;
    card.appendChild(verdict);

    var out = document.createElement("div");
    out.className = "dsa-run-out";
    out.hidden = true;
    card.appendChild(out);

    function setVerdict(state, text) {
      verdict.hidden = false;
      verdict.className = "dsa-pyex__verdict is-" + state;
      verdict.textContent = text;
    }

    // Поддержка Tab внутри textarea (вставка отступа вместо перехода фокуса).
    ta.addEventListener("keydown", function (e) {
      if (e.key === "Tab") {
        e.preventDefault();
        var s = ta.selectionStart, en = ta.selectionEnd;
        ta.value = ta.value.slice(0, s) + "    " + ta.value.slice(en);
        ta.selectionStart = ta.selectionEnd = s + 4;
      }
    });

    resetBtn.addEventListener("click", function () {
      ta.value = data.solution;
      verdict.hidden = true;
      out.hidden = true;
      clearOut(out);
    });

    var busy = false;
    checkBtn.addEventListener("click", async function () {
      if (busy) return;
      busy = true;
      checkBtn.disabled = true;
      var solution = ta.value;
      var combined = solution + "\n" + data.tests;
      out.hidden = false;
      clearOut(out);
      var firstRun = !pyodidePromise;
      if (firstRun) writeInfo(out, "Загружаю Python (первый запуск, ~несколько секунд)...\n");
      setVerdict("run", "Проверяю...");
      try {
        var py = await ensurePyodide();
        if (firstRun) clearOut(out);
        attachIO(py, out);
        if (HEAVY_RE.test(combined)) {
          writeInfo(out, "Подгружаю библиотеки...\n");
        }
        try { await py.loadPackagesFromImports(combined); } catch (e) {}
        clearOut(out);
        attachIO(py, out);
        // 1) код ученика (ошибки тут — его, показываем полностью)
        await py.runPythonAsync(solution);
        await renderFigures(py, out);
        // 2) скрытые проверки
        py.globals.set("_dsa_tests_src", data.tests);
        var res;
        try {
          res = await py.runPythonAsync(TEST_RUNNER);
        } finally {
          if (py.globals.has("_dsa_tests_src")) py.globals.delete("_dsa_tests_src");
        }
        if (res === "OK") {
          setVerdict("ok", "Решение принято");
          if (window.DSA && DSA.award) {
            DSA.award(15, "pyex:" + location.pathname + ":" + idx);
          }
          if (!out.childNodes.length) { out.hidden = true; }
        } else {
          var msg = String(res || "");
          if (msg.indexOf("FAIL:") === 0) setVerdict("bad", "Тесты не прошли — " + msg.slice(5).trim());
          else if (msg.indexOf("ERR:") === 0) setVerdict("bad", "Ошибка проверки — " + msg.slice(4).trim());
          else setVerdict("bad", "Решение не принято");
        }
      } catch (e) {
        // Ошибка в коде ученика (этап solution) либо загрузки Pyodide.
        setVerdict("bad", "Ошибка в коде");
        writeErr(out, String(e && e.message ? e.message : e));
      } finally {
        checkBtn.disabled = false;
        busy = false;
      }
    });

    return card;
  }

  function initPyex(root) {
    var i = 0;
    // ```pyexercise (неизвестный язык) приходит как div.language-text — parsePyex
    // отфильтрует чужие блоки по отсутствию маркеров solution/tests.
    root.querySelectorAll("div.language-pyexercise, div.language-text").forEach(function (block) {
      if (block.dataset.dsaPyex) return;
      var code = block.querySelector("code");
      var raw = code ? (code.textContent || "") : (block.textContent || "");
      var data = parsePyex(raw);
      if (!data.solution && !data.tests) return;
      block.dataset.dsaPyex = "1";
      var widget = buildPyex(block, data, i);
      i++;
      block.replaceWith(widget);
    });
  }

  // ============================================================ BOOT
  function run() {
    var root = document.querySelector(".md-content__inner") || document.body;
    if (!root) return;
    initPyex(root);
    initRun(root);
  }

  if (typeof document$ !== "undefined" && document$.subscribe) document$.subscribe(run);
  else document.addEventListener("DOMContentLoaded", run);
})();
