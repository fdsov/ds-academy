(function () {
  "use strict";
  // Движок интерактивных визуализаций. Блок ```viz рендерится как
  // <div class="language-viz highlight">...<code>КОНФИГ</code></div>.
  // Конфиг — строки KEY: value, обязателен TYPE. Реализованы 3 типа:
  //   TYPE: ci-width  — ширина доверительного интервала ~ 1/sqrt(n)
  //   TYPE: fwer      — рост вероятности ложного срабатывания при множественных тестах
  //   TYPE: simpson   — парадокс Симпсона на A/B-тесте
  // Виджет заменяет исходный блок, перерисовывается на input слайдеров.

  var SVGNS = "http://www.w3.org/2000/svg";

  function el(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }
  function svg(tag, attrs) {
    var e = document.createElementNS(SVGNS, tag);
    if (attrs) for (var k in attrs) if (attrs.hasOwnProperty(k)) e.setAttribute(k, attrs[k]);
    return e;
  }
  function num(v, d) {
    if (v == null) return d;
    var x = parseFloat(("" + v).replace(",", "."));
    return isFinite(x) ? x : d;
  }
  function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }
  function fmtInt(n) {
    try { return Math.round(n).toLocaleString("ru-RU"); }
    catch (e) { return "" + Math.round(n); }
  }
  function fmtPct(x, d) { return (x * 100).toFixed(d == null ? 1 : d) + "%"; }

  function parseConfig(raw) {
    var cfg = {};
    raw.replace(/\r/g, "").split("\n").forEach(function (line) {
      var m = line.match(/^\s*([A-Za-z_]+)\s*:\s*(.*)$/);
      if (m) cfg[m[1].toUpperCase()] = m[2].trim();
    });
    return cfg;
  }

  // ----- общий конструктор управления-слайдера -----
  function sliderRow(labelText) {
    var row = el("div", "dsa-viz__ctrl");
    var head = el("div", "dsa-viz__ctrl-head");
    head.appendChild(el("span", "dsa-viz__ctrl-label", labelText));
    var val = el("span", "dsa-viz__ctrl-val");
    head.appendChild(val);
    var input = document.createElement("input");
    input.type = "range";
    input.className = "dsa-viz__range";
    row.appendChild(head);
    row.appendChild(input);
    return { row: row, input: input, val: val };
  }
  function statBlock(value, label) {
    var s = el("div", "dsa-viz__stat");
    s.appendChild(el("b", null, value));
    s.appendChild(el("span", null, label));
    return s;
  }
  function shell(titleText) {
    var root = el("div", "dsa-viz");
    root.appendChild(el("div", "dsa-viz__label", titleText));
    return root;
  }

  // ============================================================ CI-WIDTH
  function buildCiWidth(cfg) {
    var root = shell(cfg.TITLE || "Доверительный интервал и размер выборки");

    var controls = el("div", "dsa-viz__controls");
    var nCtrl = sliderRow("Размер выборки n");
    nCtrl.input.min = 0; nCtrl.input.max = 1000; nCtrl.input.step = 1;
    var nStart = clamp(num(cfg.N, 1000), 10, 100000);
    nCtrl.input.value = Math.round((Math.log(nStart) / Math.LN10 - 1) / 4 * 1000);

    var pCtrl = sliderRow("Доля p");
    pCtrl.input.min = 0; pCtrl.input.max = 1; pCtrl.input.step = 0.01;
    pCtrl.input.value = clamp(num(cfg.P, 0.5), 0, 1);

    controls.appendChild(nCtrl.row);
    controls.appendChild(pCtrl.row);
    root.appendChild(controls);

    var canvas = el("div", "dsa-viz__canvas");
    root.appendChild(canvas);
    var stats = el("div", "dsa-viz__stats");
    root.appendChild(stats);
    var note = el("div", "dsa-viz__note");
    note.innerHTML = "Ширина интервала пропорциональна <strong>1/&radic;n</strong>: " +
      "чтобы сузить интервал вдвое, размер выборки нужно увеличить в 4 раза.";
    root.appendChild(note);

    function nFromSlider() {
      var t = +nCtrl.input.value / 1000;
      return Math.round(Math.pow(10, 1 + 4 * t));
    }

    function render() {
      var n = nFromSlider();
      var p = +pCtrl.input.value;
      nCtrl.val.textContent = fmtInt(n);
      pCtrl.val.textContent = fmtPct(p, 0);

      var se = Math.sqrt(p * (1 - p) / n);
      var half = 1.96 * se;
      var lo = p - half, hi = p + half;
      var width = 2 * half;

      // ось: x 30..300 -> доля 0..1
      var X0 = 30, X1 = 300, Y = 62, YBAR = 40;
      function xp(v) { return X0 + (X1 - X0) * clamp(v, 0, 1); }

      var s = svg("svg", { viewBox: "0 0 330 86" });
      s.appendChild(svg("line", { x1: X0, y1: Y, x2: X1, y2: Y, "class": "dsa-viz-axis" }));
      [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
        var x = xp(t);
        s.appendChild(svg("line", { x1: x, y1: Y - 3, x2: x, y2: Y + 3, "class": "dsa-viz-tick" }));
        var lab = svg("text", { x: x, y: Y + 16, "text-anchor": "middle", "class": "dsa-viz-ticklabel" });
        lab.textContent = t === 0 ? "0" : t === 1 ? "1" : ("" + t);
        s.appendChild(lab);
      });
      // интервал
      var xl = xp(lo), xh = xp(hi), xc = xp(p);
      s.appendChild(svg("line", { x1: xl, y1: YBAR, x2: xh, y2: YBAR, "class": "dsa-viz-interval" }));
      s.appendChild(svg("line", { x1: xl, y1: YBAR - 7, x2: xl, y2: YBAR + 7, "class": "dsa-viz-cap" }));
      s.appendChild(svg("line", { x1: xh, y1: YBAR - 7, x2: xh, y2: YBAR + 7, "class": "dsa-viz-cap" }));
      s.appendChild(svg("circle", { cx: xc, cy: YBAR, r: 4, "class": "dsa-viz-point" }));
      canvas.innerHTML = "";
      canvas.appendChild(s);

      stats.innerHTML = "";
      stats.appendChild(statBlock(fmtPct(p, 1), "Оценка p"));
      stats.appendChild(statBlock("[" + fmtPct(clamp(lo, 0, 1), 1) + ", " + fmtPct(clamp(hi, 0, 1), 1) + "]", "95% ДИ"));
      stats.appendChild(statBlock("±" + fmtPct(half, 2), "Полуширина"));
      stats.appendChild(statBlock(fmtPct(width, 2), "Ширина интервала"));
    }

    nCtrl.input.addEventListener("input", render);
    pCtrl.input.addEventListener("input", render);
    render();
    return root;
  }

  // =============================================================== FWER
  function buildFwer(cfg) {
    var root = shell(cfg.TITLE || "Множественные сравнения и FWER");

    var controls = el("div", "dsa-viz__controls");
    var mCtrl = sliderRow("Число тестов m");
    mCtrl.input.min = 1; mCtrl.input.max = 50; mCtrl.input.step = 1;
    mCtrl.input.value = clamp(Math.round(num(cfg.M, 10)), 1, 50);

    var aCtrl = sliderRow("Уровень alpha");
    aCtrl.input.min = 0.01; aCtrl.input.max = 0.1; aCtrl.input.step = 0.005;
    aCtrl.input.value = clamp(num(cfg.ALPHA, 0.05), 0.01, 0.1);

    controls.appendChild(mCtrl.row);
    controls.appendChild(aCtrl.row);
    root.appendChild(controls);

    var canvas = el("div", "dsa-viz__canvas");
    root.appendChild(canvas);
    var stats = el("div", "dsa-viz__stats");
    root.appendChild(stats);
    var note = el("div", "dsa-viz__note");
    root.appendChild(note);

    var X0 = 40, X1 = 308, Y0 = 142, Y1 = 18;
    function xm(m) { return X0 + (X1 - X0) * (m - 1) / 49; }
    function yp(p) { return Y0 + (Y1 - Y0) * clamp(p, 0, 1); }

    function render() {
      var m = Math.round(+mCtrl.input.value);
      var alpha = +aCtrl.input.value;
      mCtrl.val.textContent = m;
      aCtrl.val.textContent = alpha.toFixed(3);

      var fwer = 1 - Math.pow(1 - alpha, m);
      var bonf = alpha / m;

      var s = svg("svg", { viewBox: "0 0 330 162" });
      // оси
      s.appendChild(svg("line", { x1: X0, y1: Y0, x2: X1, y2: Y0, "class": "dsa-viz-axis" }));
      s.appendChild(svg("line", { x1: X0, y1: Y0, x2: X0, y2: Y1, "class": "dsa-viz-axis" }));
      [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
        var y = yp(t);
        s.appendChild(svg("line", { x1: X0 - 3, y1: y, x2: X0, y2: y, "class": "dsa-viz-tick" }));
        var lab = svg("text", { x: X0 - 6, y: y + 3, "text-anchor": "end", "class": "dsa-viz-ticklabel" });
        lab.textContent = "" + t;
        s.appendChild(lab);
      });
      [1, 10, 20, 30, 40, 50].forEach(function (mm) {
        var x = xm(mm);
        s.appendChild(svg("line", { x1: x, y1: Y0, x2: x, y2: Y0 + 3, "class": "dsa-viz-tick" }));
        var lab = svg("text", { x: x, y: Y0 + 14, "text-anchor": "middle", "class": "dsa-viz-ticklabel" });
        lab.textContent = "" + mm;
        s.appendChild(lab);
      });
      // кривая FWER(m) при текущем alpha
      var pts = [];
      for (var k = 1; k <= 50; k++) {
        pts.push(xm(k).toFixed(1) + "," + yp(1 - Math.pow(1 - alpha, k)).toFixed(1));
      }
      s.appendChild(svg("polyline", { points: pts.join(" "), "class": "dsa-viz-curve" }));
      // линия номинального alpha
      s.appendChild(svg("line", { x1: X0, y1: yp(alpha), x2: X1, y2: yp(alpha), "class": "dsa-viz-bonf" }));
      // текущая точка + направляющие
      var cx = xm(m), cy = yp(fwer);
      s.appendChild(svg("line", { x1: cx, y1: Y0, x2: cx, y2: cy, "class": "dsa-viz-guide" }));
      s.appendChild(svg("line", { x1: X0, y1: cy, x2: cx, y2: cy, "class": "dsa-viz-guide" }));
      s.appendChild(svg("circle", { cx: cx, cy: cy, r: 4, "class": "dsa-viz-point" }));

      canvas.innerHTML = "";
      canvas.appendChild(s);

      stats.innerHTML = "";
      stats.appendChild(statBlock(fmtPct(fwer, 1), "P(≥ 1 ложной)"));
      stats.appendChild(statBlock(fmtPct(alpha, 1), "Номинальный alpha"));
      stats.appendChild(statBlock(bonf.toFixed(4), "Порог Бонферрони"));

      note.innerHTML = "При " + m + " независимых тестах вероятность хотя бы одной ложной находки " +
        "<strong>1 − (1 − alpha)<sup>m</sup> = " + fmtPct(fwer, 1) + "</strong>. " +
        "Поправка Бонферрони возвращает контроль: тестируем каждую гипотезу на уровне " +
        "alpha/m = <strong>" + bonf.toFixed(4) + "</strong>.";
    }

    mCtrl.input.addEventListener("input", render);
    aCtrl.input.addEventListener("input", render);
    render();
    return root;
  }

  // ============================================================ SIMPSON
  function buildSimpson(cfg) {
    var root = shell(cfg.TITLE || "Парадокс Симпсона в A/B-тесте");

    // фиксированные конверсии сегментов (десктоп выше мобайла, B лучше A в каждом)
    var DA = num(cfg.DA, 19) / 100, MA = num(cfg.MA, 10) / 100;
    var DB = num(cfg.DB, 22) / 100, MB = num(cfg.MB, 12) / 100;
    var total = Math.max(100, num(cfg.TOTAL, 10000));

    var controls = el("div", "dsa-viz__controls");
    var aCtrl = sliderRow("Доля десктопа в группе A");
    aCtrl.input.min = 0; aCtrl.input.max = 1; aCtrl.input.step = 0.01;
    aCtrl.input.value = clamp(num(cfg.WA, 0.8), 0, 1);

    var bCtrl = sliderRow("Доля десктопа в группе B");
    bCtrl.input.min = 0; bCtrl.input.max = 1; bCtrl.input.step = 0.01;
    bCtrl.input.value = clamp(num(cfg.WB, 0.2), 0, 1);

    controls.appendChild(aCtrl.row);
    controls.appendChild(bCtrl.row);
    root.appendChild(controls);

    var legend = el("div", "dsa-viz__note");
    legend.innerHTML = "Конверсия по сегментам фиксирована и в каждом сегменте B лучше A: " +
      "десктоп A " + fmtPct(DA, 0) + " / B " + fmtPct(DB, 0) + ", " +
      "мобайл A " + fmtPct(MA, 0) + " / B " + fmtPct(MB, 0) + ".";
    root.appendChild(legend);

    var tableWrap = el("div", "dsa-viz__canvas");
    root.appendChild(tableWrap);
    var note = el("div", "dsa-viz__note");
    root.appendChild(note);

    function cellHTML(cr, visits) {
      return "<span class='cr'>" + fmtPct(cr, 1) + "</span><span class='sub'>" +
        fmtInt(visits) + " визитов</span>";
    }

    function render() {
      var wA = +aCtrl.input.value, wB = +bCtrl.input.value;
      aCtrl.val.textContent = fmtPct(wA, 0);
      bCtrl.val.textContent = fmtPct(wB, 0);

      var dAv = total * wA, mAv = total - dAv;
      var dBv = total * wB, mBv = total - dBv;
      var convA = dAv * DA + mAv * MA;
      var convB = dBv * DB + mBv * MB;
      var aggA = convA / total, aggB = convB / total;

      var EPS = 1e-9;
      // B лучше в каждом сегменте по построению (DB>DA, MB>MA)
      var paradox = aggA > aggB + EPS;

      var t = el("table", "dsa-viz-table");
      var thead = el("thead");
      var hr = el("tr");
      hr.appendChild(el("th", null, "Сегмент"));
      hr.appendChild(el("th", null, "Группа A"));
      hr.appendChild(el("th", null, "Группа B"));
      thead.appendChild(hr);
      t.appendChild(thead);

      var tb = el("tbody");
      function row(name, crA, vA, crB, vB, isTotal) {
        var tr = el("tr", isTotal ? "is-total" : null);
        tr.appendChild(el("td", null, name));
        var tdA = el("td");
        var tdB = el("td");
        tdA.innerHTML = cellHTML(crA, vA);
        tdB.innerHTML = cellHTML(crB, vB);
        // подсветка победителя
        if (crB > crA + EPS) tdB.classList.add("is-win");
        else if (crA > crB + EPS) tdA.classList.add(isTotal && paradox ? "is-paradox" : "is-win");
        tr.appendChild(tdA);
        tr.appendChild(tdB);
        return tr;
      }
      tb.appendChild(row("Десктоп", DA, dAv, DB, dBv, false));
      tb.appendChild(row("Мобайл", MA, mAv, MB, mBv, false));
      tb.appendChild(row("Итого", aggA, total, aggB, total, true));
      t.appendChild(tb);

      tableWrap.innerHTML = "";
      tableWrap.appendChild(t);

      if (paradox) {
        note.classList.add("dsa-viz__note--warn");
        note.innerHTML = "<strong>Парадокс Симпсона.</strong> В каждом сегменте конверсия группы B выше, " +
          "но в среднем выигрывает A (" + fmtPct(aggA, 1) + " против " + fmtPct(aggB, 1) + "). " +
          "Причина — разный микс трафика: в A преобладает десктоп с высокой конверсией, " +
          "а в B доминирует мобайл. Агрегат смешивает эффект сегмента с эффектом группы.";
      } else {
        note.classList.remove("dsa-viz__note--warn");
        note.innerHTML = "Сейчас миксы трафика близки: агрегат согласован с сегментами — " +
          "B лучше и в среднем (" + fmtPct(aggB, 1) + " против " + fmtPct(aggA, 1) + "). " +
          "Сдвиньте доли так, чтобы в A было больше десктопа, а в B — мобайла, и агрегат перевернётся.";
      }
    }

    aCtrl.input.addEventListener("input", render);
    bCtrl.input.addEventListener("input", render);
    render();
    return root;
  }

  var BUILDERS = { "ci-width": buildCiWidth, "fwer": buildFwer, "simpson": buildSimpson };

  function build(raw) {
    var cfg = parseConfig(raw);
    var type = (cfg.TYPE || "").toLowerCase();
    var fn = BUILDERS[type];
    if (!fn) return null;
    try { return fn(cfg); } catch (e) { return null; }
  }

  function initViz() {
    // pymdownx отдаёт неизвестный язык ```viz как div.language-text — детектим
    // по содержимому (наличие TYPE и известного билдера), как quiz.js/task.js.
    document.querySelectorAll("div.language-viz, div.language-text").forEach(function (container) {
      if (container.dataset.vizDone) return;
      var code = container.querySelector("code");
      if (!code) return;
      var raw = code.textContent || "";
      if (!/^\s*TYPE\s*:/im.test(raw)) return;
      var widget = build(raw);
      if (!widget) return;
      container.dataset.vizDone = "1";
      container.replaceWith(widget);
    });
  }

  if (typeof document$ !== "undefined" && document$.subscribe) document$.subscribe(initViz);
  else document.addEventListener("DOMContentLoaded", initViz);
})();
