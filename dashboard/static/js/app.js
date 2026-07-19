(() => {
  const $ = (id) => document.getElementById(id);
  let token = localStorage.getItem("tt_token") || "";
  let symbol = "BTCUSDT";

  async function api(path, options = {}) {
    const headers = {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    };
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(path, { ...options, headers });
    if (res.status === 401) {
      token = "";
      localStorage.removeItem("tt_token");
      showLogin();
      throw new Error("Unauthorized");
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  function showLogin() {
    $("loginView").hidden = false;
    $("appView").hidden = true;
  }

  function showApp() {
    $("loginView").hidden = true;
    $("appView").hidden = false;
  }

  $("loginBtn").addEventListener("click", async () => {
    $("loginErr").textContent = "";
    try {
      const data = await api("/api/login", {
        method: "POST",
        body: JSON.stringify({
          username: $("user").value.trim(),
          password: $("pass").value,
        }),
      });
      token = data.token;
      localStorage.setItem("tt_token", token);
      showApp();
      await refreshAll();
    } catch (e) {
      $("loginErr").textContent = e.message || "Login failed";
    }
  });

  function drawChart(candles) {
    const canvas = $("chart");
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (!candles || !candles.length) return;
    const closes = candles.map((c) => Number(c.close));
    const min = Math.min(...closes);
    const max = Math.max(...closes);
    const pad = (max - min) * 0.08 || 1;
    const lo = min - pad;
    const hi = max + pad;
    ctx.strokeStyle = "rgba(61,206,167,0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    closes.forEach((v, i) => {
      const x = (i / (closes.length - 1 || 1)) * (w - 8) + 4;
      const y = h - ((v - lo) / (hi - lo)) * (h - 12) - 6;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    // fill
    const lastY = h - ((closes[closes.length - 1] - lo) / (hi - lo)) * (h - 12) - 6;
    ctx.lineTo(w - 4, h);
    ctx.lineTo(4, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(61,206,167,0.25)");
    grad.addColorStop(1, "rgba(61,206,167,0)");
    ctx.fillStyle = grad;
    ctx.fill();
  }

  async function refreshAll() {
    const [status, trades, risk, signals, ml, audits, market] = await Promise.all([
      api("/api/status"),
      api("/api/trades"),
      api("/api/risk"),
      api("/api/signals"),
      api("/api/ml"),
      api("/api/audits?limit=30"),
      api(`/api/market/${symbol}`),
    ]);

    $("modeBadge").textContent = status.testnet ? "TESTNET" : "MAINNET";
    const pct = Number(risk.pnl_pct || 0);
    $("pnlPct").textContent = `${pct.toFixed(2)}%`;
    $("pnlPct").className = `metric ${pct >= 0 ? "pos" : "neg"}`;
    $("pnlAbs").textContent = `${Number(risk.realized_pnl || 0).toFixed(4)} USDT`;
    $("openCount").textContent = String((trades.open || []).length);
    $("riskLine").textContent = risk.loss_limit_hit
      ? "денний ліміт збитку"
      : `goals: ${(risk.goals_hit || []).join(", ") || "—"}`;

    const xgb = (ml.xgb_metrics && ml.xgb_metrics.accuracy) || 0;
    const lstm = (ml.lstm_metrics && ml.lstm_metrics.accuracy) || 0;
    $("mlAcc").textContent = `XGB ${(xgb * 100).toFixed(1)}% · LSTM ${(lstm * 100).toFixed(1)}%`;
    $("mlLine").textContent = `shadow trades: ${ml.shadow?.trades || 0}`;

    const sig = signals.latest || {};
    const keys = Object.keys(sig);
    $("signals").innerHTML = keys.length
      ? keys
          .map((k) => {
            const s = sig[k];
            return `<div><b>${k}</b> · ${s.direction} · ${(s.confidence * 100).toFixed(1)}% · ${s.strategy}</div>`;
          })
          .join("")
      : "Немає сигналів";

    $("openBody").innerHTML = (trades.open || [])
      .map(
        (t) => `<tr>
          <td>${t.symbol}</td><td>${t.side}</td>
          <td class="mono">${Number(t.entry_price).toFixed(4)}</td>
          <td><button class="btn danger" data-close="${t.symbol}">Close</button></td>
        </tr>`
      )
      .join("");

    $("closedBody").innerHTML = (trades.closed || [])
      .map(
        (t) => `<tr>
          <td>${t.symbol}</td>
          <td class="mono">${Number(t.pnl || 0).toFixed(4)}</td>
          <td>${t.exit_reason || ""}</td>
        </tr>`
      )
      .join("");

    $("auditBody").innerHTML = (audits.items || [])
      .map(
        (a) => `<tr>
          <td class="mono">${(a.created_at || "").replace("T", " ").slice(0, 19)}</td>
          <td>${a.action}</td>
          <td>${(a.details || "").slice(0, 80)}</td>
        </tr>`
      )
      .join("");

    $("chartSymbol").textContent = symbol;
    drawChart(market.candles || []);

    document.querySelectorAll("[data-close]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        await api("/api/close", {
          method: "POST",
          body: JSON.stringify({ symbol: btn.dataset.close, reason: "manual" }),
        });
        await refreshAll();
      });
    });
  }

  document.querySelectorAll("[data-sym]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      symbol = btn.dataset.sym;
      await refreshAll();
    });
  });

  $("refreshBtn").addEventListener("click", () => refreshAll().catch(console.error));
  $("scanBtn").addEventListener("click", async () => {
    await api("/api/scan", { method: "POST" });
    await refreshAll();
  });

  if (token) {
    showApp();
    refreshAll().catch(() => showLogin());
    setInterval(() => refreshAll().catch(() => {}), 15000);
  } else {
    showLogin();
  }
})();
