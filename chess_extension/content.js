// Chess Model Overlay — content.js (ISOLATED world)
//
// Talks to chess_server.py, which drives the real nnue_engine UCI binary.
// Renders:
//   - a dynamic eval sidebar docked to the LEFT of the chessboard (bar +
//     live depth/nodes/PV readout), replacing the old thin standalone bar
//   - up to two move arrows on the board itself (the engine's suggested
//     move, and the reply it expects)

const POLL_MS = 1000;
const SIDEBAR_WIDTH = 168;
const SIDEBAR_GAP   = 12;
const BAR_WIDTH      = 28;

// Anything with |cp| beyond this is a forced-mate score, not a real
// centipawn evaluation — mirrors MATE_THRESHOLD in chess_server.py.
const MATE_THRESHOLD = 90_000;
const MATE_SCORE     = 900_000;

let lastFen        = null;
let currentFen     = null;
let currentHistoryFens = [];
let lastHistoryLen = -1;   // used to detect a fresh game (send newGame flag)

let overlayCanvas = null;
let sidebarEl      = null;
let enabled    = true;
let lastResult = null;     // most recent parsed info/final payload
let lastTurn   = "white";
let movetimeMs = 800;
let watchSide  = "both";   // "white" | "black" | "both"

let streamPort = null;     // long-lived port to background for SSE

// ── Receive FEN from fen_reader.js (MAIN world) ───────────────────────────────

window.addEventListener("message", (e) => {
  if (e.source === window && e.data?.type === "CHESS_MODEL_FEN") {
    currentFen         = e.data.fen;
    currentHistoryFens = e.data.historyFens || [];
  }
});

function getFen() { return currentFen || null; }

// ── Board helpers ─────────────────────────────────────────────────────────────

function isFlipped() {
  const board = document.querySelector("wc-chess-board") ||
                document.querySelector("chess-board");
  if (!board) return false;
  return board.classList.contains("flipped") ||
         board.getAttribute("flipped") === "true" ||
         board.getAttribute("board-orientation") === "black";
}

function getBoardEl() {
  return document.querySelector("wc-chess-board") ||
         document.querySelector("chess-board");
}

function getBoardRect() {
  const b = getBoardEl();
  return b ? b.getBoundingClientRect() : null;
}

// ── Side filter ───────────────────────────────────────────────────────────────

function shouldAnalyzeFen(fen) {
  if (watchSide === "both") return true;
  const parts = fen.split(" ");
  const fenTurn = parts[1] === "w" ? "white" : "black";
  return fenTurn === watchSide;
}

// ── Eval formatting ───────────────────────────────────────────────────────────

/**
 * scoreCp is from the perspective of the side to move (standard UCI
 * convention). Convert to White's perspective, then to a display string
 * and a 0-100 bar-fill percentage (White's share).
 */
function formatEval(scoreCp, turn) {
  const whiteCp = turn === "white" ? scoreCp : -scoreCp;
  const isMate  = Math.abs(scoreCp) >= MATE_THRESHOLD;

  if (isMate) {
    const pliesToMate = MATE_SCORE - Math.abs(scoreCp);
    const movesToMate = Math.max(1, Math.ceil(pliesToMate / 2));
    const whiteMates  = whiteCp > 0;
    return {
      text: whiteMates ? `M${movesToMate}` : `-M${movesToMate}`,
      whitePct: whiteMates ? 100 : 0,
      isMate: true,
    };
  }

  const pawns = whiteCp / 100;
  const text  = (pawns >= 0 ? "+" : "") + pawns.toFixed(2);
  // Smooth compression so the bar doesn't peg to an extreme on a merely
  // large-but-not-decisive material lead.
  const whitePct = 50 + 50 * Math.tanh(whiteCp / 400);
  return { text, whitePct: Math.max(1, Math.min(99, whitePct)), isMate: false };
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

function ensureSidebar() {
  if (sidebarEl && document.body.contains(sidebarEl)) return sidebarEl;
  if (sidebarEl) sidebarEl.remove();

  sidebarEl = document.createElement("div");
  sidebarEl.id = "chess-model-sidebar";
  sidebarEl.style.cssText = `
    position: fixed;
    z-index: 99998;
    display: flex;
    flex-direction: row;
    background: #17181a;
    border: 1px solid #2c2d30;
    border-radius: 8px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.45);
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  `;

  sidebarEl.innerHTML = `
    <div id="cms-bar" style="position:relative; width:${BAR_WIDTH}px; height:100%; background:#2c2c2c; display:flex; flex-direction:column-reverse;">
      <div id="cms-bar-fill" style="width:100%; background:#eee0c8; height:50%; transition:height 0.35s ease;"></div>
      <div id="cms-bar-label" style="position:absolute; left:0; right:0; text-align:center; font-size:10px; font-weight:700; font-family:monospace; color:#111; bottom:3px;"></div>
    </div>
    <div id="cms-info" style="flex:1; min-width:0; padding:9px 10px; display:flex; flex-direction:column; gap:6px; color:#eee;">
      <div style="display:flex; align-items:center; gap:6px;">
        <div id="cms-dot" style="width:7px; height:7px; border-radius:50%; background:#555; flex-shrink:0;"></div>
        <span style="font-size:10px; font-weight:700; letter-spacing:0.5px; color:#888;">NNUE</span>
      </div>
      <div id="cms-eval" style="font-size:22px; font-weight:800; line-height:1; color:#fff;">--</div>
      <div id="cms-meta" style="font-size:10px; color:#8a8a8a; font-family:monospace; line-height:1.4;"></div>
      <div id="cms-pv" style="font-size:10px; color:#aaa; font-family:monospace; line-height:1.45; word-break:break-all; overflow:hidden;"></div>
    </div>
  `;

  document.body.appendChild(sidebarEl);
  return sidebarEl;
}

function positionSidebar() {
  const rect = getBoardRect();
  if (!rect || !sidebarEl) return;
  // Clamp so a narrow viewport (or a collapsed-but-nonzero left nav) can't
  // push the sidebar off-screen — it stays visible, just closer to the
  // board's left edge than the nominal gap would place it.
  const left = Math.max(8, rect.left - SIDEBAR_WIDTH - SIDEBAR_GAP);
  sidebarEl.style.top    = `${rect.top}px`;
  sidebarEl.style.left   = `${left}px`;
  sidebarEl.style.width  = `${SIDEBAR_WIDTH}px`;
  sidebarEl.style.height = `${rect.height}px`;
}

function renderSidebar(result, searching) {
  if (!getBoardRect()) return; // board not found yet — don't flash a stray box
  const bar = ensureSidebar();
  positionSidebar();

  const dot = bar.querySelector("#cms-dot");
  dot.style.background = searching ? "#ff6b35" : "#4caf50";

  if (!result || result.scoreCp === undefined) return;

  const { text, whitePct, isMate } = formatEval(result.scoreCp, result.turn || lastTurn);
  const whiteBottom = isFlipped() ? (100 - whitePct) : whitePct;

  const fill  = bar.querySelector("#cms-bar-fill");
  const label = bar.querySelector("#cms-bar-label");
  fill.style.height = `${whiteBottom}%`;
  fill.style.background = isMate ? "#ff5555" : "#eee0c8";
  label.textContent = text;
  label.style.color = whiteBottom > 55 ? "#111" : "#eee";
  label.style.bottom = whiteBottom > 12 ? "3px" : "auto";
  label.style.top    = whiteBottom > 12 ? "auto" : "3px";

  const evalEl = bar.querySelector("#cms-eval");
  evalEl.textContent = text;
  evalEl.style.color = isMate ? "#ff5555" : (text.startsWith("-") ? "#ff8a65" : "#8bc98b");

  const metaEl = bar.querySelector("#cms-meta");
  const depth  = result.depth ?? "-";
  const nodes  = result.nodes != null ? formatCount(result.nodes) : "-";
  const nps    = result.nps   != null ? formatCount(result.nps) + "/s" : "-";
  metaEl.textContent = `depth ${depth}\n${nodes} nodes\n${nps}`;
  metaEl.style.whiteSpace = "pre-line";

  const pvEl = bar.querySelector("#cms-pv");
  if (result.pv?.length) {
    pvEl.textContent = result.pv.slice(0, 8).join(" ");
  }
}

function clearSidebar() {
  if (sidebarEl) sidebarEl.style.display = "none";
}
function showSidebar() {
  if (sidebarEl) sidebarEl.style.display = "flex";
}

function formatCount(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000)      return (n / 1_000).toFixed(1) + "k";
  return String(n);
}

// ── Canvas overlay (move arrows) ────────────────────────────────────────────────

function ensureCanvas() {
  if (overlayCanvas && document.body.contains(overlayCanvas)) return overlayCanvas;
  if (overlayCanvas) overlayCanvas.remove();

  overlayCanvas = document.createElement("canvas");
  overlayCanvas.id = "chess-model-overlay";
  overlayCanvas.style.cssText = `
    position: fixed;
    pointer-events: none;
    z-index: 99999;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
  `;
  document.body.appendChild(overlayCanvas);
  return overlayCanvas;
}

function squareToXY(sqName, rect, flipped) {
  const file = sqName.charCodeAt(0) - 97;
  const rank = parseInt(sqName[1]) - 1;
  const sq   = rect.width / 8;
  const col  = flipped ? 7 - file : file;
  const row  = flipped ? rank : 7 - rank;
  return {
    x: rect.left + col * sq + sq / 2,
    y: rect.top  + row * sq + sq / 2,
    sq
  };
}

// rank 0 = the engine's suggested move (bold), rank 1 = the reply it expects (dim)
const ARROW_COLORS = [
  "rgba(255, 107, 53, 0.92)",
  "rgba(255, 180, 80, 0.55)",
];

function drawArrow(ctx, from, to, color, rank) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.sqrt(dx*dx + dy*dy);
  if (len < 1) return;

  const sq      = from.sq;
  const lw      = sq * (rank === 0 ? 0.16 : 0.10);
  const headLen = sq * (rank === 0 ? 0.40 : 0.30);
  const angle   = Math.atan2(dy, dx);
  const ux = dx/len, uy = dy/len;

  const x1 = from.x + ux * sq * 0.22;
  const y1 = from.y + uy * sq * 0.22;
  const x2 = to.x   - ux * headLen * 0.4;
  const y2 = to.y   - uy * headLen * 0.4;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle   = color;
  ctx.lineWidth   = lw;
  ctx.lineCap     = "round";
  ctx.shadowColor = "rgba(0,0,0,0.4)";
  ctx.shadowBlur  = rank === 0 ? 6 : 3;

  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI/6),
             y2 - headLen * Math.sin(angle - Math.PI/6));
  ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI/6),
             y2 - headLen * Math.sin(angle + Math.PI/6));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function renderMoves(pv) {
  const rect = getBoardRect();
  if (!rect || !pv?.length) return;

  const canvas  = ensureCanvas();
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  const ctx     = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const flipped = isFlipped();
  const plies = pv.slice(0, 2); // suggested move + expected reply

  [...plies].reverse().forEach((uci, ri) => {
    const rank = plies.length - 1 - ri;
    if (!uci || uci.length < 4) return;
    const from = squareToXY(uci.slice(0,2), rect, flipped);
    const to   = squareToXY(uci.slice(2,4), rect, flipped);
    drawArrow(ctx, from, to, ARROW_COLORS[rank], rank);
  });
}

function clearOverlay() {
  if (!overlayCanvas) return;
  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// ── Streaming query ───────────────────────────────────────────────────────────

function startStream(fen, newGame) {
  if (streamPort) {
    try { streamPort.disconnect(); } catch (_) {}
    streamPort = null;
  }

  streamPort = chrome.runtime.connect({ name: "stream_analysis" });

  streamPort.onMessage.addListener((data) => {
    if (data.error) {
      console.error("[chess-model] stream error:", data.error);
      return;
    }

    if (data.scoreCp !== undefined) {
      lastResult = data;
      lastTurn   = data.turn;
      renderSidebar(lastResult, !data.final);
      showSidebar();
    }

    if (data.pv?.length) {
      renderMoves(data.pv);
    }

    if (data.final) {
      console.log(`[chess-model] search done: depth=${data.depth} best=${data.bestMove} score=${data.scoreCp}`);
    }
  });

  streamPort.onDisconnect.addListener(() => {
    streamPort = null;
  });

  streamPort.postMessage({ type: "start_stream", fen, movetimeMs, newGame });
}

// ── Poll loop ─────────────────────────────────────────────────────────────────

async function tick(force = false) {
  if (!enabled) { clearOverlay(); clearSidebar(); return; }

  const fen = getFen();
  if (!fen) return;
  if (!force && fen === lastFen) return;

  // ── Side filter: skip opponent's turn if watchSide is set ────────────────
  if (!shouldAnalyzeFen(fen)) {
    if (streamPort) { try { streamPort.disconnect(); } catch (_) {} streamPort = null; }
    lastResult = null;
    clearOverlay();
    lastFen = fen;
    return;
  }

  // A shrinking (or reset-to-zero) history length means a new game loaded
  // on the page — tell the server to clear the engine's TT/history tables.
  const historyLen = currentHistoryFens.length;
  const newGame = lastHistoryLen >= 0 && historyLen < lastHistoryLen;
  lastHistoryLen = historyLen;

  lastFen    = fen;
  lastResult = null;
  clearOverlay();
  console.log("[chess-model] FEN:", fen, "movetimeMs:", movetimeMs, "history:", historyLen, "moves");

  startStream(fen, newGame);
}

setInterval(tick, POLL_MS);
window.addEventListener("resize", () => {
  positionSidebar();
  if (lastResult?.pv?.length) renderMoves(lastResult.pv);
});
window.addEventListener("scroll", () => {
  positionSidebar();
  if (lastResult?.pv?.length) renderMoves(lastResult.pv);
}, { passive: true, capture: true });

// ── Messages from popup ───────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "toggle") {
    enabled = msg.enabled;
    if (!enabled) { clearOverlay(); clearSidebar(); }
    else { if (lastResult?.pv?.length) renderMoves(lastResult.pv); showSidebar(); }
  }
  if (msg.type === "set_movetime") {
    movetimeMs = msg.movetimeMs;
    console.log("[chess-model] movetime set to", movetimeMs, "ms");
  }
  if (msg.type === "set_side") {
    watchSide = msg.side;   // "white" | "black" | "both"
    console.log("[chess-model] watching side:", watchSide);
    lastFen = null;
  }
  if (msg.type === "analyze_now") {
    lastFen = null;
    tick(true);
  }
});

// ── Restore saved preferences on load ────────────────────────────────────────
chrome.storage.local.get(
  { overlayEnabled: true, currentMovetimeMs: 800, currentSide: "both" },
  (prefs) => {
    enabled     = prefs.overlayEnabled;
    movetimeMs  = prefs.currentMovetimeMs;
    watchSide   = prefs.currentSide;
    console.log("[chess-model] prefs restored — enabled:", enabled,
                "movetimeMs:", movetimeMs, "side:", watchSide);
  }
);

console.log("[chess-model] content script loaded");
