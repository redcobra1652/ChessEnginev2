// fen_reader.js — runs in MAIN world, can access board.game directly

setInterval(() => {
  try {
    const board = document.querySelector("wc-chess-board") ||
                  document.querySelector("chess-board");
    if (!board || !board.game) return;

    // ── Current FEN ──────────────────────────────────────────────────────────
    let fen = null;
    if (typeof board.game.getFEN === "function") fen = board.game.getFEN();
    else if (typeof board.game.fen === "string")    fen = board.game.fen;
    else if (typeof board.game.fen === "function")  fen = board.game.fen();
    if (!fen) return;

    // ── Move history as FEN sequence ─────────────────────────────────────────
    // getHistoryFENs() returns an array of FEN strings, one per half-move,
    // starting from the position AFTER move 1 (index 0 = after white's first
    // move).  We send them to chess_server.py which derives the UCI move list
    // from consecutive FEN pairs using python-chess — this avoids needing TCN
    // decoding or any other chess.com-specific format in the extension.
    let historyFens = [];
    try {
      if (typeof board.game.getHistoryFENs === "function") {
        historyFens = board.game.getHistoryFENs() || [];
      }
    } catch (_) {}

    window.postMessage({
      type: "CHESS_MODEL_FEN",
      fen,
      historyFens,
    }, "*");

  } catch(e) {}
}, 500);
