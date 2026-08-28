// background.js — service worker

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {

  if (msg.type === "ping_server") {
    fetch("http://127.0.0.1:5001/ping")
      .then(r => r.json())
      .then(data => sendResponse({ ok: true, engine: data.engine }))
      .catch(() => sendResponse({ ok: false }));
    return true;
  }
});

// ── Long-lived port for SSE streaming ─────────────────────────────────────────
// The content script opens a port named "stream_analysis"; we relay SSE
// events back over that port so the sidebar updates on every completed
// iterative-deepening depth, not just at the end of the search.
chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "stream_analysis") return;

  let aborted = false;
  let reader = null;

  port.onMessage.addListener(async (msg) => {
    if (msg.type !== "start_stream") return;

    aborted = false;
    try {
      const resp = await fetch("http://127.0.0.1:5001/analyze_stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fen:        msg.fen,
          movetimeMs: msg.movetimeMs ?? 800,
          newGame:    !!msg.newGame,
        }),
      });

      if (!resp.ok || !resp.body) {
        port.postMessage({ error: `HTTP ${resp.status}` });
        return;
      }

      reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (!aborted) {
        const { done, value } = await reader.read();
        if (done) break;

        buf += decoder.decode(value, { stream: true });
        // SSE events are separated by double newline
        const parts = buf.split("\n\n");
        buf = parts.pop(); // keep incomplete tail

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;
          try {
            const data = JSON.parse(line.slice(5).trim());
            port.postMessage(data);
          } catch (_) {}
        }
      }
    } catch (e) {
      if (!aborted) port.postMessage({ error: e.message });
    }
  });

  port.onDisconnect.addListener(() => {
    aborted = true;
    try { reader?.cancel(); } catch (_) {}
  });
});
