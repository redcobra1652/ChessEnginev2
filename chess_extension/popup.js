// popup.js — preferences persisted via chrome.storage.local so they survive
// popup close/reopen and page navigation.

let overlayEnabled    = true;
let currentMovetimeMs = 800;
let currentSide       = "both";

const btnToggle  = document.getElementById("btn-toggle");
const btnAnalyze = document.getElementById("btn-analyze");
const dot        = document.getElementById("server-dot");
const statusText = document.getElementById("status-text");
const simsSlider = document.getElementById("sims-slider");
const simsValue  = document.getElementById("sims-value");
const presets    = document.querySelectorAll(".preset");
const sideBtns   = document.querySelectorAll(".side-btn");

// ── Persistence ───────────────────────────────────────────────────────────────

function savePrefs() {
  chrome.storage.local.set({ overlayEnabled, currentMovetimeMs, currentSide });
}

async function loadPrefs() {
  return new Promise((resolve) => {
    chrome.storage.local.get(
      { overlayEnabled: true, currentMovetimeMs: 800, currentSide: "both" },
      (prefs) => resolve(prefs)
    );
  });
}

// ── Server check ──────────────────────────────────────────────────────────────

function checkServer() {
  chrome.runtime.sendMessage({ type: "ping_server" }, (response) => {
    if (response?.ok) {
      dot.className      = "dot green";
      statusText.textContent = "Server connected";
    } else {
      dot.className      = "dot red";
      statusText.textContent = "Server offline";
    }
  });
}

// ── Tab messaging ─────────────────────────────────────────────────────────────

async function sendToTab(msg) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab) chrome.tabs.sendMessage(tab.id, msg);
}

// ── UI updaters (do NOT call sendToTab — used during init) ───────────────────

function applySimsUI(val) {
  simsSlider.value = val;
  simsValue.textContent = val >= 1000 ? `${(val / 1000).toFixed(1)}s` : `${val}ms`;
  presets.forEach(b => b.classList.toggle("active", parseInt(b.dataset.val) === val));
}

function applyToggleUI(enabled) {
  btnToggle.textContent = enabled ? "● Overlay ON" : "○ Overlay OFF";
  btnToggle.classList.toggle("off", !enabled);
}

function applySideUI(side) {
  sideBtns.forEach(b => b.classList.toggle("active", b.dataset.side === side));
}

// ── Setters (update state + UI + tab + storage) ───────────────────────────────

function setSims(val) {
  currentMovetimeMs = val;
  applySimsUI(val);
  sendToTab({ type: "set_movetime", movetimeMs: val });
  savePrefs();
}

function setSide(side) {
  currentSide = side;
  applySideUI(side);
  sendToTab({ type: "set_side", side });
  savePrefs();
}

function setOverlay(enabled) {
  overlayEnabled = enabled;
  applyToggleUI(enabled);
  sendToTab({ type: "toggle", enabled });
  savePrefs();
}

// ── Event listeners ───────────────────────────────────────────────────────────

simsSlider.addEventListener("input", () => setSims(parseInt(simsSlider.value)));
presets.forEach(btn => btn.addEventListener("click", () => setSims(parseInt(btn.dataset.val))));
sideBtns.forEach(btn => btn.addEventListener("click", () => setSide(btn.dataset.side)));

btnToggle.addEventListener("click", () => setOverlay(!overlayEnabled));

btnAnalyze.addEventListener("click", async () => {
  btnAnalyze.textContent = "⏳ Analyzing…";
  btnAnalyze.disabled    = true;
  await sendToTab({ type: "analyze_now" });
  setTimeout(() => {
    btnAnalyze.textContent = "⚡ Analyze Now";
    btnAnalyze.disabled    = false;
  }, 1200);
});

// ── Init: restore preferences, sync content script, check server ──────────────

(async () => {
  const prefs = await loadPrefs();

  overlayEnabled    = prefs.overlayEnabled;
  currentMovetimeMs = prefs.currentMovetimeMs;
  currentSide       = prefs.currentSide;

  applyToggleUI(overlayEnabled);
  applySimsUI(currentMovetimeMs);
  applySideUI(currentSide);

  // Push saved state into the content script so it's in sync even if it
  // reloaded (e.g. page navigation) while the popup was closed.
  sendToTab({ type: "toggle",       enabled:    overlayEnabled    });
  sendToTab({ type: "set_movetime", movetimeMs: currentMovetimeMs });
  sendToTab({ type: "set_side",     side:       currentSide       });

  checkServer();
})();
