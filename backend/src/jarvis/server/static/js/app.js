/* ==========================================================================
   J.A.R.V.I.S. — neural interface controller
   Vanilla ES2020. No build step, no external dependencies.
   ========================================================================== */
(() => {
  "use strict";

  // -----------------------------------------------------------------------
  // Utilities
  // -----------------------------------------------------------------------
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const TOKEN = new URLSearchParams(location.search).get("token") || localStorage.getItem("jarvis.token") || "";
  if (TOKEN) localStorage.setItem("jarvis.token", TOKEN);

  const escapeHtml = (value) =>
    String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[char]);

  const formatBytes = (value) => {
    if (!value || value < 1) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let size = Number(value);
    let index = 0;
    while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; }
    return `${size >= 100 ? size.toFixed(0) : size.toFixed(1)} ${units[index]}`;
  };

  const formatRate = (value) => `${formatBytes(value)}/s`;

  const formatDuration = (seconds) => {
    if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "—";
    let total = Math.max(0, Math.floor(seconds));
    const days = Math.floor(total / 86400); total %= 86400;
    const hours = Math.floor(total / 3600); total %= 3600;
    const minutes = Math.floor(total / 60);
    const secs = total % 60;
    if (days) return `${days}d ${hours}h`;
    if (hours) return `${hours}h ${minutes}m`;
    if (minutes) return `${minutes}m ${secs}s`;
    return `${secs}s`;
  };

  const timeOf = (iso) => {
    if (!iso) return "—";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "—";
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  const clockOf = (iso) => {
    if (!iso) return "—";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "—";
    return date.toLocaleTimeString([], { hour12: false });
  };

  const greeting = () => {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 12) return "Good morning";
    if (hour >= 12 && hour < 18) return "Good afternoon";
    return "Good evening";
  };

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  const authHeaders = (extra = {}) => {
    const headers = { Accept: "application/json", ...extra };
    if (TOKEN) headers.Authorization = `Bearer ${TOKEN}`;
    return headers;
  };

  // -----------------------------------------------------------------------
  // State
  // -----------------------------------------------------------------------
  const state = {
    ws: null,
    wsRetry: 0,
    connected: false,
    hello: null,
    setup: null,
    setupSelection: { llm: null, stt: null, tts: null },
    models: null,
    health: {},
    system: null,
    history: [],
    sparkHistory: [],
    conversations: [],
    activeConversation: null,
    messages: [],
    events: [],
    memory: [],
    activeModelKind: "llm",
    activeTab: "models",
    routingMode: "smart",
    streaming: null,
    busy: false,
    lastLatency: null,
    sessionTokens: 0,
    sessionCost: 0,
    audio: localStorage.getItem("jarvis.audio") === "on",
    voice: localStorage.getItem("jarvis.voice") !== "off",
    audioCtx: null,
    audioEl: null,
    orb: null,
    bootOrb: null,
    meter: null,
    recognition: null,
    recorder: null,
    recorderChunks: [],
    micTimer: null,
    listening: false,
    speaking: false,
    // Streamed speech queue: sentences spoken while the answer is still arriving.
    speechQueue: [],
    speechBuffer: "",
    speechToken: 0,
    speechPumping: false,
    speechStarted: false,
    speechActive: false,
    utter: null,
    // Wake word: opt-in, browser recognition in continuous mode.
    wakeEnabled: localStorage.getItem("jarvis.wake") === "on",
    wakeRecognition: null,
    wakeArmed: false,
    wakeFailures: 0,
    wakeRestart: null,
    wakeHeld: false,
    coreKindPrev: "idle",
    captionActive: false,
    bootDone: false,
    bootTimer: null,
    clockTicks: 0,
    metricsTimer: null,
    setupFilter: { llm: "all", stt: "all", tts: "all" },
    // Model/voice/delivery tuning, independent of which core is engaged.
    preferences: null,
    // A custom core being edited, or null when adding a new one.
    editingCore: null,
  };

  const els = {
    boot: $("#boot"),
    bootLog: $("#bootLog"),
    bootBar: $("#bootBar"),
    bootStatus: $("#bootStatus"),
    bootSkip: $("#bootSkip"),
    setup: $("#setup"),
    setupSummary: $("#setupSummary"),
    setupEngage: $("#setupEngage"),
    setupCancel: $("#setupCancel"),
    app: $("#app"),
    clockTime: $("#clockTime"),
    clockDate: $("#clockDate"),
    chipCore: $("#chipCore"),
    chipUplink: $("#chipUplink"),
    chipMode: $("#chipMode"),
    chipHost: $("#chipHost"),
    audioToggle: $("#audioToggle"),
    wakeToggle: $("#wakeToggle"),
    openTune: $("#openTune"),
    tuneFromMatrix: $("#tuneFromMatrix"),
    openSetup: $("#openSetup"),
    fullscreen: $("#fullscreen"),
    coreStage: $("#coreStage"),
    coreKicker: $("#coreKicker"),
    coreState: $("#coreState"),
    coreDetail: $("#coreDetail"),
    coreModel: $("#coreModel"),
    coreVoiceName: $("#coreVoiceName"),
    coreRouting: $("#coreRouting"),
    coreLatency: $("#coreLatency"),
    chatLog: $("#chatLog"),
    conversationTitle: $("#conversationTitle"),
    input: $("#input"),
    sendBtn: $("#sendBtn"),
    micBtn: $("#micBtn"),
    composerHint: $("#composerHint"),
    composerModel: $("#composerModel"),
    newThread: $("#newThread"),
    openArchive: $("#openArchive"),
    routingSeg: $("#routingSeg"),
    modelTabs: $("#modelTabs"),
    modelList: $("#modelList"),
    refreshHealth: $("#refreshHealth"),
    metrics: $("#metrics"),
    memoryForm: $("#memoryForm"),
    memoryInput: $("#memoryInput"),
    memoryList: $("#memoryList"),
    eventList: $("#eventList"),
    clearLogView: $("#clearLogView"),
    conversationList: $("#conversationList"),
    archiveNew: $("#archiveNew"),
    wsDot: $("#wsDot"),
    wsState: $("#wsState"),
    sbBoot: $("#sbBoot"),
    sbVersion: $("#sbVersion"),
    sbTokens: $("#sbTokens"),
    sbCost: $("#sbCost"),
    sbClients: $("#sbClients"),
    sbUptime: $("#sbUptime"),
    ticker: $("#ticker"),
    toasts: $("#toasts"),
    leftPanel: $("#leftPanel"),
    rightPanel: $("#rightPanel"),
    drawerScrim: $("#drawerScrim"),
    spark: $("#spark"),
    orb: $("#orb"),
    bootOrb: $("#bootOrb"),
    coreBars: $("#coreBars"),
    processList: $("#processList"),
    netRate: $("#netRate"),
    hostName: $("#hostName"),
    hostPlatform: $("#hostPlatform"),
    hostAddress: $("#hostAddress"),
    memTotal: $("#memTotal"),
    diskTotal: $("#diskTotal"),
    hostUptime: $("#hostUptime"),
    powerState: $("#powerState"),
    hostState: $("#hostState"),
    coreCount: $("#coreCount"),
    sparkRange: $("#sparkRange"),
    addCore: $("#addCore"),
    addCoreForm: $("#addCoreForm"),
    addCoreClose: $("#addCoreClose"),
    addCoreCancel: $("#addCoreCancel"),
    addCoreSave: $("#addCoreSave"),
    addCoreSaveLabel: $("#addCoreSaveLabel"),
    addCoreLede: $("#addCoreLede"),
    addCoreTitle: $("#addCoreTitle"),
    addCoreKicker: $("#addCoreKicker"),
    coreFootNote: $("#coreFootNote"),
    coreKind: $("#coreKind"),
    coreId: $("#coreId"),
    coreUrl: $("#coreUrl"),
    coreProbe: $("#coreProbe"),
    coreProbeNote: $("#coreProbeNote"),
    coreModelInput: $("#coreModelInput"),
    coreModelList: $("#coreModelList"),
    coreKey: $("#coreKey"),
    coreType: $("#coreType"),
    coreContext: $("#coreContext"),
    coreMaxOut: $("#coreMaxOut"),
    coreVoiceRow: $("#coreVoiceRow"),
    coreVoice: $("#coreVoice"),
    coreFormat: $("#coreFormat"),
    coreLocalRow: $("#coreLocalRow"),
    coreModelPath: $("#coreModelPath"),
    coreReference: $("#coreReference"),
    coreCaps: $("#coreCaps"),
    coreTools: $("#coreTools"),
    coreVision: $("#coreVision"),
    coreJson: $("#coreJson"),
    coreLocal: $("#coreLocal"),
    coreError: $("#coreError"),
    setupAddCore: $("#setupAddCore"),
    addCoreFromMatrix: $("#addCoreFromMatrix"),
  };

  // -----------------------------------------------------------------------
  // Boot sequence
  // -----------------------------------------------------------------------
  const bootLines = [
    ["Loading arc reactor firmware", "ok", 250],
    ["Mounting neural mesh", "ok", 420],
    ["Scanning model registry", "ok", 520],
    ["Calibrating speech subsystems", "ok", 330],
    ["Verifying cryptographic uplink", "ok", 410],
    ["Synchronizing host telemetry", "ok", 260],
    ["All systems nominal", "ok", 240],
  ];

  function addBootLine(text, kind = "") {
    const line = document.createElement("div");
    line.className = `line ${kind}`;
    line.textContent = text;
    els.bootLog.appendChild(line);
  }

  function runBoot() {
    let elapsed = 0;
    bootLines.forEach(([text, kind, delay], index) => {
      elapsed += delay;
      setTimeout(() => {
        if (state.bootDone) return;
        addBootLine(text, kind);
        els.bootBar.style.width = `${Math.round(((index + 1) / bootLines.length) * 100)}%`;
        els.bootStatus.textContent = index === bootLines.length - 1 ? "ALL SYSTEMS NOMINAL" : "INITIALIZING";
      }, elapsed);
    });
    state.bootTimer = setTimeout(finishBoot, elapsed + 550);
  }

  function finishBoot() {
    if (state.bootDone) return;
    state.bootDone = true;
    clearTimeout(state.bootTimer);
    retireBootOrb();
    els.boot.classList.add("is-done");
    setTimeout(() => els.boot.classList.add("hidden"), 750);
    if (state.setup && !state.setup.completed) {
      showSetup();
    } else if (state.setup && state.setup.completed) {
      showApp();
      if (state.wakeEnabled) wakeStart();
      if (state.audio) setTimeout(() => speak(`${greeting()}, sir. All systems online.`), 900);
    } else {
      showApp();
      toast("Uplink not established. Retrying…", "warn");
    }
  }

  function showSetup() {
    els.setup.classList.remove("hidden");
    els.app.classList.add("hidden");
    els.setupCancel.classList.toggle("hidden", !state.setup?.completed);
    // A previous engage attempt may have left the button mid-flight.
    els.setupEngage.querySelector("span").textContent = "ENGAGE J.A.R.V.I.S.";
    renderSetup();
    document.body.dataset.state = "idle";
  }

  function showApp() {
    els.setup.classList.add("hidden");
    els.app.classList.remove("hidden");
    renderAll();
  }

  // -----------------------------------------------------------------------
  // WebSocket
  // -----------------------------------------------------------------------
  function wsUrl() {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const suffix = TOKEN ? `?token=${encodeURIComponent(TOKEN)}` : "";
    return `${protocol}://${location.host}/ws${suffix}`;
  }

  function setWsState(status) {
    const map = {
      connecting: ["warn", "CONNECTING"],
      online: ["ok", "UPLINK ONLINE"],
      offline: ["err", "UPLINK LOST"],
    };
    const [tone, label] = map[status] || map.offline;
    els.wsDot.className = `dot ${tone}`;
    els.wsState.textContent = label;
    els.chipUplink.innerHTML = `<span class="dot ${tone}"></span>${label}`;
  }

  // The masthead states the truth about the selected language core: the config
  // may name a core whose endpoint is down, and a green chip must not hide it.
  function updateCoreChip() {
    const selected = state.setup?.selection?.llm || state.models?.active?.llm || null;
    if (!selected) {
      els.chipCore.innerHTML = '<span class="dot off"></span>CORE UNSET';
      return;
    }
    const health = healthFor("llm", selected);
    const tone = healthTone(health?.status);
    const label = { ok: "CORE ONLINE", warn: "CORE DEGRADED", err: "CORE OFFLINE" }[tone] || "CORE UNKNOWN";
    els.chipCore.innerHTML = `<span class="dot ${tone}"></span>${label}`;
  }

  function connect() {
    if (state.ws && [WebSocket.OPEN, WebSocket.CONNECTING].includes(state.ws.readyState)) return;
    setWsState("connecting");
    let socket;
    try {
      socket = new WebSocket(wsUrl());
    } catch (error) {
      setWsState("offline");
      scheduleReconnect();
      return;
    }
    state.ws = socket;
    socket.addEventListener("open", () => {
      state.wsRetry = 0;
      state.connected = true;
      setWsState("online");
    });
    socket.addEventListener("message", (event) => {
      let frame;
      try { frame = JSON.parse(event.data); } catch { return; }
      handleFrame(frame);
    });
    socket.addEventListener("close", () => {
      state.connected = false;
      setWsState("offline");
      state.busy = false;
      state.streaming = null;
      scheduleReconnect();
    });
    socket.addEventListener("error", () => {
      setWsState("offline");
    });
  }

  function scheduleReconnect() {
    state.wsRetry += 1;
    const delay = Math.min(15000, 700 * 2 ** Math.min(state.wsRetry, 4));
    setTimeout(connect, delay);
  }

  function sendFrame(type, data = {}) {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
      toast("Uplink offline. Reconnecting…", "warn");
      return false;
    }
    state.ws.send(JSON.stringify({ type, data }));
    return true;
  }

  // -----------------------------------------------------------------------
  // Frame handling
  // -----------------------------------------------------------------------
  function handleFrame(frame) {
    const data = frame.data || {};
    switch (frame.type) {
      case "hello":
        handleHello(data);
        break;
      case "telemetry":
        renderTelemetry(data);
        break;
      case "health":
        state.health = data.health || {};
        renderModels();
        if (!els.setup.classList.contains("hidden")) renderSetup();
        break;
      case "setup.complete":
        handleSetupComplete(data);
        break;
      case "setup.required":
        state.setup = data;
        showSetup();
        toast("Core selection required, sir.", "warn");
        break;
      case "model.switched":
        handleModelSwitched(data);
        break;
      case "models.changed":
        handleModelsChanged(data);
        break;
      case "preferences.updated":
        state.preferences = data.preferences || state.preferences;
        updateCoreIdle();
        if (tune.open) {
          tuneWriteForm(prefsOf());
          renderTune();
        }
        break;
      case "routing.mode":
        state.routingMode = data.mode || state.routingMode;
        applyRouting();
        break;
      case "conversation.created":
        upsertConversation(data);
        renderConversations();
        break;
      case "conversation.renamed":
        updateConversation(data.conversation_id, { title: data.title });
        renderConversations();
        break;
      case "conversation.deleted":
        state.conversations = state.conversations.filter((item) => item.id !== data.conversation_id);
        if (state.activeConversation?.id === data.conversation_id) {
          const next = state.conversations[0];
          if (next) loadConversation(next.id);
          else newThread();
        }
        renderConversations();
        break;
      case "conversation.cleared":
        if (state.activeConversation?.id === data.conversation_id) {
          state.messages = [];
          renderChat();
        }
        break;
      case "memory.created":
        if (!state.memory.some((item) => item.id === data.id)) state.memory.unshift(data);
        renderMemory();
        break;
      case "memory.deleted":
        state.memory = state.memory.filter((item) => item.id !== data.id);
        renderMemory();
        break;
      case "memory.cleared":
        state.memory = [];
        renderMemory();
        break;
      case "event":
        pushEvent(data);
        break;
      case "chat.start":
        handleChatStart(data);
        break;
      case "chat.meta":
        handleChatMeta(data);
        break;
      case "chat.delta":
        handleChatDelta(data);
        break;
      case "chat.tool":
        handleChatTool(data);
        break;
      case "chat.usage":
        break;
      case "chat.done":
        handleChatDone(data);
        break;
      case "chat.error":
        handleChatError(data);
        break;
      case "chat.cancelled":
        handleChatCancelled(data);
        break;
      case "command.result":
        handleCommandResult(data);
        break;
      case "error":
        toast(data.message || "Socket error", "error");
        break;
      case "pong":
        break;
      default:
        break;
    }
  }

  function handleHello(data) {
    state.hello = data;
    state.setup = data.setup || null;
    state.models = data.models || null;
    state.health = data.health || {};
    state.preferences = data.preferences || data.status?.preferences || null;
    state.system = data.system || null;
    state.history = data.history || [];
    state.sparkHistory = state.history.slice(-90);
    state.conversations = data.conversations || [];
    state.activeConversation = data.active_conversation || null;
    state.messages = data.messages || [];
    state.events = data.events || [];
    state.memory = data.memory || [];
    state.routingMode = data.status?.routing_mode || "smart";
    state.busy = false;
    state.streaming = null;
    const selection = state.setup?.selection || {};
    state.setupSelection = {
      llm: selection.llm || null,
      stt: selection.stt ?? null,
      tts: selection.tts || null,
    };
    renderAll();
    if (!state.bootDone) {
      els.bootStatus.textContent = "UPLINK ESTABLISHED";
    } else if (state.setup?.completed && els.app.classList.contains("hidden")) {
      showApp();
    } else if (state.setup && !state.setup.completed && els.setup.classList.contains("hidden")) {
      showSetup();
    }
  }

  function handleSetupComplete(data) {
    state.setup = data.setup || state.setup;
    if (data.status?.routing_mode) state.routingMode = data.status.routing_mode;
    const selection = state.setup?.selection || {};
    state.setupSelection = {
      llm: selection.llm || null,
      stt: selection.stt ?? null,
      tts: selection.tts || null,
    };
    els.setup.classList.add("hidden");
    els.app.classList.remove("hidden");
    renderAll();
    toast("Neural cores engaged.", "ok");
    if (state.audio) speak(`${greeting()}, sir. All systems online.`);
    refreshModels();
  }

  async function handleModelsChanged(data) {
    if (data.health) state.health = data.health;
    state.models = null;
    await refreshModels();
    syncSetupSelection();
    if (state.setup && !state.setup.completed && data.action === "removed") showSetup();
    else if (!els.setup.classList.contains("hidden")) renderSetup();
  }

  function handleModelSwitched(data) {
    if (data.setup) state.setup = data.setup;
    if (data.selection) {
      state.setupSelection = {
        ...state.setupSelection,
        ...data.selection,
      };
    }
    if (data.setup?.selection) {
      state.setupSelection = {
        llm: data.setup.selection.llm || state.setupSelection.llm,
        stt: data.setup.selection.stt ?? state.setupSelection.stt,
        tts: data.setup.selection.tts || state.setupSelection.tts,
      };
    }
    if (data.preferences) state.preferences = data.preferences;
    toast(`${String(data.kind || "core").toUpperCase()} routed through ${data.provider_id}.`, "ok");
    refreshModels();
    updateCoreIdle();
    // A different core means a different catalogue, and the server may have just
    // cleared a model/voice name that belonged to the old one. Re-read the form
    // from server truth so the UI never shows a choice that no longer applies.
    if (data.kind) tune.catalog[data.kind] = null;
    tune.draft = null;
    tune.voiceChoice = null;
    tuneWriteForm(prefsOf());
    if (tune.open) {
      tune.tab = data.kind === "tts" ? "tts" : data.kind === "stt" ? "stt" : "llm";
      renderTune();
      if (data.kind) loadCatalog(data.kind).then(() => { if (tune.open) renderTune(); });
    }
  }

  // -----------------------------------------------------------------------
  // Setup rendering
  // -----------------------------------------------------------------------
  function providerItems(kind) {
    return state.setup?.providers?.[kind] || [];
  }

  function healthFor(kind, id) {
    if (id === "browser") return { status: "ok", detail: "Client-side; always available" };
    return state.health?.[kind]?.[id]?.health || null;
  }

  function healthTone(status) {
    if (status === "ok") return "ok";
    if (status === "degraded") return "warn";
    if (status === "error") return "err";
    return "off";
  }

  function renderSetup() {
    if (!state.setup) return;
    for (const kind of ["llm", "stt", "tts"]) {
      const container = $(`#setup-${kind}`);
      if (!container) continue;
      const filter = state.setupFilter[kind] || "all";
      let items = providerItems(kind).slice();
      if (kind === "stt") {
        items = [
          {
            id: "",
            kind: "stt",
            type: "keyboard",
            model: "Keyboard only",
            local: true,
            enabled: true,
            health: { status: "ok", detail: "No microphone capture" },
            _keyboard: true,
          },
          ...items,
        ];
      }
      items = items.filter((item) => {
        if (item._keyboard) return filter === "all" || filter === "local";
        if (filter === "local") return item.local;
        if (filter === "cloud") return !item.local;
        return true;
      });
      items.sort((a, b) => {
        if (a._keyboard) return -1;
        if (b._keyboard) return 1;
        if (a.enabled !== b.enabled) return a.enabled ? -1 : 1;
        if (a.local !== b.local) return a.local ? -1 : 1;
        return a.id.localeCompare(b.id);
      });
      container.innerHTML = "";
      for (const item of items) {
        const selected = item._keyboard
          ? !state.setupSelection.stt
          : state.setupSelection[kind] === item.id;
        const health = item._keyboard ? { status: "ok" } : (healthFor(kind, item.id) || item.health || { status: "unknown" });
        const tone = healthTone(health.status);
        const locked = !item.enabled;
        const custom = Boolean(item.custom);
        const badge = item._keyboard ? "MANUAL" : item.type === "browser" ? "CLIENT" : item.local ? "LOCAL" : "CLOUD";
        const button = document.createElement("button");
        button.type = "button";
        button.className = `core-card${selected ? " is-selected" : ""}`;
        button.dataset.kind = kind;
        button.dataset.id = item.id;
        if (custom) button.dataset.custom = "1";
        button.disabled = locked;
        button.setAttribute("role", "radio");
        button.setAttribute("aria-checked", selected ? "true" : "false");
        button.title = locked
          ? `${item.id} is disabled. Add credentials in .env and enable it in config/config.yaml.`
          : `${item.id} · ${item.type} · ${item.model}`;
        button.innerHTML = `
          <span class="core-dot ${tone}"></span>
          <span class="core-main">
            <strong>${escapeHtml(item.id || "KEYBOARD ONLY")}</strong>
            <em>${escapeHtml(item.model || item.type || "")}</em>
          </span>
          <span class="core-badges">
            <span class="core-badge ${item.local ? "local" : "cloud"}">${badge}</span>
            ${custom ? '<span class="core-badge custom">YOURS</span>' : ""}
            ${locked ? '<span class="core-badge locked">LOCKED</span>' : ""}
            ${health.status && health.status !== "ok" && !locked ? `<span class="core-badge locked">${escapeHtml(health.status.toUpperCase())}</span>` : ""}
          </span>`;
        if (!custom) {
          container.appendChild(button);
          continue;
        }
        // A user-defined core gets its own row so edit/remove are real
        // controls rather than nested interactive elements.
        const row = document.createElement("div");
        row.className = "core-row";
        const tools = document.createElement("div");
        tools.className = "core-tools";
        const edit = document.createElement("button");
        edit.type = "button";
        edit.className = "icon-btn tiny";
        edit.dataset.editCore = item.id;
        edit.dataset.kind = kind;
        edit.title = `Edit ${item.id}`;
        edit.setAttribute("aria-label", `Edit ${item.id}`);
        edit.innerHTML = '<svg><use href="#i-edit"/></svg>';
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "icon-btn tiny danger";
        remove.dataset.removeCore = item.id;
        remove.dataset.kind = kind;
        remove.title = `Remove ${item.id}`;
        remove.setAttribute("aria-label", `Remove ${item.id}`);
        remove.innerHTML = '<svg><use href="#i-trash"/></svg>';
        tools.append(edit, remove);
        row.append(button, tools);
        container.appendChild(row);
      }
    }
    updateSetupSummary();
  }

  function updateSetupSummary() {
    const llm = state.setupSelection.llm;
    const tts = state.setupSelection.tts;
    const stt = state.setupSelection.stt;
    const bits = [];
    bits.push(llm ? `LANGUAGE <b>${escapeHtml(llm)}</b>` : "LANGUAGE <b>NOT SELECTED</b>");
    bits.push(tts ? `VOICE <b>${escapeHtml(tts)}</b>` : "VOICE <b>NOT SELECTED</b>");
    bits.push(stt ? `AUDIO <b>${escapeHtml(stt)}</b>` : "AUDIO <b>KEYBOARD</b>");
    els.setupSummary.innerHTML = bits.join(" <span style=\"opacity:.35\">·</span> ");
    els.setupEngage.disabled = !(llm && tts);
  }

  function selectCore(kind, id) {
    if (kind === "stt" && !id) state.setupSelection.stt = null;
    else state.setupSelection[kind] = id;
    renderSetup();
    const el = $(`.core-card[data-kind="${kind}"][data-id="${CSS.escape(id)}"]`);
    if (el) el.focus();
  }

  async function saveSetup() {
    const { llm, tts, stt } = state.setupSelection;
    if (!llm || !tts) {
      toast("Select a language core and a voice core, sir.", "warn");
      return;
    }
    els.setupEngage.disabled = true;
    els.setupEngage.querySelector("span").textContent = "ENGAGING…";
    try {
      const data = await api("/setup", {
        method: "POST",
        body: { llm, tts, stt: stt || null },
      });
      state.setup = data.setup || state.setup;
      state.models = null;
      if (data.status?.routing_mode) state.routingMode = data.status.routing_mode;
      els.setup.classList.add("hidden");
      els.app.classList.remove("hidden");
      renderAll();
      // The server also broadcasts setup.complete; handleSetupComplete owns the
      // announcement so this path must not repeat it.
      if (state.audio) speak(`${greeting()}, sir. All systems online.`);
      refreshModels();
    } catch (error) {
      toast(error.message || "Could not engage the selected cores.", "error");
      els.setupEngage.disabled = false;
      els.setupEngage.querySelector("span").textContent = "ENGAGE J.A.R.V.I.S.";
    }
  }

  function openSetup() {
    if (!state.setup) {
      toast("Uplink is still initializing.", "warn");
      return;
    }
    const selection = state.setup.selection || {};
    state.setupSelection = {
      llm: selection.llm || state.setupSelection.llm || null,
      stt: selection.stt ?? state.setupSelection.stt ?? null,
      tts: selection.tts || state.setupSelection.tts || null,
    };
    els.setupEngage.querySelector("span").textContent = "ENGAGE J.A.R.V.I.S.";
    showSetup();
  }

  function closeSetup() {
    if (!state.setup?.completed) {
      toast("Select a language core and a voice core first, sir.", "warn");
      return;
    }
    els.setup.classList.add("hidden");
    els.app.classList.remove("hidden");
    updateCoreIdle();
  }

  // -----------------------------------------------------------------------
  // Model matrix
  // -----------------------------------------------------------------------
  function modelItems(kind) {
    const items = state.models?.providers?.[kind] || [];
    if (kind === "stt") {
      return [
        ...items,
        {
          id: "browser",
          kind: "stt",
          type: "browser",
          model: "browser speech recognition",
          local: false,
          enabled: true,
          capabilities: ["client"],
          voices: [],
        },
      ];
    }
    if (kind === "tts") {
      return [
        ...items,
        {
          id: "browser",
          kind: "tts",
          type: "browser",
          model: "system speech synthesis",
          local: true,
          enabled: true,
          capabilities: ["offline"],
          voices: [],
        },
      ];
    }
    return items;
  }

  function activeFor(kind) {
    const selected = state.setup?.selection?.[kind];
    if (selected) return selected;
    return state.models?.active?.[kind] || null;
  }

  function modelLabel(kind, id) {
    if (!id) return "NO CORE";
    // A tuned model name is what actually runs, so show it instead of the
    // core's default. The core id stays visible in the matrix.
    if (kind === "llm") {
      const chosen = state.preferences?.llm?.model;
      if (chosen) return chosen;
    }
    const item = modelItems(kind).find((candidate) => candidate.id === id);
    return item?.model || id;
  }

  function renderRouting() {
    $$("#routingSeg button").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.mode === state.routingMode);
    });
    els.chipMode.textContent = `MODE · ${state.routingMode.toUpperCase()}`;
    els.coreRouting.textContent = state.routingMode.toUpperCase();
  }

  function applyRouting() {
    renderRouting();
  }

  function renderModelTabs() {
    $$("#modelTabs button").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.kind === state.activeModelKind);
    });
  }

  function renderModels() {
    renderModelTabs();
    const kind = state.activeModelKind;
    const items = modelItems(kind).slice();
    const active = activeFor(kind);
    items.sort((a, b) => {
      if ((a.id === active) !== (b.id === active)) return a.id === active ? -1 : 1;
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1;
      if (a.local !== b.local) return a.local ? -1 : 1;
      return a.id.localeCompare(b.id);
    });
    els.modelList.innerHTML = "";
    for (const item of items) {
      const health = healthFor(kind, item.id) || { status: "unknown" };
      const tone = healthTone(health.status);
      const selected = item.id === active;
      const card = document.createElement("button");
      card.type = "button";
      card.className = `model-card${selected ? " is-active" : ""}`;
      card.dataset.kind = kind;
      card.dataset.id = item.id;
      if (item.custom) card.dataset.custom = "1";
      card.disabled = !item.enabled;
      card.title = item.enabled
        ? `Route ${kind.toUpperCase()} through ${item.id}`
        : `${item.id} is disabled in config/config.yaml`;
      // On the active LLM card the model line must show what really runs: a
      // tuned model name wins over the core's configured default.
      const modelText = kind === "llm" && selected && state.preferences?.llm?.model
        ? state.preferences.llm.model
        : (item.model || "");
      card.innerHTML = `
        <span class="mdot ${tone}"></span>
        <span class="mmain">
          <span class="mid">${escapeHtml(item.id)}</span>
          <span class="mmodel">${escapeHtml(modelText)}</span>
          <span class="mmeta">
            <span class="mtag ${item.local ? "local" : "cloud"}">${item.type === "browser" ? "CLIENT" : item.local ? "LOCAL" : "CLOUD"}</span>
            ${item.custom ? '<span class="mtag yours">YOURS</span>' : ""}
            ${selected ? '<span class="mtag active">ACTIVE</span>' : ""}
            ${!item.enabled ? '<span class="mtag">LOCKED</span>' : ""}
          </span>
        </span>
        <span class="mright">
          <span class="mhealth">${escapeHtml((health.status || "unknown").toUpperCase())}</span>
        </span>`;
      if (!item.custom) {
        els.modelList.appendChild(card);
        continue;
      }
      const row = document.createElement("div");
      row.className = "model-row";
      const tools = document.createElement("div");
      tools.className = "model-tools";
      const edit = document.createElement("button");
      edit.type = "button";
      edit.className = "icon-btn tiny";
      edit.dataset.editCore = item.id;
      edit.dataset.kind = kind;
      edit.title = `Edit ${item.id}`;
      edit.setAttribute("aria-label", `Edit ${item.id}`);
      edit.innerHTML = '<svg><use href="#i-edit"/></svg>';
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "icon-btn tiny danger";
      remove.dataset.removeCore = item.id;
      remove.dataset.kind = kind;
      remove.title = `Remove ${item.id}`;
      remove.setAttribute("aria-label", `Remove ${item.id}`);
      remove.innerHTML = '<svg><use href="#i-trash"/></svg>';
      tools.append(edit, remove);
      row.append(card, tools);
      els.modelList.appendChild(row);
    }
    const metrics = state.models?.metrics || {};
    let requests = 0; let tokens = 0; let cost = 0; let latSum = 0; let latCount = 0;
    for (const item of Object.values(metrics)) {
      requests += item.requests || 0;
      tokens += (item.input_tokens || 0) + (item.output_tokens || 0);
      cost += item.cost_usd || 0;
      if (item.avg_latency_ms) { latSum += item.avg_latency_ms; latCount += 1; }
    }
    state.sessionTokens = tokens;
    state.sessionCost = cost;
    els.metrics.innerHTML = `
      <div class="metric"><span>REQUESTS</span><strong>${requests}</strong></div>
      <div class="metric"><span>TOKENS</span><strong>${tokens.toLocaleString()}</strong></div>
      <div class="metric"><span>COST</span><strong>$${cost.toFixed(4)}</strong></div>
      <div class="metric"><span>AVG LATENCY</span><strong>${latCount ? `${Math.round(latSum / latCount)}ms` : "—"}</strong></div>`;
    els.sbTokens.textContent = tokens.toLocaleString();
    els.sbCost.textContent = `$${cost.toFixed(4)}`;
    updateCoreChip();
    updateCoreIdle();
  }

  async function refreshModels() {
    try {
      const data = await api("/models");
      state.models = data.models || state.models;
      state.health = data.health || state.health;
      state.routingMode = data.routing_mode || state.routingMode;
      state.setup = data.setup || state.setup;
      renderModels();
      renderRouting();
      if (!els.setup.classList.contains("hidden")) renderSetup();
    } catch (error) {
      // A transient probe failure should not blank the HUD.
    }
  }

  async function switchModel(kind, id) {
    if (!id) return;
    try {
      await api("/models/switch", { method: "POST", body: { kind, provider_id: id } });
      toast(`${kind.toUpperCase()} core switched to ${id}.`, "ok");
      refreshModels();
    } catch (error) {
      toast(error.message || "Switch failed.", "error");
    }
  }

  async function setRoutingMode(mode) {
    try {
      await api("/routing/mode", { method: "POST", body: { mode } });
      state.routingMode = mode;
      renderRouting();
      toast(`Routing mode: ${mode}.`, "ok");
    } catch (error) {
      toast(error.message || "Could not change routing mode.", "error");
    }
  }

  // -----------------------------------------------------------------------
  // Tuning: the model inside a core, and the voice behind it
  // -----------------------------------------------------------------------
  // The matrix answers "which core". This answers "which actual model, which
  // actual voice, how does it speak" — the choice a user expects to make
  // directly, without hand-editing YAML or accepting a config default.
  const tune = {
    tab: "llm",
    open: false,
    draft: null,
    catalog: { llm: null, tts: null },
    browserVoices: [],
  };

  const sel = {
    tune: $("#tune"),
    tuneTabs: $("#tuneTabs"),
    tuneLede: $("#tuneLede"),
    tuneSave: $("#tuneSave"),
    tuneSaveLabel: $("#tuneSaveLabel"),
    tuneReset: $("#tuneReset"),
    tuneLlmCore: $("#tuneLlmCore"),
    tuneLlmModel: $("#tuneLlmModel"),
    tuneLlmModelFree: $("#tuneLlmModelFree"),
    tuneLlmModelNote: $("#tuneLlmModelNote"),
    tuneLlmReload: $("#tuneLlmReload"),
    tuneTemp: $("#tuneTemp"),
    tuneTempOut: $("#tuneTempOut"),
    tuneMaxTokens: $("#tuneMaxTokens"),
    tuneTtsCore: $("#tuneTtsCore"),
    tuneVoiceList: $("#tuneVoiceList"),
    tuneVoiceSource: $("#tuneVoiceSource"),
    tuneVoiceNote: $("#tuneVoiceNote"),
    tuneVoiceFree: $("#tuneVoiceFree"),
    tuneRate: $("#tuneRate"),
    tuneRateOut: $("#tuneRateOut"),
    tunePitch: $("#tunePitch"),
    tunePitchOut: $("#tunePitchOut"),
    tuneSpeed: $("#tuneSpeed"),
    tuneSpeedOut: $("#tuneSpeedOut"),
    tuneTestVoice: $("#tuneTestVoice"),
    tuneTestNote: $("#tuneTestNote"),
    tuneSttCore: $("#tuneSttCore"),
    tuneSttModel: $("#tuneSttModel"),
    tuneSttLanguage: $("#tuneSttLanguage"),
    tuneSttNote: $("#tuneSttNote"),
  };

  const prefsOf = () => state.preferences || {};
  const selectedCore = (kind) => state.setup?.selection?.[kind] || null;

  /** The language hint for recognition, as a BCP-47 tag.
   *
   * The recogniser wants a locale ("en-GB"), while the STT tab takes the same
   * short code a transcriber does ("en"). A bare code is widened only when it
   * matches what the user actually picked, so an explicit "en-GB" is kept.
   */
  function recognitionLanguage() {
    const hint = String(prefsOf().stt?.language || "").trim();
    if (!hint) return "en-US";
    return /^[a-z]{2,3}$/i.test(hint) ? `${hint.toLowerCase()}-US` : hint;
  }

  function descriptorFor(kind, id) {
    if (!id) return null;
    return providerItems(kind).find((item) => item.id === id) || null;
  }

  function browserVoices() {
    const synth = window.speechSynthesis;
    if (!synth) return [];
    return (synth.getVoices() || []).slice().sort((a, b) => {
      if (a.lang !== b.lang) return a.lang.localeCompare(b.lang);
      return a.name.localeCompare(b.name);
    });
  }

  /** The contents of the "engaged core" strip, which owns the border. */
  function tuneCoreSummary(kind, id) {
    const descriptor = descriptorFor(kind, id);
    if (!id) {
      // Keyboard-only input is a real choice, not a missing core.
      if (kind === "stt" && state.setup?.completed) {
        return {
          html: "<span>No transcriber engaged · <strong>keyboard only</strong></span>"
            + '<span class="tune-core-meta">MANUAL</span>',
          warn: false,
        };
      }
      return {
        html: `<span>No ${kind.toUpperCase()} core engaged. Pick one from the core matrix first.</span>`,
        warn: true,
      };
    }
    // Show what actually runs: a tuned model/voice wins over the core's default.
    const tuned = kind === "llm"
      ? prefsOf().llm?.model
      : kind === "tts"
        ? (tuneIsBrowserVoice() ? prefsOf().tts?.browser_voice : prefsOf().tts?.voice)
        : prefsOf().stt?.model;
    const label = tuned || descriptor?.model || descriptor?.type || id;
    const where = descriptor?.local ? "LOCAL" : descriptor ? "CLOUD" : "CORE";
    return {
      html: `<span>Engaged core <strong>${escapeHtml(id)}</strong> · ${escapeHtml(label)}</span>`
        + `<span class="tune-core-meta">${where}</span>`,
      warn: false,
    };
  }

  function fillSelect(select, values, chosen, emptyLabel) {
    select.innerHTML = "";
    if (emptyLabel) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = emptyLabel;
      select.appendChild(option);
    }
    for (const value of values) {
      const option = document.createElement("option");
      option.value = typeof value === "string" ? value : value.value;
      option.textContent = typeof value === "string" ? value : value.label;
      if (option.value === chosen) option.selected = true;
      select.appendChild(option);
    }
    if (chosen && !values.some((v) => (typeof v === "string" ? v : v.value) === chosen)) {
      const option = document.createElement("option");
      option.value = chosen;
      option.textContent = `${chosen} (current)`;
      option.selected = true;
      select.appendChild(option);
    }
  }

  async function loadCatalog(kind, { force = false } = {}) {
    const id = selectedCore(kind);
    if (!id || id === "browser") return null;
    if (tune.catalog[kind] && !force && tune.catalog[kind].provider_id === id) {
      return tune.catalog[kind];
    }
    try {
      const data = await api(`/models/${kind}/${encodeURIComponent(id)}/catalog`);
      tune.catalog[kind] = data;
      return data;
    } catch (error) {
      tune.catalog[kind] = { provider_id: id, models: [], voices: [], error: error.message };
      return tune.catalog[kind];
    }
  }

  function tuneReadForm() {
    const draft = tune.draft || (tune.draft = {});
    const browser = tuneIsBrowserVoice();
    const previous = draft.tts || prefsOf().tts || {};
    draft.llm = {
      model: (sel.tuneLlmModelFree.value.trim() || sel.tuneLlmModel.value || null),
      temperature: Number(sel.tuneTemp.value),
      max_tokens: Number(sel.tuneMaxTokens.value) || 2048,
    };
    draft.tts = {
      // Only the field belonging to the engaged engine changes; the other keeps
      // whatever it had, so switching engines back and forth loses nothing.
      voice: browser ? previous.voice ?? null : (tune.voiceChoice || null),
      browser_voice: browser
        ? (tune.voiceChoice || null)
        : previous.browser_voice ?? null,
      rate: Number(sel.tuneRate.value),
      pitch: Number(sel.tunePitch.value),
      speed: Number(sel.tuneSpeed.value),
    };
    draft.stt = {
      model: sel.tuneSttModel.value.trim() || null,
      language: sel.tuneSttLanguage.value.trim() || null,
    };
    return draft;
  }

  function tuneWriteForm(preferences) {
    const llm = preferences.llm || {};
    const tts = preferences.tts || {};
    const stt = preferences.stt || {};
    tune.draft = {
      llm: { ...llm },
      tts: { ...tts },
      stt: { ...stt },
    };

    // The list carries the saved model (fillSelect appends it when the endpoint
    // does not offer it), so the free-text field stays empty until the user
    // needs a name the endpoint will not list.
    sel.tuneLlmModelFree.value = "";
    sel.tuneTemp.value = llm.temperature ?? 0.65;
    sel.tuneTempOut.textContent = Number(sel.tuneTemp.value).toFixed(2);
    sel.tuneMaxTokens.value = llm.max_tokens ?? 2048;

    tune.browserVoices = browserVoices();
    // Read the choice from the record only: the form is being re-synced from
    // server truth, so a stale typed value must not be resurrected.
    tune.voiceChoice = (tuneIsBrowserVoice() ? tts.browser_voice : tts.voice) || null;

    sel.tuneVoiceFree.value = "";
    sel.tuneRate.value = tts.rate ?? 1;
    sel.tuneRateOut.textContent = `${Number(sel.tuneRate.value).toFixed(2)}×`;
    sel.tunePitch.value = tts.pitch ?? 1;
    sel.tunePitchOut.textContent = Number(sel.tunePitch.value).toFixed(2);
    sel.tuneSpeed.value = tts.speed ?? 1;
    sel.tuneSpeedOut.textContent = `${Number(sel.tuneSpeed.value).toFixed(2)}×`;

    sel.tuneSttModel.value = stt.model || "";
    sel.tuneSttLanguage.value = stt.language || "";
  }

  function renderVoiceList(voices, chosen, { browser = false } = {}) {
    sel.tuneVoiceList.innerHTML = "";
    if (!voices.length) {
      const empty = document.createElement("p");
      empty.className = "voice-empty";
      empty.textContent = browser
        ? "This browser exposes no speech voices. It may have none installed, or has not "
          + "published its list yet — reopen this dialog after installing one."
        : "The core has not listed its voices. Type a voice id below, or press TEST SERVER "
          + "VOICE to hear whatever it uses by default.";
      sel.tuneVoiceList.appendChild(empty);
      return;
    }
    for (const voice of voices) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `voice-chip${voice.value === chosen ? " is-active" : ""}`;
      button.dataset.voice = voice.value;
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", voice.value === chosen ? "true" : "false");
      button.innerHTML = `<span>${escapeHtml(voice.label)}</span>${voice.meta ? `<em>${escapeHtml(voice.meta)}</em>` : ""}`;
      sel.tuneVoiceList.appendChild(button);
    }
  }

  function tuneIsBrowserVoice() {
    return selectedCore("tts") === "browser";
  }

  function renderTune() {
    const preferences = prefsOf();
    const llmCore = selectedCore("llm");
    const ttsCore = selectedCore("tts");
    const sttCore = selectedCore("stt");

    for (const [node, kind, id] of [
      [sel.tuneLlmCore, "llm", llmCore],
      [sel.tuneTtsCore, "tts", ttsCore],
      [sel.tuneSttCore, "stt", sttCore],
    ]) {
      const summary = tuneCoreSummary(kind, id);
      node.innerHTML = summary.html;
      node.classList.toggle("warn", summary.warn);
    }

    // Language model list: from the endpoint, never from the YAML defaults.
    const llmCatalog = tune.catalog.llm;
    const models = llmCatalog?.models || [];
    const currentModel = preferences.llm?.model || "";
    fillSelect(
      sel.tuneLlmModel,
      models,
      currentModel,
      models.length ? "Endpoint default" : "Press LIST to read the endpoint",
    );
    sel.tuneLlmModel.disabled = !llmCore || llmCore === "browser" || !models.length;
    const modelHint = $("#tuneLlmModelFreeHint");
    if (modelHint) {
      modelHint.textContent = currentModel
        ? "Overrides the list above. Empty keeps the chosen model."
        : "Any name the endpoint serves. Empty uses the core's own default.";
    }
    // Placeholder must not read as a selected value or repeat the core default.
    sel.tuneLlmModelFree.placeholder = currentModel
      ? "another model name"
      : descriptorFor("llm", llmCore)?.model || "qwen2.5:14b";
    sel.tuneLlmModelNote.textContent = llmCatalog?.error
      ? models.length
        ? `${models.length} name${models.length === 1 ? "" : "s"} from the core's own config; `
          + `the live list could not be read — ${llmCatalog.error}`
        : `Endpoint said: ${llmCatalog.error}`
      : models.length
        ? `${models.length} model${models.length === 1 ? "" : "s"} served by this endpoint.`
        : "Read from the endpoint, not the config file.";

    // Voice: browser engines expose their own catalogue; server cores ask their API.
    const browser = tuneIsBrowserVoice();
    const ttsCatalog = tune.catalog.tts;
    const serverVoices = ttsCatalog?.voices || [];
    const voices = browser
      ? tune.browserVoices.map((voice) => ({
          value: voice.voiceURI || voice.name,
          label: voice.name,
          meta: voice.lang + (voice.default ? " · default" : ""),
        }))
      : serverVoices.map((name) => ({ value: name, label: name, meta: "" }));

    sel.tuneVoiceSource.textContent = browser ? "BROWSER SPEECH ENGINE" : "FROM YOUR VOICE CORE";
    sel.tuneVoiceFree.placeholder = browser ? "uri or exact voice name" : "af_bella";
    const voiceHint = $("#tuneVoiceFreeHint");
    if (voiceHint) {
      voiceHint.textContent = browser
        ? "A voice name or URI this browser may expose later; leave empty for the system default."
        : "For a voice the core owns but does not list.";
    }
    sel.tuneVoiceNote.textContent = browser
      ? voices.length
        ? `${voices.length} voice${voices.length === 1 ? "" : "s"} installed in this browser. Click one to hear it.`
        : "No voices from this browser's speech engine."
      : ttsCatalog?.error
        ? `The voice core said: ${ttsCatalog.error}`
        : `${voices.length} voice${voices.length === 1 ? "" : "s"} offered by this core. Click one to hear it.`;

    // A saved voice the catalogue does not list is still a real choice: show it
    // at the top instead of silently dropping the user's selection.
    const chosen = tune.voiceChoice || "";
    if (chosen && !voices.some((voice) => voice.value === chosen)) {
      voices.unshift({ value: chosen, label: chosen, meta: "current" });
    }
    renderVoiceList(voices, chosen, { browser });

    sel.tuneTestVoice.querySelector("span").textContent = browser
      ? "TEST BROWSER VOICE"
      : "TEST SERVER VOICE";

    // Rate and pitch belong to the browser engine; speed belongs to a server
    // core. Leaving the wrong sliders live invites a change that never applies.
    for (const input of [sel.tuneRate, sel.tunePitch]) input.disabled = !browser;
    sel.tuneSpeed.disabled = browser;
    const deliveryNote = $("#tuneDeliveryNote");
    if (deliveryNote) {
      deliveryNote.textContent = browser
        ? "RATE and PITCH are applied by this browser's speech engine as it speaks."
        : "SPEED is sent to the voice core with each request; RATE and PITCH do not apply to server voices.";
    }
    sel.tuneRate.closest(".field")?.classList.toggle("is-muted", !browser);
    sel.tunePitch.closest(".field")?.classList.toggle("is-muted", !browser);
    sel.tuneSpeed.closest(".field")?.classList.toggle("is-muted", browser);

    // Browser recognition consumes the language hint but not a model name;
    // keyboard-only input consumes neither. Disabling what cannot apply keeps
    // the form from storing a choice that never runs.
    const sttBrowser = sttCore === "browser";
    const sttServer = Boolean(sttCore) && !sttBrowser;
    const sttUsable = sttServer || sttBrowser;
    sel.tuneSttNote.textContent = sttServer
      ? "The model and language are sent with each clip you speak."
      : sttBrowser
        ? "Your browser's recogniser takes the language hint; it has no model to choose."
        : "Keyboard-only input sends no audio, so there is nothing to tune here. "
          + "Engage an audio input core in the matrix first.";
    sel.tuneSttModel.disabled = !sttServer;
    sel.tuneSttModel.closest(".field")?.classList.toggle("is-muted", !sttServer);
    sel.tuneSttLanguage.disabled = !sttUsable;
    sel.tuneSttLanguage.closest(".field")?.classList.toggle("is-muted", !sttUsable);
    const sttModelHint = $("#tuneSttModelHint");
    if (sttModelHint) {
      sttModelHint.textContent = sttServer
        ? "A model the core also serves. Empty keeps its own."
        : sttBrowser
          ? "Fixed by the browser; there is no model name to send."
          : "No transcriber is engaged.";
    }
    const sttLanguageHint = $("#tuneSttLanguageHint");
    if (sttLanguageHint) {
      sttLanguageHint.textContent = sttBrowser
        ? "Locale for the recogniser, e.g. en-GB."
        : sttServer
          ? "ISO code. Empty auto-detects."
          : "No transcriber is engaged.";
    }

    sel.tuneSave.disabled = !llmCore || !ttsCore;
    renderTuneTab();
  }

  function renderTuneTab() {
    for (const button of $$("#tuneTabs button")) {
      const active = button.dataset.tuneTab === tune.tab;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    }
    for (const panel of $$("[data-tune-panel]")) {
      panel.classList.toggle("is-active", panel.dataset.tunePanel === tune.tab);
    }
  }

  async function openTune(tab = null) {
    if (tab) tune.tab = tab;
    if (!state.preferences) {
      try {
        state.preferences = await api("/preferences");
      } catch {
        state.preferences = {};
      }
    }
    tuneWriteForm(prefsOf());
    renderTune();
    sel.tune.classList.remove("hidden");
    tune.open = true;
    document.body.classList.add("no-scroll");
    // Fetch the real catalogues in the background; the pickers fill in when they land.
    if (selectedCore("llm")) {
      loadCatalog("llm").then(() => { if (tune.open) renderTune(); });
    }
    if (selectedCore("tts") && !tuneIsBrowserVoice()) {
      loadCatalog("tts").then(() => { if (tune.open) renderTune(); });
    }
    const target = document.querySelector(`#tuneTabs button[data-tune-tab="${tune.tab}"]`);
    target?.focus();
  }

  /** Keep Tab inside the open dialog; the page behind it is not reachable. */
  function trapTuneFocus(event) {
    if (event.key !== "Tab" || !tune.open) return;
    const focusable = $$(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      sel.tune,
    ).filter((node) => node.offsetParent !== null);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function closeTune() {
    sel.tune.classList.add("hidden");
    tune.open = false;
    document.body.classList.remove("no-scroll");
    $("#openTune")?.focus();
  }

  async function saveTune() {
    const draft = tuneReadForm();
    sel.tuneSave.disabled = true;
    sel.tuneSaveLabel.textContent = "APPLYING…";
    try {
      const data = await api("/preferences", { method: "PATCH", body: draft });
      state.preferences = data;
      sel.tuneSaveLabel.textContent = "APPLIED";
      toast("Model and voice settings applied.", "ok");
      closeTune();
    } catch (error) {
      sel.tuneSaveLabel.textContent = "APPLY";
      toast(error.message || "Could not apply those settings.", "error");
    } finally {
      sel.tuneSave.disabled = false;
      setTimeout(() => { sel.tuneSaveLabel.textContent = "APPLY"; }, 1200);
    }
  }

  async function resetTune() {
    sel.tuneReset.disabled = true;
    try {
      const data = await api("/preferences", {
        method: "PATCH",
        body: {
          llm: { model: null, temperature: 0.65, max_tokens: 2048 },
          tts: { voice: null, rate: 1, pitch: 1, speed: 1 },
          stt: { model: null, language: null },
        },
      });
      state.preferences = data;
      tuneWriteForm(data);
      renderTune();
      toast("Tuning reset to the cores' own defaults.", "ok");
    } catch (error) {
      toast(error.message || "Could not reset.", "error");
    } finally {
      sel.tuneReset.disabled = false;
    }
  }

  async function testVoice() {
    const draft = tuneReadForm();
    const note = sel.tuneTestNote;
    note.className = "tune-test-note";
    if (tuneIsBrowserVoice()) {
      const synth = window.speechSynthesis;
      if (!synth) {
        note.textContent = "This browser has no speech engine.";
        note.classList.add("err");
        return;
      }
      synth.cancel();
      const utterance = new SpeechSynthesisUtterance("Good evening, sir. All systems are nominal.");
      const wanted = draft.tts.browser_voice;
      const match = tune.browserVoices.find(
        (voice) => (voice.voiceURI || voice.name) === wanted || voice.name === wanted,
      );
      if (match) utterance.voice = match;
      utterance.rate = clamp(draft.tts.rate || 1, 0.5, 2);
      utterance.pitch = clamp(draft.tts.pitch || 1, 0.5, 2);
      utterance.onend = () => { note.textContent = "Played."; note.classList.add("ok"); };
      utterance.onerror = () => { note.textContent = "The browser refused to play it."; note.classList.add("err"); };
      synth.speak(utterance);
      note.textContent = "Playing…";
      return;
    }

    // A local cloning core computes for minutes per sentence, and a silent
    // button looks broken. Show elapsed time, say what is happening, and let
    // the user stop waiting without leaving the dialog stuck.
    const started = Date.now();
    const core = descriptorFor("tts", selectedCore("tts"));
    const local = Boolean(core?.local);
    const describeWait = () => {
      const secs = Math.round((Date.now() - started) / 1000);
      const clock = `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
      note.textContent = local
        ? `Cloning on this machine · ${clock} — a local voice takes minutes per sentence.`
        : `Synthesizing… ${clock}`;
    };
    const ticker = setInterval(describeWait, 1000);
    describeWait();
    sel.tuneTestVoice.disabled = true;
    try {
      const response = await fetch("/api/tts/preview", {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          provider_id: selectedCore("tts"),
          voice: draft.tts.voice,
          speed: draft.tts.speed,
        }),
      });
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const payload = await response.json();
          detail = payload.error || payload.detail || detail;
        } catch { /* not JSON */ }
        throw new Error(detail);
      }
      const blob = await response.blob();
      await new Promise((resolve) => {
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => { URL.revokeObjectURL(url); resolve(); };
        audio.onerror = () => { URL.revokeObjectURL(url); resolve(); };
        audio.play().catch(resolve);
      });
      note.textContent = `Played after ${Math.round((Date.now() - started) / 1000)}s.`;
      note.classList.add("ok");
    } catch (error) {
      note.textContent = error.message || "The voice core refused.";
      note.classList.add("err");
    } finally {
      clearInterval(ticker);
      sel.tuneTestVoice.disabled = false;
    }
  }

  async function refreshHealth() {
    els.refreshHealth.disabled = true;
    els.refreshHealth.textContent = "PROBING…";
    try {
      const data = await api("/models/refresh", { method: "POST" });
      state.health = data.health || {};
      renderModels();
      toast("Health sweep complete.", "ok");
    } catch (error) {
      toast(error.message || "Health sweep failed.", "error");
    } finally {
      els.refreshHealth.disabled = false;
      els.refreshHealth.innerHTML = '<svg><use href="#i-refresh"/></svg>REFRESH HEALTH';
    }
  }

  // -----------------------------------------------------------------------
  // Telemetry
  // -----------------------------------------------------------------------
  function setGauge(name, percent, text) {
    const gauge = document.querySelector(`[data-gauge="${name}"]`);
    if (!gauge) return;
    const circle = $(".gauge-fill", gauge);
    const value = clamp(percent, 0, 100);
    circle.style.strokeDashoffset = String(301.6 * (1 - value / 100));
    gauge.classList.toggle("warn", value >= 75 && value < 90);
    gauge.classList.toggle("high", value >= 90);
    const strong = $(".gauge-copy strong", gauge);
    strong.innerHTML = text || `${Math.round(value)}<i>%</i>`;
    // Long readings (a load average of 21.39, say) get a smaller step so the
    // number never touches the arc.
    gauge.classList.toggle("compact", strong.textContent.length > 4);
  }

  function renderTelemetry(snapshot) {
    if (!snapshot) return;
    state.system = snapshot;
    const compact = {
      timestamp: snapshot.timestamp,
      cpu: snapshot.cpu?.percent || 0,
      memory: snapshot.memory?.percent || 0,
      disk: snapshot.disk?.percent || 0,
    };
    state.sparkHistory.push(compact);
    if (state.sparkHistory.length > 90) state.sparkHistory.shift();

    const cpu = snapshot.cpu || {};
    const memory = snapshot.memory || {};
    const disk = snapshot.disk || {};
    const network = snapshot.network || {};
    const host = snapshot.host || {};
    const cores = cpu.count || 1;
    const load = cpu.load?.[0] || 0;
    setGauge("cpu", cpu.percent || 0);
    setGauge("mem", memory.percent || 0);
    setGauge("disk", disk.percent || 0);
    setGauge("load", (load / cores) * 100, `${load.toFixed(2)}<i>×</i>`);
    els.coreCount.textContent = `${cores} CORES`;
    els.hostState.textContent = (cpu.percent || 0) > 85 ? "LOAD HIGH" : "NOMINAL";
    els.hostState.style.color = (cpu.percent || 0) > 85 ? "var(--gold)" : "";
    els.hostName.textContent = host.hostname || "—";
    els.hostPlatform.textContent = `${host.system || ""} ${host.release || ""}`.trim() || "—";
    els.hostAddress.textContent = host.address || "—";
    els.memTotal.textContent = `${formatBytes(memory.used)} / ${formatBytes(memory.total)}`;
    els.diskTotal.textContent = `${formatBytes(disk.used)} / ${formatBytes(disk.total)}`;
    els.hostUptime.textContent = formatDuration(snapshot.uptime_s);
    if (snapshot.battery) {
      const battery = snapshot.battery;
      els.powerState.textContent = `${Math.round(battery.percent)}% ${battery.plugged ? "· AC" : "· BATTERY"}`;
    } else {
      els.powerState.textContent = "AC / UNKNOWN";
    }
    els.netRate.textContent = `${formatRate(network.recv_rate)} ↓ · ${formatRate(network.send_rate)} ↑`;
    els.coreBars.innerHTML = "";
    for (const value of cpu.per_core || []) {
      const bar = document.createElement("span");
      bar.className = "core-bar";
      const fill = document.createElement("span");
      fill.style.height = `${clamp(value, 2, 100)}%`;
      bar.appendChild(fill);
      els.coreBars.appendChild(bar);
    }
    els.processList.innerHTML = "";
    for (const process of snapshot.processes || []) {
      const row = document.createElement("div");
      row.className = "proc-row";
      row.innerHTML = `<span class="pname">${escapeHtml(process.name)}</span><span class="pcpu">${Number(process.cpu).toFixed(1)}%</span><span class="pmem">${Number(process.memory).toFixed(1)}%</span>`;
      els.processList.appendChild(row);
    }
    drawSpark();
    updateStatusBar();
  }

  function drawSpark() {
    const canvas = els.spark;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    const dpr = window.devicePixelRatio || 1;
    const height = 58;
    if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, height);
    ctx.strokeStyle = "rgba(255,255,255,.055)";
    ctx.lineWidth = 1;
    for (let y = 0; y <= height; y += height / 3) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(rect.width, y); ctx.stroke();
    }
    const data = state.sparkHistory.slice(-60);
    if (data.length < 2) return;
    const draw = (values, color, fill) => {
      const points = values.map((value, index) => ({
        x: (index / Math.max(1, values.length - 1)) * rect.width,
        y: height - (clamp(value, 0, 100) / 100) * (height - 6) - 3,
      }));
      if (fill) {
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, `${color}33`);
        gradient.addColorStop(1, `${color}00`);
        ctx.beginPath();
        ctx.moveTo(points[0].x, height);
        points.forEach((point) => ctx.lineTo(point.x, point.y));
        ctx.lineTo(points[points.length - 1].x, height);
        ctx.closePath();
        ctx.fillStyle = gradient;
        ctx.fill();
      }
      ctx.beginPath();
      points.forEach((point, index) => { if (index === 0) ctx.moveTo(point.x, point.y); else ctx.lineTo(point.x, point.y); });
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.3;
      ctx.stroke();
    };
    draw(data.map((item) => item.cpu), "#7ee2f5", true);
    draw(data.map((item) => item.memory), "rgba(255,255,255,.34)", false);
  }

  function updateStatusBar() {
    if (!state.hello) return;
    const started = state.hello.status?.started_at || (Date.now() / 1000);
    els.sbUptime.textContent = formatDuration((Date.now() / 1000) - started);
    els.sbClients.textContent = state.hello.status?.clients ?? "—";
    els.sbBoot.textContent = state.hello.boot_id || "—";
    els.sbVersion.textContent = state.hello.version || "0.2.0";
  }

  // -----------------------------------------------------------------------
  // Chat
  // -----------------------------------------------------------------------
  function appendMessage(message, options = {}) {
    const role = message.role || "assistant";
    const article = document.createElement("article");
    const thinking = options.thinking || false;
    article.className = `msg msg-${role}${thinking ? " msg-thinking" : ""}`;
    if (message.id) article.dataset.id = message.id;
    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.textContent = role === "user" ? "U" : role === "error" ? "!" : role === "system" ? "S" : "J";
    const body = document.createElement("div");
    body.className = "msg-body";
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    const author = role === "user" ? "OPERATOR" : role === "error" ? "FAULT" : role === "system" ? "SYSTEM" : "J.A.R.V.I.S.";
    meta.innerHTML = `<span class="author">${author}</span><span class="mtime">${timeOf(message.created_at)}</span>${message.model ? `<span class="model">${escapeHtml(message.model)}</span>` : ""}`;
    const content = document.createElement("div");
    content.className = "msg-content";
    if (thinking) {
      content.innerHTML = 'Establishing neural link<span class="dots"></span>';
    } else if (options.streaming) {
      content.classList.add("is-streaming");
      content.textContent = message.content || "";
      const cursor = document.createElement("span");
      cursor.className = "stream-cursor";
      content.appendChild(cursor);
    } else {
      content.innerHTML = renderMarkdown(message.content || "");
    }
    body.appendChild(meta);
    body.appendChild(content);
    article.appendChild(avatar);
    article.appendChild(body);
    els.chatLog.appendChild(article);
    scrollChat();
    return article;
  }

  function appendWelcome() {
    if (state.messages.length > 0) return;
    const selection = state.setup?.selection || {};
    const llm = selection.llm || state.models?.active?.llm || "UNSET";
    const tts = selection.tts || "UNSET";
    const stt = selection.stt || "KEYBOARD";
    const text = `**All systems online.** ${greeting()}, sir.\n\nLanguage core **${llm}** · voice core **${tts}** · audio input **${stt}** · routing **${state.routingMode}**.\n\nType \`/help\` for the command index, ask me anything, or enable audio and speak.`;
    appendMessage({ role: "assistant", content: text, created_at: new Date().toISOString(), model: llm });
  }

  function renderChat() {
    els.chatLog.innerHTML = "";
    els.conversationTitle.textContent = state.activeConversation?.title || "Primary Interface";
    if (!state.messages.length) {
      appendWelcome();
      return;
    }
    for (const message of state.messages) appendMessage(message);
    scrollChat();
  }

  function scrollChat() {
    els.chatLog.scrollTop = els.chatLog.scrollHeight;
  }

  /**
   * Render the answer as it arrives.
   *
   * Repainting real Markdown on every token would thrash the layout, so the
   * partial text is written as plain text between frames and upgraded to full
   * Markdown once per animation frame. The result is that headings, bold and
   * lists appear while streaming instead of raw `###` and `**`.
   */
  function updateStreaming(text) {
    if (!state.streaming) return;
    state.streaming.raw += text;
    state.streaming.dirty = true;
    if (state.streaming.frame !== null) return;
    const stream = state.streaming;
    stream.frame = requestAnimationFrame(() => {
      stream.frame = null;
      paintStreaming(true);
    });
    scrollChat();
  }

  /** Paint the current streamed text; `markdown` upgrades it from plain text. */
  function paintStreaming(markdown) {
    const stream = state.streaming;
    if (!stream) return;
    const content = stream.el.querySelector(".msg-content");
    if (!content) return;
    if (markdown) {
      content.innerHTML = renderMarkdown(stream.raw);
      content.classList.add("is-streaming");
      const cursor = document.createElement("span");
      cursor.className = "stream-cursor";
      content.appendChild(cursor);
    } else {
      content.textContent = stream.raw;
    }
    scrollChat();
  }

  function handleChatStart(data) {
    state.busy = true;
    updateComposer();
    setCoreState("processing", "PROCESSING", "Neural link engaged. Awaiting the first token.");
    speechReset();
    if (!state.streaming) {
      const el = appendMessage({ role: "assistant", content: "", created_at: new Date().toISOString(), model: "" }, { thinking: true });
      state.streaming = { requestId: data.request_id, el, raw: "", frame: null, dirty: false };
    }
    if (data.request_id) state.streaming.requestId = data.request_id;
  }

  function handleChatMeta(data) {
    if (!state.streaming) return;
    const model = data.model || "";
    const provider = data.provider_id || "";
    const meta = state.streaming.el.querySelector(".msg-meta");
    if (meta) {
      meta.innerHTML = `<span class="author">J.A.R.V.I.S.</span><span class="mtime">${timeOf(new Date().toISOString())}</span><span class="model">${escapeHtml(model || provider)}</span>`;
    }
    state.streaming.el.classList.remove("msg-thinking");
    const content = state.streaming.el.querySelector(".msg-content");
    if (content && content.textContent.includes("Establishing neural link")) content.textContent = "";
    if (provider) {
      setCoreState("processing", "PROCESSING", `Routing through ${provider}${model ? ` · ${model}` : ""}.`);
      els.coreModel.textContent = model || provider;
      els.composerModel.textContent = model || provider;
    }
  }

  function handleChatDelta(data) {
    if (!state.streaming) {
      const el = appendMessage({ role: "assistant", content: "", created_at: new Date().toISOString(), model: "" }, { streaming: true });
      state.streaming = { requestId: data.request_id, el, raw: "", frame: null, dirty: false };
    }
    state.streaming.el.classList.remove("msg-thinking");
    updateStreaming(data.text || "");
    // Speak each finished sentence now, instead of waiting for the whole answer.
    speechFeed(data.text || "");
  }

  function handleChatTool(data) {
    if (!state.streaming) return;
    const tool = data.tool_call || {};
    const badge = document.createElement("div");
    badge.className = "msg-tools";
    badge.textContent = `TOOL CALL · ${tool.name || "unknown"} ${tool.arguments ? JSON.stringify(tool.arguments) : ""}`;
    state.streaming.el.querySelector(".msg-body")?.appendChild(badge);
  }

  function handleChatDone(data) {
    const message = data.message || {
      role: "assistant",
      content: state.streaming?.raw || "",
      created_at: new Date().toISOString(),
      provider_id: data.provider_id,
      model: data.model,
    };
    if (state.streaming) {
      const el = state.streaming.el;
      // A queued repaint must not run after the final content is written.
      if (state.streaming.frame !== null) {
        cancelAnimationFrame(state.streaming.frame);
        state.streaming.frame = null;
      }
      el.classList.remove("msg-thinking");
      const content = el.querySelector(".msg-content");
      if (content) {
        content.classList.remove("is-streaming");
        content.innerHTML = renderMarkdown(message.content || state.streaming.raw || "");
      }
      if (message.id) el.dataset.id = message.id;
      const meta = el.querySelector(".msg-meta");
      if (meta) {
        meta.innerHTML = `<span class="author">J.A.R.V.I.S.</span><span class="mtime">${timeOf(message.created_at)}</span>${message.model ? `<span class="model">${escapeHtml(message.model)}</span>` : ""}`;
      }
    } else {
      appendMessage(message);
    }
    state.messages.push(message);
    state.streaming = null;
    state.busy = false;
    state.lastLatency = data.latency_ms ?? message.latency_ms ?? null;
    if (data.usage) {
      state.sessionTokens += (data.usage.input_tokens || 0) + (data.usage.output_tokens || 0);
    }
    if (data.cost_usd) state.sessionCost += data.cost_usd;
    if (state.lastLatency) els.coreLatency.textContent = `${Math.round(state.lastLatency)}ms`;
    updateComposer();
    updateConversationAfterMessage(message);
    // Anything after the final full stop still gets said. If the provider sent
    // the answer in one piece, nothing was fed yet, so queue it now.
    if (!state.speechStarted && state.audio && state.voice) {
      state.speechBuffer = message.content || "";
    }
    speechFlush();
    // The turn is over, but the voice may not be: keep the orb in its speaking
    // state until the queue drains, then fall back to standby.
    if (state.speechActive) {
      setCoreState("speaking", "SPEAKING", "Voice core active.");
    } else {
      setCoreState("idle", "STANDBY", "At your service, sir.");
    }
    chime("receive");
    scrollChat();
    refreshModels();
  }

  function handleChatError(data) {
    const el = state.streaming?.el;
    const persisted = data.error_message || null;
    const provider = data.provider_id || data.selected_llm || persisted?.provider_id || "unknown";
    const hints = Array.isArray(data.hints) ? data.hints : [];
    const text = persisted?.content || [
      "**Language core unavailable.**",
      "",
      `\`${data.code || "provider_error"}\` — ${data.message || "The provider did not respond."}`,
      "",
      `Core: \`${provider}\``,
      hints.length ? `\n${hints.map((hint) => `- ${hint}`).join("\n")}` : "",
    ].join("\n");
    const rendered = renderMarkdown(text);
    if (el) {
      el.className = "msg msg-assistant msg-error";
      if (data.message_id) el.dataset.id = data.message_id;
      const avatar = el.querySelector(".msg-avatar");
      if (avatar) avatar.textContent = "!";
      const content = el.querySelector(".msg-content");
      if (content) {
        content.classList.remove("is-streaming");
        content.innerHTML = rendered;
      }
      const meta = el.querySelector(".msg-meta");
      if (meta) meta.innerHTML = `<span class="author">FAULT</span><span class="mtime">${timeOf(new Date().toISOString())}</span><span class="model">${escapeHtml(provider)}</span>`;
    } else {
      const article = appendMessage({
        role: "error",
        content: text,
        created_at: new Date().toISOString(),
        model: provider,
      });
      if (article && data.message_id) article.dataset.id = data.message_id;
    }
    if (persisted) {
      state.messages.push(persisted);
    } else {
      state.messages.push({
        role: "error",
        content: text,
        created_at: new Date().toISOString(),
        model: provider,
      });
    }
    state.streaming = null;
    state.busy = false;
    // A fault ends the turn: do not finish reading a half answer aloud.
    speechReset();
    setCoreState("error", "FAULT", data.message || "The selected core is offline.");
    updateComposer();
    toast(data.message || "Language core unavailable.", "error");
    chime("error");
    if (state.audio && state.voice) speak("I am afraid the language core is offline, sir. Check the model matrix.");
  }

  function handleChatCancelled(data) {
    if (state.streaming) {
      const el = state.streaming.el;
      el.classList.remove("msg-thinking");
      const content = el.querySelector(".msg-content");
      if (content) {
        content.classList.remove("is-streaming");
        content.innerHTML = renderMarkdown("*Transmission cancelled.*");
      }
    }
    state.streaming = null;
    state.busy = false;
    speechReset();
    setCoreState("idle", "STANDBY", "Transmission cancelled.");
    updateComposer();
  }

  function handleCommandResult(data) {
    state.busy = false;
    state.streaming = null;
    if (data.message) {
      state.messages.push(data.message);
      appendMessage(data.message);
    } else if (data.text) {
      appendMessage({ role: "assistant", content: data.text, created_at: new Date().toISOString(), model: "COMMAND" });
    }
    setCoreState("idle", "STANDBY", "Directive executed.");
    updateComposer();
    if (data.action === "clear") {
      state.messages = [];
      renderChat();
    }
    if (data.action === "switch_conversation" && data.action_data?.conversation) {
      state.activeConversation = data.action_data.conversation;
      upsertConversation(data.action_data.conversation);
      state.messages = [];
      renderChat();
      renderConversations();
    }
    if (data.action === "open_setup") {
      if (data.action_data?.setup) state.setup = data.action_data.setup;
      openSetup();
    }
    if (data.action === "voice_on") setAudio(true);
    if (data.action === "voice_off") setAudio(false);
    if (data.error) toast(data.text || "Command failed.", "warn");
    if (data.text) chime("receive");
  }

  function updateComposer() {
    els.sendBtn.disabled = state.busy;
    els.composerHint.textContent = state.busy ? "Neural link busy. Please stand by." : "Type a directive, or use /help.";
    const active = activeFor("llm");
    els.composerModel.textContent = active ? `${modelLabel("llm", active)} · ${state.routingMode.toUpperCase()}` : "NO CORE";
  }

  function setCoreState(kind, label, detail) {
    document.body.dataset.state = kind;
    els.coreState.textContent = label;
    // While a sentence is being spoken the caption already shows it; a generic
    // status line would wipe the words the user is hearing.
    if (!(kind === "speaking" && state.captionActive)) {
      els.coreDetail.textContent = detail || "";
    }
    if (kind !== "speaking") state.captionActive = false;
    const kickers = {
      idle: "NEURAL INTERFACE",
      processing: "COGNITIVE ENGINE",
      listening: "AUDIO INPUT",
      speaking: "VOICE SYNTHESIS",
      error: "SYSTEM FAULT",
    };
    els.coreKicker.textContent = kickers[kind] || "NEURAL INTERFACE";
    state.orb?.setState(kind);
    // The wake listener must not hear JARVIS's own voice, so it is held while
    // the voice speaks and released when the turn ends.
    if (kind === "speaking") wakeHold();
    else if (state.coreKindPrev === "speaking") wakeRelease();
    state.coreKindPrev = kind;
  }

  function voiceLabel() {
    const tts = state.setup?.selection?.tts;
    if (!tts) return "—";
    const prefs = state.preferences?.tts || {};
    if (tts === "browser") {
      const chosen = (prefs.browser_voice || "").trim();
      if (!chosen) return "browser default";
      const voices = browserVoices();
      const match = voices.find(
        (voice) => voice.voiceURI === chosen || voice.name === chosen,
      );
      return match ? match.name : chosen;
    }
    return prefs.voice || modelLabel("tts", tts);
  }

  function updateCoreIdle() {
    if (state.busy) return;
    const active = activeFor("llm");
    els.coreModel.textContent = active ? modelLabel("llm", active) : "NO CORE";
    els.coreVoiceName.textContent = voiceLabel();
    els.composerModel.textContent = active ? `${modelLabel("llm", active)} · ${state.routingMode.toUpperCase()}` : "NO CORE";
    // The voice outlives the turn: the answer is finished long before the last
    // sentence is spoken, and standby must not win that race.
    if (state.speechActive) {
      setCoreState("speaking", "SPEAKING", "Voice core active.");
    } else if (state.hello?.setup_complete === false || (state.setup && !state.setup.completed)) {
      setCoreState("idle", "STANDBY", "Select a language core and a voice core to begin.");
    } else if (document.body.dataset.state !== "error") {
      setCoreState("idle", "STANDBY", "At your service, sir.");
    }
  }

  function sendMessage(text) {
    const message = String(text ?? "").trim();
    if (!message) return;
    if (state.busy) {
      toast("The neural link is still processing, sir.", "warn");
      return;
    }
    if (!state.setup?.completed) {
      toast("Select a language core and a voice core first.", "warn");
      openSetup();
      return;
    }
    const requestId = (crypto.randomUUID && crypto.randomUUID()) || `req_${Date.now()}_${Math.random().toString(16).slice(2)}`;    const local = {
      id: `local_${requestId}`,
      role: "user",
      content: message,
      created_at: new Date().toISOString(),
      model: null,
    };
    state.messages.push(local);
    appendMessage(local);
    els.input.value = "";
    autoGrow();
    state.busy = true;
    updateComposer();
    setCoreState("processing", "PROCESSING", message.startsWith("/") ? "Executing directive." : "Engaging the language core.");
    chime("send");
    const sent = sendFrame("chat", {
      request_id: requestId,
      message,
      conversation_id: state.activeConversation?.id || null,
    });
    if (!sent) {
      state.busy = false;
      updateComposer();
      setCoreState("idle", "STANDBY", "Uplink offline.");
    }
  }

  // -----------------------------------------------------------------------
  // Conversations, memory, events
  // -----------------------------------------------------------------------
  function upsertConversation(conversation) {
    if (!conversation?.id) return;
    const index = state.conversations.findIndex((item) => item.id === conversation.id);
    if (index >= 0) state.conversations[index] = { ...state.conversations[index], ...conversation };
    else state.conversations.unshift(conversation);
  }

  function updateConversation(id, patch) {
    const item = state.conversations.find((conversation) => conversation.id === id);
    if (item) Object.assign(item, patch);
    if (state.activeConversation?.id === id) Object.assign(state.activeConversation, patch);
  }

  function updateConversationAfterMessage(message) {
    const cid = state.activeConversation?.id;
    if (!cid) return;
    updateConversation(cid, {
      message_count: (state.activeConversation.message_count || 0) + 1,
      updated_at: message.created_at || new Date().toISOString(),
    });
    renderConversations();
    setTimeout(refreshConversations, 500);
  }

  async function refreshConversations() {
    try {
      const items = await api("/conversations");
      state.conversations = items;
      renderConversations();
    } catch { /* transient */ }
  }

  async function loadConversation(id) {
    if (!id) return;
    try {
      const data = await api(`/conversations/${encodeURIComponent(id)}`);
      state.activeConversation = data.conversation;
      state.messages = data.messages || [];
      state.busy = false;
      state.streaming = null;
      renderChat();
      renderConversations();
      els.conversationTitle.textContent = state.activeConversation?.title || "Conversation";
      closeDrawers();
      toast(`Thread loaded: ${state.activeConversation?.title || id}`, "ok");
    } catch (error) {
      toast(error.message || "Could not load thread.", "error");
    }
  }

  async function newThread() {
    if (state.busy) {
      toast("Stand by until the current response finishes, sir.", "warn");
      return;
    }
    try {
      const conversation = await api("/conversations", { method: "POST", body: { title: "New conversation" } });
      upsertConversation(conversation);
      state.activeConversation = conversation;
      state.messages = [];
      renderChat();
      renderConversations();
      els.conversationTitle.textContent = conversation.title;
      closeDrawers();
      toast("New thread opened.", "ok");
    } catch (error) {
      toast(error.message || "Could not open a new thread.", "error");
    }
  }

  async function deleteConversation(id) {
    if (!id) return;
    if (!window.confirm("Erase this thread permanently?")) return;
    try {
      await api(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
      state.conversations = state.conversations.filter((item) => item.id !== id);
      if (state.activeConversation?.id === id) {
        const next = state.conversations[0];
        if (next) await loadConversation(next.id);
        else await newThread();
      }
      renderConversations();
      toast("Thread erased.", "ok");
    } catch (error) {
      toast(error.message || "Could not erase thread.", "error");
    }
  }

  function renderConversations() {
    if (!els.conversationList) return;
    els.conversationList.innerHTML = "";
    if (!state.conversations.length) {
      els.conversationList.innerHTML = '<p class="setup-note">No archived threads yet.</p>';
      return;
    }
    for (const conversation of state.conversations) {
      const row = document.createElement("div");
      row.className = `conv-item${conversation.id === state.activeConversation?.id ? " is-active" : ""}`;
      const main = document.createElement("button");
      main.type = "button";
      main.className = "conv-main";
      main.innerHTML = `<span class="conv-title">${escapeHtml(conversation.title || "Untitled")}</span><span class="conv-meta">${conversation.message_count || 0} messages · ${timeOf(conversation.updated_at)}</span>`;
      main.addEventListener("click", () => loadConversation(conversation.id));
      const del = document.createElement("button");
      del.type = "button";
      del.className = "del";
      del.title = "Erase thread";
      del.innerHTML = '<svg><use href="#i-trash"/></svg>';
      del.addEventListener("click", (event) => { event.stopPropagation(); deleteConversation(conversation.id); });
      row.appendChild(main);
      row.appendChild(del);
      els.conversationList.appendChild(row);
    }
  }

  function renderMemory() {
    els.memoryList.innerHTML = "";
    if (!state.memory.length) {
      els.memoryList.innerHTML = '<p class="setup-note">Long-term memory is empty. Store a fact above.</p>';
      return;
    }
    for (const item of state.memory) {
      const row = document.createElement("div");
      row.className = "memory-item";
      const text = document.createElement("div");
      text.className = "mtext";
      text.innerHTML = `${escapeHtml(item.text)}<span class="mtime">${item.created_at ? new Date(item.created_at).toLocaleDateString() : ""}</span>`;
      const del = document.createElement("button");
      del.type = "button";
      del.title = "Erase engram";
      del.innerHTML = '<svg><use href="#i-trash"/></svg>';
      del.addEventListener("click", () => deleteMemory(item.id));
      row.appendChild(text);
      row.appendChild(del);
      els.memoryList.appendChild(row);
    }
  }

  async function addMemory(text) {
    if (!text.trim()) return;
    try {
      const memory = await api("/memory", { method: "POST", body: { text: text.trim() } });
      if (!state.memory.some((item) => item.id === memory.id)) state.memory.unshift(memory);
      renderMemory();
      els.memoryInput.value = "";
      toast("Engram stored.", "ok");
    } catch (error) {
      toast(error.message || "Could not store memory.", "error");
    }
  }

  async function deleteMemory(id) {
    try {
      await api(`/memory/${encodeURIComponent(id)}`, { method: "DELETE" });
      state.memory = state.memory.filter((item) => item.id !== id);
      renderMemory();
      toast("Engram erased.", "ok");
    } catch (error) {
      toast(error.message || "Could not erase engram.", "error");
    }
  }

  function pushEvent(event) {
    state.events.push(event);
    if (state.events.length > 160) state.events.shift();
    const row = document.createElement("div");
    row.className = `event-row ${event.level || "info"}`;
    row.innerHTML = `<span class="etime">${clockOf(event.created_at)}</span><span class="etext">${escapeHtml(event.message)}</span>`;
    els.eventList.appendChild(row);
    while (els.eventList.children.length > 160) els.eventList.removeChild(els.eventList.firstChild);
    if (state.activeTab === "log") els.eventList.scrollTop = els.eventList.scrollHeight;
  }

  function renderEvents() {
    els.eventList.innerHTML = "";
    for (const event of state.events.slice(-120)) {
      const row = document.createElement("div");
      row.className = `event-row ${event.level || "info"}`;
      row.innerHTML = `<span class="etime">${clockOf(event.created_at)}</span><span class="etext">${escapeHtml(event.message)}</span>`;
      els.eventList.appendChild(row);
    }
    els.eventList.scrollTop = els.eventList.scrollHeight;
  }

  // -----------------------------------------------------------------------
  // Tabs / drawers
  // -----------------------------------------------------------------------
  function setTab(tab) {
    state.activeTab = tab;
    $$(".tab").forEach((button) => button.classList.toggle("is-active", button.dataset.tab === tab));
    $$(".tab-panel").forEach((panel) => panel.classList.toggle("is-active", panel.dataset.panel === tab));
    if (tab === "archive") renderConversations();
    if (tab === "memory") renderMemory();
    if (tab === "log") { renderEvents(); els.eventList.scrollTop = els.eventList.scrollHeight; }
  }

  function openDrawer(which) {
    if (which === "left") els.leftPanel.classList.add("is-open");
    if (which === "right") els.rightPanel.classList.add("is-open");
    els.drawerScrim.classList.add("is-open");
  }

  function closeDrawers() {
    els.leftPanel.classList.remove("is-open");
    els.rightPanel.classList.remove("is-open");
    els.drawerScrim.classList.remove("is-open");
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
    else document.exitFullscreen?.();
  }

  // -----------------------------------------------------------------------
  // Core orb
  // -----------------------------------------------------------------------
  /**
   * The orb is the interface's presence: it mirrors the current state and
   * swells with whatever JARVIS is hearing or saying. Both the stage orb and
   * the boot orb are real canvases; the boot one only spins while the boot
   * screen is up, then it is paused so it costs nothing.
   */
  function setupOrb() {
    const Meter = window.JarvisAudioMeter;
    const Orb = window.JarvisOrb;
    if (Meter) state.meter = new Meter();
    if (!Orb) return;
    if (els.orb) {
      state.orb = new Orb(els.orb);
      if (state.meter) state.orb.bindLevel(() => state.meter.level());
      state.orb.setState(document.body.dataset.state || "idle");
    }
    if (els.bootOrb) {
      state.bootOrb = new Orb(els.bootOrb);
      state.bootOrb.setState("processing");
      // The boot sphere is decorative: give it a slow synthetic swell rather
      // than silence, so the first screen still breathes before any audio.
      let phase = 0;
      state.bootOrb.bindLevel(() => {
        phase += 0.02;
        return 0.16 + Math.sin(phase) * 0.1;
      });
    }
  }

  /** Stop the boot sphere once the boot screen is gone. */
  function retireBootOrb() {
    state.bootOrb?.stop();
  }

  // -----------------------------------------------------------------------
  // Wake word — "hey JARVIS"
  // -----------------------------------------------------------------------
  /**
   * An opt-in, local wake word.
   *
   * This uses the browser's own continuous recognition, so it costs nothing on
   * the server and never sends audio anywhere except the browser's speech
   * service. It is deliberately conservative:
   *
   *   - it stays armed between turns, but pauses while JARVIS speaks or the
   *     user is already dictating, so it cannot hear itself;
   *   - two consecutive errors disarm it, because a loop of failing restarts
   *     would drain the battery and spam the log;
   *   - it requires a final result containing a wake phrase before it acts.
   *
   * Saying the wake word while JARVIS is speaking interrupts it, which is how
   * the reference assistants behave.
   */
  const WAKE_PHRASES = [/\bhey\s+jarvis\b/i, /\bhi\s+jarvis\b/i, /\bok(?:ay)?\s+jarvis\b/i, /\bjarvis\b/i];

  function wakeSupported() {
    return Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
  }

  function setWake(enabled) {
    state.wakeEnabled = Boolean(enabled);
    localStorage.setItem("jarvis.wake", state.wakeEnabled ? "on" : "off");
    updateWakeButton();
    if (state.wakeEnabled) wakeStart();
    else wakeStop();
  }

  function updateWakeButton() {
    if (!els.wakeToggle) return;
    const armed = state.wakeEnabled && state.wakeArmed;
    els.wakeToggle.setAttribute("aria-pressed", state.wakeEnabled ? "true" : "false");
    els.wakeToggle.classList.toggle("is-armed", armed);
    els.wakeToggle.title = !wakeSupported()
      ? "Wake word needs a browser with speech recognition"
      : state.wakeEnabled
        ? (armed ? "Wake word armed — say “hey JARVIS”" : "Wake word listening…")
        : "Arm the wake word (“hey JARVIS”)";
  }

  function wakeStart() {
    if (!state.wakeEnabled || state.wakeRecognition) return;
    if (!wakeSupported()) {
      toast("This browser cannot arm the wake word. Chrome, Edge and Safari can.", "warn");
      state.wakeEnabled = false;
      localStorage.setItem("jarvis.wake", "off");
      updateWakeButton();
      return;
    }
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new Recognition();
    recognition.lang = recognitionLanguage();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    state.wakeRecognition = recognition;
    state.wakeFailures = 0;

    recognition.onstart = () => {
      state.wakeArmed = true;
      state.wakeFailures = 0;
      updateWakeButton();
    };
    recognition.onresult = (event) => {
      // Only final results act, so a half-heard phrase cannot fire the wake word.
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        if (!result.isFinal) continue;
        const transcript = (result[0]?.transcript || "").trim();
        if (!transcript) continue;
        if (!WAKE_PHRASES.some((pattern) => pattern.test(transcript))) continue;
        wakeFire(transcript);
        return;
      }
    };
    recognition.onerror = (event) => {
      // "no-speech" and "aborted" are normal between utterances.
      if (event.error === "no-speech" || event.error === "aborted") return;
      state.wakeFailures = (state.wakeFailures || 0) + 1;
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        toast("Microphone access is needed for the wake word.", "warn");
        setWake(false);
        return;
      }
      if (state.wakeFailures >= 2) {
        toast(`Wake word paused: ${event.error}.`, "warn");
        setWake(false);
      }
    };
    recognition.onend = () => {
      state.wakeArmed = false;
      state.wakeRecognition = null;
      updateWakeButton();
      // Continuous mode still ends on its own in some browsers; restart quietly
      // unless something deliberately holds the listener (speech or dictation).
      if (state.wakeEnabled && !state.wakeHeld) {
        state.wakeRestart = setTimeout(() => wakeStart(), 450);
      }
    };
    try {
      recognition.start();
    } catch {
      state.wakeRecognition = null;
      state.wakeArmed = false;
      updateWakeButton();
    }
  }

  function wakeStop() {
    clearTimeout(state.wakeRestart);
    state.wakeRestart = null;
    state.wakeHeld = false;
    if (state.wakeRecognition) {
      try { state.wakeRecognition.abort(); } catch { /* already stopped */ }
      state.wakeRecognition = null;
    }
    state.wakeArmed = false;
    updateWakeButton();
  }

  /**
   * Hold the wake listener back without disarming it.
   *
   * Used while JARVIS speaks (it must not hear its own voice) and while the
   * user dictates (the device is already in use). Releasing it restarts the
   * listener if the user still has it armed.
   */
  function wakeHold() {
    state.wakeHeld = true;
    if (state.wakeRecognition) {
      try { state.wakeRecognition.abort(); } catch { /* already stopped */ }
    }
    updateWakeButton();
  }

  function wakeRelease() {
    if (!state.wakeHeld) return;
    state.wakeHeld = false;
    if (state.wakeEnabled && !state.listening && !state.speaking) {
      clearTimeout(state.wakeRestart);
      state.wakeRestart = setTimeout(() => wakeStart(), 500);
    }
    updateWakeButton();
  }

  function wakeFire(transcript) {
    if (state.listening || state.busy) return;
    // Interrupting JARVIS mid-sentence is the point of a wake word.
    if (state.speaking) speechStop();
    toast("Wake word detected.", "ok");
    chime("receive");
    wakeHold();
    // Give the browser a moment to release the mic before dictation opens it.
    setTimeout(() => {
      setCoreState("listening", "LISTENING", "Wake word acknowledged. Speak, sir.");
      toggleMic();
    }, 260);
  }

  // -----------------------------------------------------------------------
  // Audio / voice
  // -----------------------------------------------------------------------
  function initAudioContext() {
    if (state.audioCtx) return state.audioCtx;
    // One context for the whole interface: chimes, microphone metering and
    // speech playback share it, so the browser only sees a single audio graph.
    // `context` is the meter's lazy getter: creating it here means the graph is
    // already in place before the first mic frame or chime.
    const meterCtx = state.meter?.context;
    if (meterCtx) {
      state.audioCtx = meterCtx;
      return state.audioCtx;
    }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    state.audioCtx = new Ctx();
    return state.audioCtx;
  }

  function chime(kind) {
    if (!state.audio) return;
    const ctx = initAudioContext();
    if (!ctx) return;
    // Called from a click or a key press, so this is the moment a suspended
    // context is allowed to start.
    if (ctx.state === "suspended") state.meter?.resume();
    const now = ctx.currentTime;
    const notes = kind === "send" ? [720, 960] : kind === "error" ? [220, 180] : [880, 1180];
    notes.forEach((frequency, index) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = frequency;
      gain.gain.setValueAtTime(0.0001, now + index * 0.07);
      gain.gain.exponentialRampToValueAtTime(0.035, now + index * 0.07 + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + index * 0.07 + 0.16);
      osc.connect(gain).connect(ctx.destination);
      osc.start(now + index * 0.07);
      osc.stop(now + index * 0.07 + 0.2);
    });
  }

  function updateAudioButton() {
    els.audioToggle.setAttribute("aria-pressed", state.audio ? "true" : "false");
    els.audioToggle.innerHTML = `<svg><use href="${state.audio ? "#i-volume" : "#i-mute"}"/></svg>`;
    els.audioToggle.title = state.audio ? "Audio online" : "Audio muted";
  }

  function setAudio(enabled) {
    state.audio = Boolean(enabled);
    localStorage.setItem("jarvis.audio", state.audio ? "on" : "off");
    if (!state.audio) {
      window.speechSynthesis?.cancel();
      speechReset();
      if (state.audioEl) { state.audioEl.pause(); state.audioEl = null; }
      state.meter?.detachElement();
    } else {
      initAudioContext();
      chime("receive");
      if (state.setup?.completed) speak(`${greeting()}, sir. Audio systems online.`);
    }
    updateAudioButton();
    toast(state.audio ? "Audio online." : "Audio muted.", state.audio ? "ok" : "info");
  }

  /**
   * Turn Markdown into something worth hearing.
   *
   * Speech is not a screen reader: pipes and rule rows must not be read out,
   * heading markers and bullet glyphs are noise, and a table read cell by cell
   * is unintelligible. Tables become a short spoken summary and code becomes a
   * single clause, so the audio stays prose.
   */
  function cleanForSpeech(text) {
    let source = String(text || "");
    // Fenced code: one clause, never the body.
    source = source.replace(/```[\s\S]*?```/g, " (code omitted) ");
    // Markdown tables: drop them entirely; the prose around them carries the
    // answer, and a table read cell by cell is unintelligible.
    source = source.replace(/^[ \t]*\|.*\|[ \t]*$/gm, "");
    source = source.replace(/^[ \t]*\|?[\s:|-]*-[\s:|-]*\|?[ \t]*$/gm, "");

    const tidy = (line) => line
      // Leading list markers, blockquote carets and rule rows.
      .replace(/^[\s>]*[-*+][ \t]+/, "")
      .replace(/^[\s>]*\d+[.)][ \t]+/, "")
      .replace(/^[\s>]+/, "")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, "$1")
      .replace(/\*\*([^*]+)\*\*/g, "$1")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1$2")
      .replace(/~~([^~]+)~~/g, "$1")
      .replace(/[#*_~|]/g, " ")
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim();

    // Each Markdown line becomes its own utterance. A heading or a bullet would
    // otherwise be run into the sentence that follows it, so every line is
    // closed with a full stop before they are joined.
    const parts = [];
    for (const raw of source.split("\n")) {
      const line = tidy(raw);
      if (!line) continue;
      parts.push(/[.!?…:;,]$/.test(line) ? line : `${line}.`);
    }
    return parts.join(" ").replace(/\.{2,}/g, ".").trim().slice(0, 1400);
  }

  // -----------------------------------------------------------------------
  // Speech — streamed a sentence at a time
  // -----------------------------------------------------------------------
  /**
   * JARVIS starts talking before the answer is finished.
   *
   * Sentences are queued as the model emits them and spoken in order, so the
   * first words are heard while the rest is still being generated. The queue is
   * reset at the start of every turn, and starting the microphone cancels the
   * voice so the user can interrupt — both behaviours are lifted straight from
   * the better open-source assistants.
   *
   * Server TTS and browser TTS both feed the same queue. If the selected server
   * voice fails, that sentence falls back to the browser voice rather than
   * dropping out of the answer.
   */

  // Punctuation that ends a spoken sentence. Kept free of lookbehind so it runs
  // in every browser this interface supports.
  const SENTENCE_BREAK = /[.!?…]+["'”’)\]]*(\s+|$)/g;

  function speechReset() {
    state.speechToken += 1;
    state.speechQueue = [];
    state.speechBuffer = "";
    state.speechStarted = false;
    state.speechActive = false;
  }

  /** Cancel speech and return the interface to standby. */
  function speechStop() {
    speechReset();
    speechCaptionClear();
    if (state.audioEl) {
      try { state.audioEl.pause(); } catch { /* already stopped */ }
      state.audioEl = null;
    }
    if (state.utter) {
      try { window.speechSynthesis?.cancel(); } catch { /* nothing queued */ }
      state.utter = null;
    }
    state.meter?.detachElement();
    state.speaking = false;
    if (!state.busy) updateCoreIdle();
  }

  function speechEnqueue(text) {
    const spoken = cleanForSpeech(text);
    if (spoken.length < 2) return;
    state.speechQueue.push(spoken);
    state.speechActive = true;
    speechPump();
  }

  /** Split text into sentences for the queue. */
  function splitSentences(text) {
    const sentences = [];
    let start = 0;
    SENTENCE_BREAK.lastIndex = 0;
    let match;
    while ((match = SENTENCE_BREAK.exec(text))) {
      const end = match.index + match[0].length;
      const piece = text.slice(start, end).trim();
      if (piece) sentences.push(piece);
      start = end;
    }
    const tail = text.slice(start).trim();
    if (tail) sentences.push(tail);
    return sentences.length ? sentences : [text.trim()];
  }

  /** Index just past the last complete sentence, or -1 if there is not one yet. */
  function speechBoundary(text) {
    SENTENCE_BREAK.lastIndex = 0;
    let end = -1;
    let match;
    while ((match = SENTENCE_BREAK.exec(text))) {
      end = match.index + match[0].length;
    }
    // A tiny fragment ("Yes.") sounds like a glitch; wait for a little context.
    if (end < 24 && end < text.length) return -1;
    return end;
  }

  /** Consume streamed model text, speaking each sentence as it completes. */
  function speechFeed(delta) {
    if (!state.audio || !state.voice || !delta) return;
    state.speechBuffer += delta;
    const buffer = state.speechBuffer;
    const boundary = speechBoundary(buffer);
    if (boundary <= 0) return;
    const chunk = buffer.slice(0, boundary);
    // Never speak from inside a code fence that is still open.
    if ((chunk.match(/```/g) || []).length % 2 === 1) return;
    state.speechBuffer = buffer.slice(boundary);
    state.speechStarted = true;
    splitSentences(chunk).forEach(speechEnqueue);
  }

  /** Speak whatever is left when the answer ends. */
  function speechFlush() {
    const rest = state.speechBuffer.trim();
    state.speechBuffer = "";
    if (!rest) return;
    if ((rest.match(/```/g) || []).length % 2 === 1) return; // unterminated code
    splitSentences(rest).forEach(speechEnqueue);
  }

  // The stage caption doubles as a subtitle for the spoken sentence, so the
  // interface shows the words at the moment they are heard.
  function speechCaption(sentence) {
    if (!els.coreDetail) return;
    const trimmed = String(sentence || "").trim();
    if (!trimmed) return;
    const short = trimmed.length > 160 ? `${trimmed.slice(0, 157)}…` : trimmed;
    els.coreDetail.textContent = short;
    els.coreDetail.classList.add("is-caption");
    state.captionActive = true;
  }

  function speechCaptionClear() {
    if (!els.coreDetail) return;
    els.coreDetail.classList.remove("is-caption");
    state.captionActive = false;
    if (document.body.dataset.state === "speaking") {
      els.coreDetail.textContent = "Voice core active.";
    }
  }

  function speechPump() {
    if (state.speechPumping) return;
    state.speechPumping = true;
    const token = state.speechToken;
    (async () => {
      while (state.speechQueue.length && token === state.speechToken) {
        const sentence = state.speechQueue.shift();
        try {
          // Claim the speaking state before the audio starts, so the orb and the
          // caption change together even if the engine is slow to fire onstart.
          if (token === state.speechToken) {
            state.speaking = true;
            setCoreState("speaking", "SPEAKING", "Voice core active.");
            speechCaption(sentence);
          }
          await speechSay(sentence, token);
        } catch {
          // One bad sentence must not stall the rest of the answer.
        }
      }
      state.speechPumping = false;
      if (token === state.speechToken && !state.speechQueue.length) {
        state.speechActive = false;
        state.speaking = false;
        speechCaptionClear();
        state.meter?.detachElement();
        if (!state.busy) updateCoreIdle();
      }
    })();
  }

  async function speechSay(sentence, token) {
    const selected = state.setup?.selection?.tts || state.setupSelection.tts;
    if (!selected || selected === "browser") return speechSayBrowser(sentence, token);
    try {
      return await speechSayServer(sentence, token);
    } catch {
      if (token !== state.speechToken) return;
      // The selected server voice is offline; the browser voice keeps JARVIS audible.
      return speechSayBrowser(sentence, token);
    }
  }

  /** The browser voice the user picked, or the best British default. */
  function resolveBrowserVoice(preferences) {
    const synth = window.speechSynthesis;
    if (!synth) return null;
    const voices = synth.getVoices() || [];
    if (!voices.length) return null;
    const wanted = (preferences?.tts?.browser_voice || "").trim();
    if (wanted) {
      const chosen = voices.find(
        (voice) => voice.voiceURI === wanted || voice.name === wanted,
      );
      // A saved voice that the browser no longer exposes is a real loss; fall
      // through to the default rather than speak nothing at all.
      if (chosen) return chosen;
    }
    return (
      voices.find(
        (voice) =>
          /en-GB/i.test(voice.lang) &&
          /(daniel|arthur|george|oliver|male|uk english male)/i.test(voice.name),
      ) ||
      voices.find((voice) => /en-GB/i.test(voice.lang)) ||
      voices.find((voice) => /^en/i.test(voice.lang)) ||
      voices[0] ||
      null
    );
  }

  function speechSayBrowser(sentence, token, rate = null, pitch = null) {
    return new Promise((resolve) => {
      const synth = window.speechSynthesis;
      if (!synth) { resolve(); return; }
      const utterance = new SpeechSynthesisUtterance(sentence);
      const preferred = resolveBrowserVoice(state.preferences);
      const tts = state.preferences?.tts || {};
      if (preferred) utterance.voice = preferred;
      utterance.lang = preferred?.lang || "en-GB";
      utterance.rate = clamp(rate ?? tts.rate ?? 0.97, 0.5, 2);
      utterance.pitch = clamp(pitch ?? tts.pitch ?? 0.86, 0.5, 2);
      utterance.volume = 1;

      let settled = false;
      let watchdog = null;
      const finish = () => {
        if (settled) return;
        settled = true;
        clearTimeout(watchdog);
        if (state.utter === utterance) state.utter = null;
        resolve();
      };
      utterance.onstart = () => {
        if (token !== state.speechToken) { finish(); return; }
        state.speaking = true;
        setCoreState("speaking", "SPEAKING", "Voice synthesis active.");
      };
      utterance.onend = finish;
      utterance.onerror = finish;
      // Some engines never fire onend for a cancelled utterance; cap the wait.
      watchdog = setTimeout(finish, 5000 + sentence.length * 90);
      state.utter = utterance;
      if (token !== state.speechToken) { finish(); return; }
      try {
        synth.speak(utterance);
      } catch {
        finish();
      }
    });
  }

  async function speechSayServer(sentence, token) {
    const tts = state.preferences?.tts || {};
    const response = await fetch("/api/tts", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        text: sentence,
        voice: tts.voice || undefined,
        speed: tts.speed || undefined,
      }),
    });
    if (!response.ok) throw new Error(`TTS HTTP ${response.status}`);
    const blob = await response.blob();
    if (token !== state.speechToken) return;
    await speechPlay(blob, token);
  }

  function speechPlay(blob, token) {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(blob);
      const audio = new Audio();
      audio.preload = "auto";
      audio.src = url;
      state.audioEl = audio;
      // Try to put the analyser on the route to the speakers, so the orb moves
      // with the actual voice. If the audio graph is not running the element
      // must be left alone: a tapped element whose context is suspended is
      // silent, which would be worse than an unmetered answer.
      state.meter?.attachElement(audio);
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        URL.revokeObjectURL(url);
        if (state.audioEl === audio) state.audioEl = null;
        resolve();
      };
      audio.onended = finish;
      audio.onerror = finish;
      audio.onplay = () => {
        if (token !== state.speechToken) { try { audio.pause(); } catch { /* gone */ } finish(); return; }
        state.speaking = true;
        setCoreState("speaking", "SPEAKING", "Voice core active.");
      };
      audio.play().catch(finish);
      if (token !== state.speechToken) finish();
    });
  }

  /**
   * Speak a complete message. Used for the greeting and audio-on notice; chat
   * answers go through speechFeed/speechFlush so they start before they end.
   */
  function speak(text) {
    if (!state.audio || !state.voice) return;
    const spoken = cleanForSpeech(text);
    if (!spoken) return;
    speechReset();
    splitSentences(spoken).forEach(speechEnqueue);
  }

  // -----------------------------------------------------------------------
  // Microphone
  // -----------------------------------------------------------------------
  function setListening(listening) {
    state.listening = listening;
    els.micBtn.classList.toggle("is-live", listening);
    els.micBtn.setAttribute("aria-pressed", listening ? "true" : "false");
    if (listening) {
      // The wake listener and the dictation mic cannot share a device cleanly.
      wakeHold();
      setCoreState("listening", "LISTENING", "Speak, sir. I am listening.");
    } else {
      if (!state.busy) setCoreState("idle", "STANDBY", "At your service, sir.");
      wakeRelease();
    }
  }

  function stopMic() {
    if (state.recognition) {
      try { state.recognition.stop(); } catch { /* already stopped */ }
      state.recognition = null;
    }
    if (state.recorder && state.recorder.state !== "inactive") {
      try { state.recorder.stop(); } catch { /* already stopped */ }
    }
    state.meter?.releaseStream();
    clearTimeout(state.micTimer);
    setListening(false);
  }

  async function startBrowserRecognition() {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      toast("Speech recognition is unavailable in this browser. Use the keyboard.", "warn");
      return;
    }
    const recognition = new Recognition();
    recognition.lang = recognitionLanguage();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    state.recognition = recognition;
    recognition.onstart = () => setListening(true);
    recognition.onresult = (event) => {
      let transcript = "";
      let final = false;
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        transcript += event.results[index][0].transcript;
        if (event.results[index].isFinal) final = true;
      }
      els.input.value = transcript.trim();
      autoGrow();
      if (final) {
        const text = transcript.trim();
        stopMic();
        if (text) setTimeout(() => sendMessage(text), 250);
      }
    };
    recognition.onerror = (event) => {
      setListening(false);
      if (event.error !== "no-speech" && event.error !== "aborted") {
        toast(`Audio input error: ${event.error}`, "warn");
      }
    };
    recognition.onend = () => {
      if (state.listening) setListening(false);
      state.recognition = null;
    };
    try {
      recognition.start();
    } catch (error) {
      toast(error.message || "Could not start audio input.", "error");
    }
  }

  async function startServerRecording() {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      toast("Microphone capture is unavailable; using browser recognition.", "warn");
      startBrowserRecognition();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (state.meter) {
        // Non-blocking: a suspended AudioContext must never delay recording.
        state.meter.resume();
        state.meter.attachStream(stream);
      }
      const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find((type) => MediaRecorder.isTypeSupported(type)) || "";
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      state.recorder = recorder;
      state.recorderChunks = [];
      recorder.ondataavailable = (event) => { if (event.data.size) state.recorderChunks.push(event.data); };
      recorder.onstop = async () => {
        state.meter?.releaseStream();
        stream.getTracks().forEach((track) => track.stop());
        clearTimeout(state.micTimer);
        setListening(false);
        const blob = new Blob(state.recorderChunks, { type: recorder.mimeType || "audio/webm" });
        state.recorder = null;
        if (!blob.size) return;
        setCoreState("processing", "PROCESSING", "Transcribing audio via the selected input core.");
        try {
          const form = new FormData();
          form.append("file", blob, "speech.webm");
          const stt = prefsOf().stt || {};
          if (stt.language) form.append("language", stt.language);
          if (stt.model) form.append("model", stt.model);
          const response = await fetch("/api/transcribe", { method: "POST", headers: authHeaders(), body: form });
          const data = await response.json();
          if (!response.ok || !data.ok) throw new Error(data.error || `HTTP ${response.status}`);
          const text = (data.data?.text || "").trim();
          if (!text) { toast("No speech detected.", "warn"); updateCoreIdle(); return; }
          els.input.value = text;
          autoGrow();
          sendMessage(text);
        } catch (error) {
          toast(`Transcription failed: ${error.message}`, "error");
          updateCoreIdle();
        }
      };
      recorder.start();
      setListening(true);
      state.micTimer = setTimeout(() => { if (state.recorder?.state === "recording") state.recorder.stop(); }, 20000);
    } catch (error) {
      toast(`Microphone permission denied: ${error.message}`, "error");
      setListening(false);
    }
  }

  function toggleMic() {
    if (state.listening) { stopMic(); return; }
    // Speaking to JARVIS interrupts it — the reference assistants all behave
    // this way, and it is the only way to get a word in over a long answer.
    if (state.speaking) speechStop();
    if (state.busy) { toast("Stand by until the current response finishes.", "warn"); return; }
    const stt = state.setup?.selection?.stt;
    if (stt && stt !== "browser") startServerRecording();
    else startBrowserRecognition();
  }

  // -----------------------------------------------------------------------
  // Ticker / clock
  // -----------------------------------------------------------------------
  function clockTick() {
    const now = new Date();
    els.clockTime.textContent = now.toLocaleTimeString([], { hour12: false });
    els.clockDate.textContent = now.toLocaleDateString([], { weekday: "short", day: "2-digit", month: "short", year: "numeric" }).toUpperCase();
    state.clockTicks += 1;
    if (state.clockTicks % 7 === 0) rotateTicker();
    updateStatusBar();
  }

  function rotateTicker() {
    const setup = state.setup?.selection || {};
    const messages = [
      `ACTIVE CORE · ${modelLabel("llm", activeFor("llm"))}`,
      `ROUTING · ${state.routingMode.toUpperCase()}`,
      `HOST · ${state.system?.host?.hostname || "UNKNOWN"}`,
      `MEMORY · ${state.memory.length} ENGRAMS`,
      `UPLINK · ${state.connected ? "ENCRYPTED" : "RECONNECTING"}`,
      `VOICE · ${setup.tts || "UNSET"}`,
      "J.A.R.V.I.S. // JUST A RATHER VERY INTELLIGENT SYSTEM",
    ];
    els.ticker.textContent = messages[Math.floor(Math.random() * messages.length)];
  }

  // -----------------------------------------------------------------------
  // Markdown
  // -----------------------------------------------------------------------
  function renderMarkdown(text) {
    if (!text) return "";
    const codeBlocks = [];
    let source = String(text).replace(/\r\n/g, "\n");
    source = source.replace(/```([^\n`]*)\n([\s\S]*?)```/g, (_, language, code) => {
      const index = codeBlocks.length;
      codeBlocks.push(`<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
      return `\u0000CB${index}\u0000`;
    });
    source = escapeHtml(source);
    source = source.replace(/`([^`\n]+)`/g, "<code>$1</code>");
    source = source.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    source = source.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    source = source.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    const lines = source.split("\n");
    let html = "";
    let paragraph = [];
    let list = null;
    const flushParagraph = () => {
      if (paragraph.length) { html += `<p>${paragraph.join("<br>")}</p>`; paragraph = []; }
    };
    const flushList = () => {
      if (list) { html += `</${list}>`; list = null; }
    };
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      if (!line.trim()) { flushParagraph(); flushList(); continue; }
      let match;
      if (/^\s*\|/.test(line)) {
        const block = [];
        let cursor = index;
        while (cursor < lines.length && /^\s*\|/.test(lines[cursor])) { block.push(lines[cursor]); cursor += 1; }
        const separator = block[1] || "";
        if (block.length >= 2 && /^[\s|:-]*-[\s|:-]*$/.test(separator) && separator.includes("|")) {
          flushParagraph(); flushList();
          const cells = (row) => row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
          const head = cells(block[0]);
          let table = `<div class="table-wrap"><table><thead><tr>${head.map((cell) => `<th>${cell}</th>`).join("")}</tr></thead><tbody>`;
          for (const row of block.slice(2)) {
            table += `<tr>${cells(row).map((cell) => `<td>${cell}</td>`).join("")}</tr>`;
          }
          html += `${table}</tbody></table></div>`;
          index = cursor - 1;
          continue;
        }
      }
      if ((match = line.match(/^(#{1,3})\s+(.*)$/))) {
        flushParagraph(); flushList();
        const level = match[1].length;
        html += `<h${level}>${match[2]}</h${level}>`;
        continue;
      }
      if ((match = line.match(/^\s*[-*]\s+(.*)$/))) {
        flushParagraph();
        if (list !== "ul") { flushList(); html += "<ul>"; list = "ul"; }
        html += `<li>${match[1]}</li>`;
        continue;
      }
      if ((match = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
        flushParagraph();
        if (list !== "ol") { flushList(); html += "<ol>"; list = "ol"; }
        html += `<li>${match[1]}</li>`;
        continue;
      }
      if ((match = line.match(/^&gt;\s?(.*)$/))) {
        flushParagraph(); flushList();
        html += `<blockquote>${match[1]}</blockquote>`;
        continue;
      }
      if (/^-{3,}$/.test(line.trim())) { flushParagraph(); flushList(); html += "<hr>"; continue; }
      flushList();
      paragraph.push(line);
    }
    flushParagraph();
    flushList();
    return html.replace(/\u0000CB(\d+)\u0000/g, (_, index) => codeBlocks[Number(index)] || "");
  }

  // -----------------------------------------------------------------------
  // API
  // -----------------------------------------------------------------------
  async function api(path, options = {}) {
    const config = { method: options.method || "GET", headers: authHeaders(options.headers || {}) };
    if (options.body instanceof FormData) {
      config.body = options.body;
      delete config.headers["Content-Type"];
    } else if (options.body !== undefined) {
      config.headers["Content-Type"] = "application/json";
      config.body = JSON.stringify(options.body);
    }
    const response = await fetch(`/api${path}`, config);
    const contentType = response.headers.get("content-type") || "";
    let payload = null;
    if (contentType.includes("application/json")) payload = await response.json();
    if (!response.ok) {
      const message = payload?.error || payload?.detail || `HTTP ${response.status}`;
      throw new Error(message);
    }
    return payload?.data ?? payload;
  }

  // -----------------------------------------------------------------------
  // Core registry: add, probe and remove endpoints
  // -----------------------------------------------------------------------
  const CORE_TYPES = {
    llm: [
      ["openai_compatible", "OpenAI-compatible (Ollama, vLLM, llama.cpp, LM Studio)"],
      ["anthropic", "Anthropic Messages API"],
      ["gemini", "Google Gemini"],
    ],
    stt: [
      ["openai_compatible", "OpenAI-compatible /audio/transcriptions"],
      ["deepgram", "Deepgram"],
      ["faster_whisper", "faster-whisper (local Python)"],
      ["parakeet", "NVIDIA Parakeet (local Python)"],
    ],
    tts: [
      ["openai_compatible", "OpenAI-compatible /audio/speech (Kokoro, openedai…)"],
      ["elevenlabs", "ElevenLabs"],
      ["cartesia", "Cartesia"],
      ["piper", "Piper (local binary)"],
      ["f5_tts", "F5-TTS (local reference audio)"],
    ],
  };

  function coreFormState() {
    const kind = els.coreKind.value;
    const fields = CORE_FIELDS[kind] || CORE_FIELDS.llm;
    const payload = {
      kind,
      id: els.coreId.value.trim(),
      type: els.coreType.value,
      model: els.coreModelInput.value.trim(),
      base_url: LOCAL_TYPES.has(els.coreType.value) ? null : els.coreUrl.value.trim() || null,
      api_key: els.coreKey.value.trim() || null,
      local: LOCAL_TYPES.has(els.coreType.value) ? true : els.coreLocal.checked,
    };
    if (fields.context) {
      payload.context_window = Number(els.coreContext.value) || null;
      payload.max_output_tokens = Number(els.coreMaxOut.value) || null;
      payload.supports_tools = els.coreTools.checked;
      payload.supports_vision = els.coreVision.checked;
      payload.supports_json = els.coreJson.checked;
    }
    if (fields.voice) {
      payload.voice = els.coreVoice.value.trim() || null;
      payload.format = els.coreFormat.value.trim() || null;
      payload.model_path = els.coreModelPath.value.trim() || null;
      payload.reference_audio = els.coreReference.value.trim() || null;
    }
    return payload;
  }

  // Which inputs a kind actually uses. Everything else stays hidden rather
  // than silently sending fields the backend would reject.
  const CORE_FIELDS = {
    llm: { context: true, caps: true, voice: false, secret: true },
    stt: { context: false, caps: false, voice: false, secret: true },
    tts: { context: false, caps: false, voice: true, secret: true },
  };

  function syncCoreFormForKind() {
    const kind = els.coreKind.value;
    const fields = CORE_FIELDS[kind] || CORE_FIELDS.llm;
    const options = CORE_TYPES[kind] || [];
    els.coreType.innerHTML = options
      .map(([value, label]) => `<option value="${value}">${escapeHtml(label)}</option>`)
      .join("");
    // Hidden sections are disabled too: a leftover value there would block
    // native form validation without anything visible to explain why.
    const hide = (element, hidden) => {
      element.closest(".field").classList.toggle("hidden", hidden);
      element.disabled = hidden;
    };
    els.coreCaps.classList.toggle("hidden", !fields.caps);
    els.coreVoiceRow.classList.toggle("hidden", !fields.voice);
    hide(els.coreContext, !fields.context);
    hide(els.coreMaxOut, !fields.context);
    hide(els.coreKey, !fields.secret);
    els.coreTools.disabled = !fields.caps;
    els.coreVision.disabled = !fields.caps;
    els.coreJson.disabled = !fields.caps;
    els.coreVoice.disabled = !fields.voice;
    els.coreFormat.disabled = !fields.voice;
    els.coreModelInput.placeholder =
      kind === "llm" ? "qwen2.5:72b" : kind === "stt" ? "large-v3" : "kokoro";
    syncCoreTypeHints();
  }

  // Some types run in-process and take no URL; the endpoint field is only
  // misleading for those.
  const LOCAL_TYPES = new Set(["faster_whisper", "parakeet", "piper", "f5_tts"]);
  function syncCoreTypeHints() {
    const type = els.coreType.value;
    const local = LOCAL_TYPES.has(type);
    els.coreUrl.closest(".field").classList.toggle("hidden", local);
    els.coreUrl.disabled = local;
    els.coreProbe.disabled = local;
    const note = els.coreProbe.closest(".field").querySelector("em");
    if (note) {
      note.innerHTML = local
        ? "Runs in-process on this machine; no endpoint needed."
        : "OpenAI-compatible <code>/v1</code> or an Ollama port — both are probed.";
    }

    // piper needs an .onnx voice file; f5_tts needs a clip to clone. Showing
    // both at once would suggest either one is optional.
    const needsVoiceFile = type === "piper";
    const needsReference = type === "f5_tts";
    els.coreModelInput.required = !(needsVoiceFile || needsReference);
    if (needsVoiceFile || needsReference) {
      els.coreModelInput.placeholder = needsVoiceFile
        ? "derived from the voice file"
        : "F5TTS_v1_Base";
    } else {
      els.coreModelInput.placeholder =
        els.coreKind.value === "llm" ? "qwen2.5:72b"
          : els.coreKind.value === "stt" ? "large-v3"
          : "kokoro";
    }
    els.coreLocalRow.classList.toggle("hidden", !(needsVoiceFile || needsReference));
    els.coreModelPath.closest(".field").classList.toggle("hidden", !needsVoiceFile);
    els.coreModelPath.required = needsVoiceFile;
    els.coreModelPath.disabled = !needsVoiceFile;
    els.coreReference.closest(".field").classList.toggle("hidden", !needsReference);
    els.coreReference.required = needsReference;
    els.coreReference.disabled = !needsReference;
    // These two voices take their identity from a file, so a named voice would
    // be a lie; the format still applies.
    const fileVoice = needsVoiceFile || needsReference;
    els.coreVoice.closest(".field").classList.toggle("hidden", fileVoice);
    els.coreVoice.disabled = fileVoice || !(CORE_FIELDS[els.coreKind.value] || {}).voice;
    els.coreFormat.disabled = !(CORE_FIELDS[els.coreKind.value] || {}).voice;

    // In-process types can only ever run here, so the box is a statement of
    // fact rather than a choice.
    els.coreLocal.disabled = local;
    const localNote = els.coreLocal.closest(".cap-row").querySelector(".cap-note");
    if (localNote) {
      localNote.textContent = local
        ? "Runs in-process; nothing leaves this machine."
        : "Detected from the endpoint when it is loopback or a private address.";
    }
    if (local) {
      els.coreLocal.checked = true;
    } else if (els.coreLocal.dataset.touched !== "1") {
      els.coreLocal.checked = looksLocal(els.coreUrl.value);
    }
  }

  function fillCoreForm(core) {
    const editable = core.editable || {};
    els.coreKind.value = core.kind;
    syncCoreFormForKind();
    els.coreType.value = core.type || els.coreType.value;
    els.coreId.value = core.id || "";
    els.coreId.readOnly = true;
    els.coreUrl.value = editable.base_url || core.base_url || "";
    els.coreModelInput.value = core.model || "";
    els.coreKey.value = "";
    els.coreKey.placeholder = core.api_key_set ? "stored key — leave blank to keep" : "stored locally, never shown again";
    els.coreContext.value = editable.context_window || "";
    els.coreMaxOut.value = editable.max_output_tokens || "";
    els.coreVoice.value = editable.voice || "";
    els.coreFormat.value = editable.format || "";
    els.coreModelPath.value = editable.model_path || "";
    els.coreReference.value = editable.reference_audio || "";
    els.coreTools.checked = editable.supports_tools !== false;
    els.coreVision.checked = Boolean(editable.supports_vision);
    els.coreJson.checked = editable.supports_json !== false;
    els.coreLocal.checked = Boolean(core.local);
    els.coreLocal.dataset.touched = "1";
    syncCoreTypeHints();
  }

  function openCoreDialog(kind, core = null) {
    state.editingCore = core;
    const editing = Boolean(core);
    els.coreError.classList.add("hidden");
    els.coreError.textContent = "";
    els.coreModelList.innerHTML = "";
    els.coreId.readOnly = editing;
    els.coreKind.disabled = editing;
    els.addCoreKicker.textContent = editing ? "CORE REGISTRY · EDIT" : "CORE REGISTRY";
    els.addCoreTitle.textContent = editing ? "Edit core" : "Add a core";
    const localCore = Boolean(core) && LOCAL_TYPES.has(core.type);
    els.addCoreLede.textContent = editing
      ? localCore
        ? "Change the model, voice file or format. Leave the key blank to keep the stored one."
        : "Change the endpoint or model. Leave the key blank to keep the stored one."
      : "Point J.A.R.V.I.S. at any server you run. Probe the endpoint to list what it actually serves, then save the model you want to use.";
    els.addCoreSaveLabel.textContent = editing ? "UPDATE CORE" : "SAVE CORE";
    els.coreFootNote.innerHTML = editing
      ? "Key left blank keeps the stored value."
      : "Saved to <code>data/runtime/providers.json</code>.";
    if (editing) {
      fillCoreForm(core);
    } else {
      els.addCoreForm.reset();
      delete els.coreLocal.dataset.touched;
      els.coreId.readOnly = false;
      els.coreKind.value = kind || state.activeModelKind || "llm";
      els.coreKey.placeholder = "stored locally, never shown again";
      syncCoreFormForKind();
    }
    els.addCore.classList.remove("hidden");
    document.body.style.overflow = "hidden";
    setTimeout(() => (editing ? els.coreUrl : els.coreId).focus(), 40);
  }

  function closeCoreDialog() {
    els.addCore.classList.add("hidden");
    document.body.style.overflow = "";
    state.editingCore = null;
    els.coreId.readOnly = false;
    els.coreKind.disabled = false;
    els.coreError.classList.add("hidden");
  }

  async function probeCore() {
    const url = els.coreUrl.value.trim();
    if (!url) {
      showCoreError("Enter an endpoint first, for example http://192.168.1.9:8000/v1");
      return;
    }
    els.coreProbe.disabled = true;
    els.coreProbe.textContent = "PROBING…";
    els.coreProbeNote.textContent = "Asking the endpoint what it serves…";
    try {
      const result = await api("/models/discover", {
        method: "POST",
        body: { base_url: url, api_key: els.coreKey.value.trim() || undefined },
      });
      const models = result.models || [];
      els.coreModelList.innerHTML = models
        .map((model) => `<option value="${escapeHtml(model)}"></option>`)
        .join("");
      els.coreProbeNote.innerHTML = models.length
        ? `Found <b>${models.length}</b> model${models.length === 1 ? "" : "s"} via <code>${escapeHtml(result.source)}</code>. Pick one below.`
        : "The endpoint answered but listed no models.";
      // Embedding models cannot answer a chat turn, so prefer something else
      // when the endpoint lists both.
      const chatModels = models.filter((name) => !/embed/i.test(name));
      const suggestion = chatModels[0] || models[0];
      if (suggestion && !els.coreModelInput.value.trim()) {
        els.coreModelInput.value = suggestion;
        els.coreModelInput.focus();
      }
      if (!els.coreId.value.trim() && suggestion) {
        els.coreId.value = slugifyCoreId(suggestCoreId(url, suggestion));
      }
      if (els.coreLocal.dataset.touched !== "1") {
        els.coreLocal.checked = looksLocal(url);
      }
      els.coreError.classList.add("hidden");
    } catch (error) {
      els.coreProbeNote.textContent = "Probe failed.";
      showCoreError(error.message || "Could not reach that endpoint.");
    } finally {
      els.coreProbe.disabled = false;
      els.coreProbe.textContent = "PROBE";
    }
  }

  // Loopback, .local and RFC1918 endpoints almost always mean "my machine".
  function looksLocal(url) {
    let host = "";
    try {
      host = new URL(url).hostname;
    } catch (error) {
      return false;
    }
    if (!host) return false;
    if (["localhost", "::1", "0.0.0.0"].includes(host)) return true;
    if (host.endsWith(".local")) return true;
    const v4 = host.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
    if (!v4) return false;
    const [a, b] = [Number(v4[1]), Number(v4[2])];
    return a === 127 || a === 10 || (a === 192 && b === 168) || (a === 172 && b >= 16 && b <= 31);
  }

  function slugifyCoreId(value) {
    return String(value)
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, "-")
      .replace(/-{2,}/g, "-")
      .replace(/^[-.]+|[-.]+$/g, "")
      .slice(0, 49);
  }

  // Prefer the model family as the id, and only fall back to the host when
  // that name is already taken.
  function suggestCoreId(url, model) {
    const family = model.split(/[:/]/)[0].trim();
    let base = slugifyCoreId(family) || "core";
    if (base.length < 2) base = `core-${base}`;
    const taken = (id) =>
      ["llm", "stt", "tts"].some((kind) =>
        providerItems(kind).some((item) => item.id === id)
      );
    if (!taken(base)) return base;
    let host = "endpoint";
    try {
      host = new URL(url).hostname;
    } catch (error) {
      host = String(url);
    }
    const suffix = slugifyCoreId(host.replace(/\./g, "-")) || "endpoint";
    return slugifyCoreId(`${base}-${suffix}`);
  }

  function showCoreError(message) {
    els.coreError.textContent = message;
    els.coreError.classList.remove("hidden");
  }

  function corePayload() {
    const payload = coreFormState();
    // Piper takes its voice from the .onnx file; F5-TTS has a default checkpoint.
    if (payload.type === "piper" && !payload.model && payload.model_path) {
      payload.model = payload.model_path.split("/").pop().replace(/\.onnx$/, "");
    }
    if (payload.type === "f5_tts" && !payload.model) {
      payload.model = "F5TTS_v1_Base";
    }
    for (const key of Object.keys(payload)) {
      if (payload[key] === null || payload[key] === undefined || payload[key] === "") delete payload[key];
    }
    return payload;
  }

  async function saveCore(event) {
    if (event) event.preventDefault();
    const payload = corePayload();
    if (!payload.id) return showCoreError("An id is required, for example my-vllm.");
    if (!payload.model) return showCoreError("A model name is required.");
    if (!LOCAL_TYPES.has(payload.type) && !payload.base_url) {
      return showCoreError("An endpoint URL is required for this core type.");
    }
    // Captured before the dialog closes and clears the edit target.
    const editing = state.editingCore;
    const action = editing ? "updated" : "added";
    els.addCoreSave.disabled = true;
    els.addCoreSaveLabel.textContent = "SAVING…";
    try {
      const path = editing
        ? `/models/custom/${editing.kind}/${editing.id}`
        : "/models/custom";
      const result = await api(path, { method: editing ? "PUT" : "POST", body: payload });
      closeCoreDialog();
      if (result.setup) state.setup = result.setup;
      state.models = null;
      toast(`${payload.kind.toUpperCase()} core “${payload.id}” ${action}.`, "ok");
      await refreshModels();
      syncSetupSelection();
      if (!els.setup.classList.contains("hidden")) renderSetup();
    } catch (error) {
      showCoreError(error.message || "Could not save that core.");
    } finally {
      els.addCoreSave.disabled = false;
      els.addCoreSaveLabel.textContent = state.editingCore ? "UPDATE CORE" : "SAVE CORE";
    }
  }

  async function removeCore(kind, providerId, label) {
    const confirmed = window.confirm(
      `Remove the ${kind.toUpperCase()} core “${label}” from JARVIS?\n\nThis deletes it from data/runtime/providers.json.`
    );
    if (!confirmed) return;
    try {
      const result = await api(`/models/custom/${kind}/${providerId}`, { method: "DELETE" });
      if (result.setup) state.setup = result.setup;
      state.models = null;
      toast(`${kind.toUpperCase()} core “${label}” removed.`, "ok");
      await refreshModels();
      syncSetupSelection();
      if (state.setup && !state.setup.completed) showSetup();
      else renderSetup();
    } catch (error) {
      toast(error.message || "Could not remove that core.", "error");
    }
  }

  // The server may have dropped a selection when its core disappeared.
  function syncSetupSelection() {
    const selection = state.setup?.selection;
    if (selection) {
      state.setupSelection = {
        llm: selection.llm || null,
        stt: selection.stt ?? null,
        tts: selection.tts || null,
      };
      return;
    }
    // Mid-setup: keep the user's picks unless the core behind one vanished.
    for (const kind of ["llm", "stt", "tts"]) {
      const id = state.setupSelection[kind];
      if (id && !providerItems(kind).some((item) => item.id === id)) {
        state.setupSelection[kind] = null;
      }
    }
  }

  // -----------------------------------------------------------------------
  // Toasts / rendering helpers
  // -----------------------------------------------------------------------
  function toast(message, kind = "info") {
    const item = document.createElement("div");
    item.className = `toast ${kind}`;
    item.textContent = message;
    els.toasts.appendChild(item);
    setTimeout(() => {
      item.style.opacity = "0";
      item.style.transform = "translateX(8px)";
      setTimeout(() => item.remove(), 250);
    }, 3800);
  }

  function renderAll() {
    renderRouting();
    renderModels();
    renderChat();
    renderMemory();
    renderEvents();
    renderConversations();
    if (state.system) renderTelemetry(state.system);
    updateComposer();
    updateAudioButton();
    updateStatusBar();
    els.chipHost.textContent = state.system?.host?.hostname || "HOST";
    if (state.setup && !state.setup.completed) renderSetup();
    else updateCoreIdle();
  }

  // -----------------------------------------------------------------------
  // Input helpers
  // -----------------------------------------------------------------------
  function autoGrow() {
    els.input.style.height = "auto";
    els.input.style.height = `${Math.min(150, els.input.scrollHeight)}px`;
  }

  function bindEvents() {
    $("#bootSkip").addEventListener("click", finishBoot);
    $("#setupEngage").addEventListener("click", saveSetup);
    $("#setupCancel").addEventListener("click", closeSetup);
    $("#openSetup").addEventListener("click", openSetup);
    $("#audioToggle").addEventListener("click", () => setAudio(!state.audio));
    els.wakeToggle?.addEventListener("click", () => setWake(!state.wakeEnabled));
    $("#fullscreen").addEventListener("click", toggleFullscreen);
    $("#sendBtn").addEventListener("click", () => sendMessage(els.input.value));
    $("#micBtn").addEventListener("click", toggleMic);
    $("#newThread").addEventListener("click", newThread);
    $("#archiveNew").addEventListener("click", newThread);
    $("#refreshHealth").addEventListener("click", refreshHealth);
    $("#clearLogView").addEventListener("click", () => { state.events = []; els.eventList.innerHTML = ""; });
    $("#openArchive").addEventListener("click", () => { setTab("archive"); openDrawer("right"); });
    $("#mobileVitals").addEventListener("click", () => openDrawer("left"));
    $("#mobileMatrix").addEventListener("click", () => openDrawer("right"));
    els.drawerScrim.addEventListener("click", closeDrawers);
    $$("[data-close-panel]").forEach((button) => button.addEventListener("click", closeDrawers));

    els.input.addEventListener("input", autoGrow);
    els.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage(els.input.value);
      }
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        if (!els.addCore.classList.contains("hidden")) {
          closeCoreDialog();
          return;
        }
        if (tune.open) {
          closeTune();
          return;
        }
        if (state.listening) stopMic();
        closeDrawers();
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        els.input.focus();
      }
      // Never steal a keystroke the user is typing into a field.
      const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName || "")
        || document.activeElement?.isContentEditable;
      if (event.key === "/" && !typing && !event.metaKey && !event.ctrlKey) {
        event.preventDefault();
        els.input.focus();
      }
    });

    els.setup.addEventListener("click", (event) => {
      const card = event.target.closest(".core-card");
      if (card && !card.disabled) selectCore(card.dataset.kind, card.dataset.id);
      const filter = event.target.closest(".filter");
      if (filter) {
        const row = filter.closest(".filter-row");
        const kind = row?.dataset.filterFor;
        if (kind) {
          state.setupFilter[kind] = filter.dataset.filter;
          row.querySelectorAll(".filter").forEach((button) => button.classList.toggle("is-active", button === filter));
          renderSetup();
        }
      }
    });

    els.routingSeg.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-mode]");
      if (button) setRoutingMode(button.dataset.mode);
    });

    els.modelTabs.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-kind]");
      if (!button) return;
      state.activeModelKind = button.dataset.kind;
      renderModels();
    });

    els.modelList.addEventListener("click", (event) => {
      const card = event.target.closest(".model-card");
      if (card && !card.disabled) switchModel(card.dataset.kind, card.dataset.id);
    });

    $(".tabs").addEventListener("click", (event) => {
      const button = event.target.closest(".tab");
      if (button) setTab(button.dataset.tab);
    });

    els.memoryForm.addEventListener("submit", (event) => {
      event.preventDefault();
      addMemory(els.memoryInput.value);
    });

    $("#suggestions").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-send]");
      if (button) sendMessage(button.dataset.send);
    });

    els.conversationTitle.addEventListener("dblclick", () => {
      if (state.activeConversation) setTab("archive");
    });

    // -- core registry ---------------------------------------------------
    // Default to the kind the user still has to choose.
    els.setupAddCore.addEventListener("click", () => {
      const missing = ["llm", "tts", "stt"].find((kind) => !state.setupSelection[kind]);
      openCoreDialog(missing || "llm");
    });
    els.addCoreFromMatrix.addEventListener("click", () => openCoreDialog(state.activeModelKind));
    els.addCoreClose.addEventListener("click", closeCoreDialog);
    els.addCoreCancel.addEventListener("click", closeCoreDialog);
    els.addCore.addEventListener("click", (event) => {
      if (event.target === els.addCore) closeCoreDialog();
    });
    els.addCoreForm.addEventListener("submit", saveCore);
    els.coreProbe.addEventListener("click", probeCore);
    els.coreKind.addEventListener("change", syncCoreFormForKind);
    els.coreType.addEventListener("change", syncCoreTypeHints);
    els.coreLocal.addEventListener("change", () => {
      els.coreLocal.dataset.touched = "1";
    });
    els.coreUrl.addEventListener("input", () => {
      if (els.coreLocal.dataset.touched !== "1") {
        els.coreLocal.checked = looksLocal(els.coreUrl.value);
      }
    });
    els.coreUrl.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        probeCore();
      }
    });

    document.addEventListener("click", (event) => {
      const edit = event.target.closest("[data-edit-core]");
      if (edit) {
        event.preventDefault();
        event.stopPropagation();
        const kind = edit.dataset.kind;
        const core = (state.setup?.custom || []).find(
          (item) => item.kind === kind && item.id === edit.dataset.editCore
        );
        if (core) {
          openCoreDialog(kind, core);
        } else {
          toast("That core is no longer in the registry. Refreshing.", "warn");
          refreshModels();
        }
        return;
      }
      const remove = event.target.closest("[data-remove-core]");
      if (remove) {
        event.preventDefault();
        event.stopPropagation();
        removeCore(remove.dataset.kind, remove.dataset.removeCore, remove.dataset.removeCore);
      }
    });

    // -- tuning ----------------------------------------------------------
    els.tuneFromMatrix.addEventListener("click", () => openTune(state.activeModelKind === "llm" ? "llm" : state.activeModelKind));
    els.openTune.addEventListener("click", () => openTune("llm"));
    $("#statCore").addEventListener("click", () => openTune("llm"));
    els.composerModel.addEventListener("click", () => openTune("llm"));
    $("#statVoice").addEventListener("click", () => openTune("tts"));
    $("#tuneClose").addEventListener("click", closeTune);
    $("#tuneCancel").addEventListener("click", closeTune);
    sel.tune.addEventListener("keydown", trapTuneFocus);
    sel.tune.addEventListener("click", (event) => {
      if (event.target === sel.tune) closeTune();
    });
    sel.tuneTabs.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-tune-tab]");
      if (!button) return;
      // Keep what the user has typed so far before switching tabs.
      if (tune.open) tuneReadForm();
      tune.tab = button.dataset.tuneTab;
      renderTuneTab();
      if (tune.tab === "llm" && selectedCore("llm")) {
        loadCatalog("llm").then(() => { if (tune.open) renderTune(); });
      }
      if (tune.tab === "tts" && selectedCore("tts") && !tuneIsBrowserVoice()) {
        loadCatalog("tts").then(() => { if (tune.open) renderTune(); });
      }
    });
    sel.tuneLlmReload.addEventListener("click", async () => {
      sel.tuneLlmReload.disabled = true;
      tune.catalog.llm = null;
      await loadCatalog("llm", { force: true });
      renderTune();
      sel.tuneLlmReload.disabled = false;
    });
    sel.tuneVoiceList.addEventListener("click", (event) => {
      const chip = event.target.closest("[data-voice]");
      if (!chip) return;
      // Picking a voice auditions it — that is the whole point of a list.
      tune.voiceChoice = chip.dataset.voice || null;
      sel.tuneVoiceFree.value = "";
      const browser = tuneIsBrowserVoice();
      tune.draft = tune.draft || {};
      tune.draft.tts = {
        ...(tune.draft.tts || {}),
        voice: browser ? (tune.draft.tts?.voice ?? null) : tune.voiceChoice,
        browser_voice: browser ? tune.voiceChoice : (tune.draft.tts?.browser_voice ?? null),
      };
      for (const node of $$("[data-voice]", sel.tuneVoiceList)) {
        const active = node === chip;
        node.classList.toggle("is-active", active);
        node.setAttribute("aria-selected", active ? "true" : "false");
      }
      if (browser) testVoice();
    });
    sel.tuneVoiceFree.addEventListener("input", () => {
      const typed = sel.tuneVoiceFree.value.trim();
      if (!typed) return;
      tune.voiceChoice = typed;
      for (const node of $$("[data-voice]", sel.tuneVoiceList)) {
        node.classList.remove("is-active");
        node.setAttribute("aria-selected", "false");
      }
    });
    sel.tuneLlmModel.addEventListener("change", () => {
      if (sel.tuneLlmModel.value) sel.tuneLlmModelFree.value = "";
    });
    sel.tuneLlmModelFree.addEventListener("input", () => {
      if (sel.tuneLlmModelFree.value.trim()) sel.tuneLlmModel.value = "";
    });
    for (const [input, output, format] of [
      [sel.tuneTemp, sel.tuneTempOut, (v) => v.toFixed(2)],
      [sel.tuneRate, sel.tuneRateOut, (v) => `${v.toFixed(2)}×`],
      [sel.tunePitch, sel.tunePitchOut, (v) => v.toFixed(2)],
      [sel.tuneSpeed, sel.tuneSpeedOut, (v) => `${v.toFixed(2)}×`],
    ]) {
      const update = () => { output.textContent = format(Number(input.value)); };
      input.addEventListener("input", update);
      update();
    }
    sel.tuneTestVoice.addEventListener("click", testVoice);
    sel.tuneSave.addEventListener("click", saveTune);
    sel.tuneReset.addEventListener("click", resetTune);
    // Browser engines load their voice list asynchronously.
    window.speechSynthesis?.addEventListener?.("voiceschanged", () => {
      if (!tune.open) return;
      tuneWriteForm(prefsOf());
      renderTune();
    });

    document.addEventListener("pointerdown", () => {
      if (state.audio) initAudioContext();
    }, { once: true });
  }

  function init() {
    setupOrb();
    updateAudioButton();
    updateWakeButton();
    bindEvents();
    clockTick();
    setInterval(clockTick, 1000);
    setInterval(() => { if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({ type: "ping" })); }, 25000);
    if (new URLSearchParams(location.search).has("skipboot")) {
      // Development shortcut: jump straight to the live interface.
      state.bootDone = true;
      els.boot.classList.add("is-done", "hidden");
    } else {
      runBoot();
    }
    connect();
    window.jarvis = {
      state,
      sendMessage,
      api,
      // Diagnostics: drive the interface from the console or a test.
      setCoreState,
      speech: { feed: speechFeed, flush: speechFlush, reset: speechReset, stop: speechStop, split: splitSentences },
      orb: { ready: () => Boolean(state.orb) },
    };
  }

  init();
})();
