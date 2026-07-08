const GALLERY_CHANNEL = "myth-gallery";
const DESKTOP_QUERY = "(min-width: 900px)";
const WEBGPU_BROWSER_SUPPORT_URL = "https://caniuse.com/webgpu";
const WEBGPU_IMPLEMENTATION_STATUS_URL = "https://github.com/gpuweb/gpuweb/wiki/Implementation-Status";
const page = document.body.dataset.page;

if (page === "gallery") {
    initGallery().catch(console.error);
} else if (page === "viewer") {
    initViewer().catch(console.error);
}

/* =========================================
   Gallery (index.html)
   ========================================= */

async function initGallery() {
    const manifest = await fetchManifest("./examples.json");
    const entries = manifest.flatMap((group) =>
        group.items.map((item) => ({ ...item, category: group.category })),
    );

    const desktopMedia = window.matchMedia(DESKTOP_QUERY);
    const sidebar = document.getElementById("sidebar");
    const sidebarToggle = document.getElementById("sidebar-toggle");
    const sidebarClose = document.getElementById("sidebar-close");
    const sidebarScrim = document.getElementById("sidebar-scrim");
    const navMenu = document.getElementById("nav-menu");
    const frame = document.getElementById("viewer-frame");
    const nativeOverlay = document.getElementById("native-overlay");
    const nativeBadge = nativeOverlay.querySelector(".native-badge");
    const nativeTitle = document.getElementById("native-title");
    const nativeCopy = document.getElementById("native-copy");
    const nativeActions = document.getElementById("native-actions");
    const actionBar = document.getElementById("action-bar");
    const btnSource = document.getElementById("btn-source");
    const btnStandalone = document.getElementById("btn-standalone");
    const hintPanel = document.getElementById("hint-panel");
    const hintLines = document.getElementById("hint-lines");
    const runtimeStatusBar = document.getElementById("runtime-status-bar");
    let selectionVersion = 0;
    let webGpuSupportPromise = null;

    // Render navigation
    navMenu.innerHTML = manifest
        .map((group) => {
            const items = group.items
                .map((item) => {
                    const note = item.note
                        ? `<span class="example-item-note">${escapeHtml(item.note)}</span>`
                        : "";
                    return `<button class="example-item" type="button" data-id="${escapeHtml(item.id)}">${escapeHtml(item.name)}${note}</button>`;
                })
                .join("");
            return `<div class="category-group"><div class="category-title">${escapeHtml(group.category)}</div>${items}</div>`;
        })
        .join("");

    // Sidebar toggling
    function setSidebarOpen(open) {
        sidebar.classList.toggle("is-open", open);
        document.body.classList.toggle("sidebar-open", open);
        sidebarScrim.hidden = !(open && !desktopMedia.matches);
    }

    setSidebarOpen(desktopMedia.matches);

    sidebarToggle.addEventListener("click", () => setSidebarOpen(true));
    sidebarClose.addEventListener("click", () => setSidebarOpen(false));
    sidebarScrim.addEventListener("click", () => setSidebarOpen(false));
    desktopMedia.addEventListener("change", () => setSidebarOpen(desktopMedia.matches));

    window.addEventListener("keydown", (event) => {
        if (event.key === "Escape") setSidebarOpen(false);
    });

    window.addEventListener("message", (event) => {
        const data = event.data;
        if (data?.channel !== GALLERY_CHANNEL) {
            return;
        }

        if (data.state === "status_update") {
            if (typeof data.text === "string" && data.text) {
                showRuntimeStatus(data.text);
            } else {
                hideRuntimeStatus();
            }
            return;
        }

        if (data.state === "mounted" || data.state === "error") {
            hideRuntimeStatus();
        }

        if (data.state !== "ready") {
            return;
        }

        const entry = entries.find((item) => item.id === data.exampleId);
        if (!entry?.instructions) {
            hideHintPanel();
            return;
        }

        showHintPanel(entry.instructions);
    });

    // Navigation click
    navMenu.addEventListener("click", (event) => {
        const button = event.target.closest("[data-id]");
        if (!button) return;
        const entry = entries.find((e) => e.id === button.dataset.id);
        if (entry) {
            selectEntry(entry, true);
            if (!desktopMedia.matches) setSidebarOpen(false);
        }
    });

    // URL-driven state
    window.addEventListener("popstate", () => {
        const entry = entryFromUrl(entries) ?? (entries.find(e => e.id === "showcase") || entries[0]);
        if (entry) selectEntry(entry, false);
    });

    // Initial load
    const initial = entryFromUrl(entries) ?? (entries.find(e => e.id === "showcase") || entries[0]);
    if (initial) selectEntry(initial, false);

    function showHintPanel(message) {
        if (!hintPanel || !hintLines || !message) {
            return;
        }

        const lines = normalizeHintLines(message);
        if (lines.length === 0) {
            hideHintPanel();
            return;
        }

        hintLines.innerHTML = lines
            .map((line) => `<div class="hint-panel-line">${escapeHtml(line)}</div>`)
            .join("");
        hintPanel.classList.remove("hidden");
        requestAnimationFrame(() => {
            hintPanel.classList.add("show");
        });
    }

    function hideHintPanel() {
        if (!hintPanel || !hintLines) {
            return;
        }

        hintPanel.classList.remove("show");
        hintPanel.classList.add("hidden");
        hintLines.innerHTML = "";
    }

    function showRuntimeStatus(message) {
        if (!runtimeStatusBar) {
            return;
        }

        runtimeStatusBar.textContent = message;
        runtimeStatusBar.classList.remove("hidden");
    }

    function hideRuntimeStatus() {
        if (!runtimeStatusBar) {
            return;
        }

        runtimeStatusBar.classList.add("hidden");
        runtimeStatusBar.textContent = "";
    }

    function selectEntry(entry, pushHistory) {
        const currentSelection = ++selectionVersion;

        // Update active state in nav
        navMenu.querySelectorAll(".example-item").forEach((el) => {
            el.classList.toggle("is-active", el.dataset.id === entry.id);
        });

        hideHintPanel();
        hideRuntimeStatus();

        // Update URL via History API
        const url = new URL(window.location.href);
        url.searchParams.set("example", entry.id);
        const currentId = new URLSearchParams(window.location.search).get("example");
        if (pushHistory && currentId !== entry.id) {
            history.pushState({ example: entry.id }, "", url);
        } else {
            history.replaceState({ example: entry.id }, "", url);
        }

        // Handle native-only entries
        if (!entry.web_supported) {
            nativeBadge.textContent = "NATIVE ONLY";
            nativeOverlay.classList.remove("hidden");
            nativeTitle.textContent = entry.name;
            nativeCopy.textContent =
                entry.note || "This example is not supported on the web. Please run the native application to view it.";
            frame.src = "about:blank";

            actionBar.classList.add("hidden");
            hideNativeActions();
            hideHintPanel();
            return;
        }

        if (entry.source_url) {
            btnSource.href = entry.source_url;
            btnSource.classList.remove("hidden");
        } else {
            btnSource.classList.add("hidden");
        }

        // Load in iframe
        const targetUrl =
            entry.type === "standalone"
                ? entry.url
                : `./viewer.html?example=${encodeURIComponent(entry.id)}`;

        btnStandalone.href = targetUrl;
        frame.src = "about:blank";

        showRuntimeStatus("Checking WebGPU support...");
        getWebGpuSupport().then((support) => {
            if (currentSelection !== selectionVersion) {
                return;
            }

            if (!support.ok) {
                hideRuntimeStatus();
                showWebGpuRequiredOverlay(entry, support);
                return;
            }

            nativeOverlay.classList.add("hidden");
            actionBar.classList.remove("hidden");
            hideRuntimeStatus();

            requestAnimationFrame(() => {
                if (currentSelection === selectionVersion) {
                    frame.src = targetUrl;
                }
            });
        });
    }

    function getWebGpuSupport() {
        webGpuSupportPromise ??= checkWebGpuSupport();
        return webGpuSupportPromise;
    }

    function showWebGpuRequiredOverlay(entry, support) {
        nativeBadge.textContent = "WEBGPU REQUIRED";
        nativeOverlay.classList.remove("hidden");
        nativeTitle.textContent = entry.name;
        nativeCopy.textContent =
            `This gallery example needs WebGPU, but this browser cannot provide it. ${webGpuReasonText(support.reason)} ` +
            "Try a browser with WebGPU support, enable hardware acceleration, and serve the gallery from HTTPS or localhost.";
        nativeActions.innerHTML = renderActionLinks(webGpuSupportActions(), "native-action");
        nativeActions.classList.remove("hidden");
        frame.src = "about:blank";
        actionBar.classList.add("hidden");
        hideHintPanel();
    }

    function hideNativeActions() {
        nativeActions.classList.add("hidden");
        nativeActions.innerHTML = "";
    }
}

/* =========================================
   Viewer (viewer.html)
   ========================================= */

async function initViewer() {
    const params = new URLSearchParams(window.location.search);
    const exampleId = params.get("example");
    const bootStart = performance.now();

    const overlay = document.getElementById("loading-overlay");
    const statusEl = document.getElementById("loading-status");
    const progressBar = document.getElementById("loading-progress-bar");
    const elapsedEl = document.getElementById("loading-elapsed");
    const messagePanel = document.getElementById("loading-message");
    const messageTitle = document.getElementById("loading-message-title");
    const messageBody = document.getElementById("loading-message-body");
    const messageList = document.getElementById("loading-message-list");
    const messageActions = document.getElementById("loading-actions");

    let displayedProgress = 0;
    let readyHandled = false;
    let activeEntry = null;
    let fadeOutTimeout = 0;

    const elapsedTimer = setInterval(() => {
        elapsedEl.textContent = formatDuration(performance.now() - bootStart);
    }, 100);

    const handleLoadingProgress = (event) => {
        if (readyHandled) {
            return;
        }

        const detail = event.detail ?? {};
        const message = typeof detail.message === "string" && detail.message
            ? detail.message
            : "Fetching assets...";
        const percentage = Number.isFinite(detail.percentage) ? detail.percentage : 0;
        updateProgress(message, mapAssetProgress(percentage));
    };

    const handleSceneReady = () => {
        if (readyHandled || !activeEntry) {
            return;
        }

        readyHandled = true;
        window.removeEventListener("myth-loading-progress", handleLoadingProgress);

        const bootMs = performance.now() - bootStart;
        clearInterval(elapsedTimer);
        elapsedEl.textContent = formatDuration(bootMs);

        updateProgress("Ready", 100, { force: true });
        sendToGallery({
            state: "ready",
            label: "Scene Ready",
            exampleId: activeEntry.id,
            bootMs,
            route: `?example=${activeEntry.id}`,
        });

        fadeOutOverlay();
    };

    function fadeOutOverlay() {
        window.clearTimeout(fadeOutTimeout);
        fadeOutTimeout = window.setTimeout(() => {
            overlay.classList.add("fade-out");
        }, 120);
    }

    const handleRuntimeStatus = (event) => {
        const detail = event.detail ?? {};
        const text = typeof detail.text === "string" ? detail.text.trim() : "";
        if (!text) {
            return;
        }

        sendToGallery({
            state: "status_update",
            text,
            exampleId: activeEntry?.id ?? exampleId,
        });
    };

    window.addEventListener("myth-loading-progress", handleLoadingProgress);
    window.addEventListener("myth-scene-ready", handleSceneReady, { once: true });
    window.addEventListener("myth-status-update", handleRuntimeStatus);

    function detachRuntimeListeners() {
        window.removeEventListener("myth-loading-progress", handleLoadingProgress);
        window.removeEventListener("myth-scene-ready", handleSceneReady);
        window.removeEventListener("myth-status-update", handleRuntimeStatus);
    }

    updateProgress("Resolving manifest...", 5);
    sendToGallery({
        state: "mounted",
        label: "Viewer Shell",
        exampleId,
        route: exampleId ? `?example=${exampleId}` : "?example=unknown",
    });

    const manifest = await fetchManifest("./examples.json");
    const entries = manifest.flatMap((group) =>
        group.items.map((item) => ({ ...item, category: group.category })),
    );
    const entry = entries.find((e) => e.id === exampleId);
    activeEntry = entry ?? null;

    if (!entry || !entry.web_supported || entry.type !== "iframe") {
        detachRuntimeListeners();
        clearInterval(elapsedTimer);
        updateProgress("Entry not available", 100, { force: true });
        showBlockingMessage({
            tone: "warning",
            title: "Entry not available",
            body: "This gallery entry is missing from the manifest or cannot run inside the shared web viewer.",
            tips: ["Choose another gallery item or open the native example from the source project."],
        });
        sendToGallery({
            state: "error",
            label: "Unavailable",
            detail: "传入的示例标识不在当前清单中或不支持网页运行。",
            exampleId,
        });
        return;
    }

    updateProgress("Checking WebGPU support...", 20);
    const webGpuSupport = await checkWebGpuSupport();
    if (!webGpuSupport.ok) {
        detachRuntimeListeners();
        clearInterval(elapsedTimer);
        elapsedEl.textContent = formatDuration(performance.now() - bootStart);
        updateProgress("WebGPU unavailable", 100, { force: true });
        showWebGpuUnavailableMessage(webGpuSupport);
        sendToGallery({
            state: "error",
            label: "WebGPU Unavailable",
            detail: webGpuSupport.detail,
            exampleId: entry.id,
            route: `?example=${entry.id}`,
        });
        return;
    }

    updateProgress(`Loading wasm/${entry.id}.js`, 30);
    sendToGallery({
        state: "booting",
        label: "Loading Module",
        exampleId: entry.id,
        route: `?example=${entry.id}`,
    });

    try {
        const module = await import(`./wasm/${entry.id}.js`);

        updateProgress("Booting runtime...", 70);
        sendToGallery({
            state: "booting",
            label: "Booting Runtime",
            exampleId: entry.id,
        });

        await module.default();

        if (!readyHandled) {
            updateProgress("Runtime ready, waiting for assets...", 80);
            sendToGallery({
                state: "runtime-ready",
                label: "Runtime Ready",
                exampleId: entry.id,
                route: `?example=${entry.id}`,
            });
        }
    } catch (error) {
        detachRuntimeListeners();
        clearInterval(elapsedTimer);
        elapsedEl.textContent = formatDuration(performance.now() - bootStart);
        const likelyWebGpuError = isLikelyWebGpuError(error);
        updateProgress(likelyWebGpuError ? "WebGPU unavailable" : "Boot failed", 100, { force: true });
        if (likelyWebGpuError) {
            showWebGpuUnavailableMessage({
                ok: false,
                reason: "runtime-error",
                detail: error instanceof Error ? error.message : String(error),
            });
        } else {
            showBlockingMessage({
                tone: "error",
                title: "Example failed to start",
                body: "The runtime stopped before the scene could start.",
                tips: [error instanceof Error ? error.message : String(error)],
            });
        }
        sendToGallery({
            state: "error",
            label: likelyWebGpuError ? "WebGPU Unavailable" : "Boot Failed",
            detail: error instanceof Error ? error.message : String(error),
            exampleId: entry.id,
            route: `?example=${entry.id}`,
        });
        throw error;
    }

    function mapAssetProgress(assetPercentage) {
        return 80 + clampPercentage(assetPercentage) * 0.2;
    }

    function updateProgress(status, pct, options = {}) {
        const { force = false } = options;
        statusEl.textContent = status;
        const clamped = clampPercentage(pct);
        displayedProgress = force ? clamped : Math.max(displayedProgress, clamped);
        progressBar.style.width = `${displayedProgress}%`;
    }

    function showWebGpuUnavailableMessage(support) {
        const reason = webGpuReasonText(support.reason);
        showBlockingMessage({
            tone: "warning",
            title: "WebGPU is not available",
            body: `This example needs WebGPU, but this browser could not provide it. ${reason}`,
            tips: [
                "Open the gallery in a browser with WebGPU support.",
                "Enable hardware acceleration, then restart the browser.",
                "Use HTTPS or localhost when serving the gallery.",
            ],
            actions: webGpuSupportActions(),
        });
    }

    function showBlockingMessage({ tone = "warning", title, body, tips = [], actions = [] }) {
        overlay.classList.remove("fade-out", "is-warning", "is-error");
        overlay.classList.add("is-blocked", tone === "error" ? "is-error" : "is-warning");

        messageTitle.textContent = title;
        messageBody.textContent = body;
        messageList.innerHTML = tips
            .filter((tip) => typeof tip === "string" && tip.trim())
            .map((tip) => `<li>${escapeHtml(tip)}</li>`)
            .join("");
        messageList.classList.toggle("hidden", messageList.children.length === 0);

        messageActions.innerHTML = renderActionLinks(actions, "loading-action");
        messageActions.classList.toggle("hidden", actions.length === 0);
        messagePanel.classList.remove("hidden");
    }
}

/* =========================================
   Shared utilities
   ========================================= */

function sendToGallery(payload) {
    if (window.parent === window) return;
    window.parent.postMessage({ channel: GALLERY_CHANNEL, ...payload }, "*");
}

async function fetchManifest(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
        throw new Error(`Failed to load manifest: ${response.status}`);
    }
    return response.json();
}

async function checkWebGpuSupport() {
    if (isWebGpuSupportForcedOff()) {
        return {
            ok: false,
            reason: "forced-off",
            detail: "WebGPU support was disabled by the gallery test query parameter.",
        };
    }

    if (!("gpu" in navigator) || !navigator.gpu) {
        return {
            ok: false,
            reason: "missing-api",
            detail: "navigator.gpu is not available.",
        };
    }

    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            return {
                ok: false,
                reason: "no-adapter",
                detail: "navigator.gpu.requestAdapter() returned no adapter.",
            };
        }
    } catch (error) {
        return {
            ok: false,
            reason: "adapter-error",
            detail: error instanceof Error ? error.message : String(error),
        };
    }

    return { ok: true };
}

function webGpuReasonText(reason) {
    switch (reason) {
        case "forced-off":
            return "The gallery is running with a WebGPU test override.";
        case "missing-api":
            return "The WebGPU API is missing.";
        case "no-adapter":
            return "No compatible GPU adapter was found.";
        case "adapter-error":
            return "The browser rejected the adapter request.";
        default:
            return "The runtime reported a WebGPU startup error.";
    }
}

function webGpuSupportActions() {
    return [
        {
            label: "Browser support",
            href: WEBGPU_BROWSER_SUPPORT_URL,
        },
        {
            label: "Implementation status",
            href: WEBGPU_IMPLEMENTATION_STATUS_URL,
        },
    ];
}

function renderActionLinks(actions, className) {
    return actions
        .map((action) => {
            const label = escapeHtml(action.label);
            const href = escapeHtml(action.href);
            return `<a class="${className}" href="${href}" target="_blank" rel="noreferrer">${label}</a>`;
        })
        .join("");
}

function isWebGpuSupportForcedOff() {
    const params = new URLSearchParams(window.location.search);
    return (
        params.has("force_no_webgpu") ||
        params.has("mock_no_webgpu") ||
        params.get("webgpu") === "0"
    );
}

function isLikelyWebGpuError(error) {
    const message = error instanceof Error ? error.message : String(error);
    return /webgpu|gpu|adapter|requestadapter|requestdevice|wgpu/i.test(message);
}

function entryFromUrl(entries) {
    const id = new URLSearchParams(window.location.search).get("example");
    return id ? entries.find((e) => e.id === id) || null : null;
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function formatDuration(ms) {
    if (!Number.isFinite(ms) || ms < 0) return "-- ms";
    return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

function normalizeHintLines(message) {
    const raw = String(message ?? "");
    const parts = raw.includes("\n") ? raw.split(/\r?\n/) : raw.split(/\s*;\s*/);
    return parts.map((line) => line.trim()).filter(Boolean);
}

function clampPercentage(value) {
    return Math.max(0, Math.min(100, value));
}
