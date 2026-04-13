const GALLERY_CHANNEL = "myth-gallery";
const DESKTOP_QUERY = "(min-width: 900px)";
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
    const nativeTitle = document.getElementById("native-title");
    const nativeCopy = document.getElementById("native-copy");

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
        const entry = entryFromUrl(entries) ?? entries[0];
        if (entry) selectEntry(entry, false);
    });

    // Initial load
    const initial = entryFromUrl(entries) ?? entries[0];
    if (initial) selectEntry(initial, false);

    function selectEntry(entry, pushHistory) {
        // Update active state in nav
        navMenu.querySelectorAll(".example-item").forEach((el) => {
            el.classList.toggle("is-active", el.dataset.id === entry.id);
        });

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
            nativeOverlay.classList.remove("hidden");
            nativeTitle.textContent = entry.name;
            nativeCopy.textContent =
                entry.note || "该示例仅支持原生运行，不提供网页运行时。";
            frame.src = "about:blank";
            return;
        }

        nativeOverlay.classList.add("hidden");

        // Load in iframe
        const targetUrl =
            entry.type === "standalone"
                ? entry.url
                : `./viewer.html?example=${encodeURIComponent(entry.id)}`;

        frame.src = "about:blank";
        requestAnimationFrame(() => {
            frame.src = targetUrl;
        });
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

    const elapsedTimer = setInterval(() => {
        elapsedEl.textContent = formatDuration(performance.now() - bootStart);
    }, 100);

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

    if (!entry || !entry.web_supported || entry.type !== "iframe") {
        clearInterval(elapsedTimer);
        updateProgress("Entry not available", 100);
        sendToGallery({
            state: "error",
            label: "Unavailable",
            detail: "传入的示例标识不在当前清单中或不支持网页运行。",
            exampleId,
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

    const module = await import(`./wasm/${entry.id}.js`);

    updateProgress("Booting runtime...", 70);
    sendToGallery({
        state: "booting",
        label: "Booting Runtime",
        exampleId: entry.id,
    });

    await module.default();
    await waitForPaints(2);

    const bootMs = performance.now() - bootStart;
    clearInterval(elapsedTimer);
    elapsedEl.textContent = formatDuration(bootMs);

    updateProgress("Ready", 100);
    sendToGallery({
        state: "ready",
        label: "Runtime Ready",
        exampleId: entry.id,
        bootMs,
        route: `?example=${entry.id}`,
    });

    // Smooth fade-out of loading overlay
    setTimeout(() => {
        overlay.classList.add("fade-out");
    }, 120);

    function updateProgress(status, pct) {
        statusEl.textContent = status;
        progressBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
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

function waitForPaints(count) {
    return new Promise((resolve) => {
        let remaining = count;
        const step = () => {
            if (--remaining <= 0) {
                resolve();
                return;
            }
            requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    });
}