const page = document.body.dataset.page;

if (page === "gallery") {
    initGallery().catch(handleFatalError);
} else if (page === "viewer") {
    initViewer().catch(handleFatalError);
}

function handleFatalError(error) {
    console.error(error);
}

async function initGallery() {
    const manifest = await fetchManifest("./examples.json");
    const entries = manifest.flatMap((category) =>
        category.items.map((item) => ({ ...item, category: category.category })),
    );

    const drawerToggle = document.getElementById("drawer-toggle");
    const sidebar = document.getElementById("sidebar");
    const sidebarClose = document.getElementById("sidebar-close");
    const navGroups = document.getElementById("nav-groups");
    const frame = document.getElementById("viewer-frame");
    const stageEmpty = document.getElementById("stage-empty");
    const stageEmptyLabel = document.getElementById("stage-empty-label");
    const stageEmptyTitle = document.getElementById("stage-empty-title");
    const stageEmptyCopy = document.getElementById("stage-empty-copy");
    const entryKind = document.getElementById("entry-kind");
    const entryName = document.getElementById("entry-name");
    const entryDescription = document.getElementById("entry-description");
    const entryStatus = document.getElementById("entry-status");
    const entrySourcePath = document.getElementById("entry-source-path");
    const entrySource = document.getElementById("entry-source");
    const entryOpen = document.getElementById("entry-open");

    let activeId = null;

    drawerToggle.addEventListener("click", () => {
        const open = !sidebar.classList.contains("is-open");
        sidebar.classList.toggle("is-open", open);
        drawerToggle.setAttribute("aria-expanded", String(open));
    });

    sidebarClose.addEventListener("click", () => {
        sidebar.classList.remove("is-open");
        drawerToggle.setAttribute("aria-expanded", "false");
    });

    navGroups.innerHTML = manifest
        .map((group) => {
            const items = group.items
                .map(
                    (item) => `
                        <button class="nav-item" type="button" data-entry-id="${item.id}">
                            <div class="nav-item-top">
                                <span class="nav-item-name">${escapeHtml(item.name)}</span>
                                <span class="nav-item-kind">${escapeHtml(item.type)}</span>
                            </div>
                            <p class="nav-item-copy">${escapeHtml(item.description)}</p>
                            ${item.note ? `<div class="nav-item-note">${escapeHtml(item.note)}</div>` : ""}
                        </button>
                    `,
                )
                .join("");
            return `
                <section class="nav-group">
                    <h2 class="nav-group-title">${escapeHtml(group.category)}</h2>
                    <div class="nav-list">${items}</div>
                </section>
            `;
        })
        .join("");

    navGroups.addEventListener("click", (event) => {
        const button = event.target.closest("[data-entry-id]");
        if (!button) {
            return;
        }

        const entry = entries.find((candidate) => candidate.id === button.dataset.entryId);
        if (!entry) {
            return;
        }

        void selectEntry(entry, true);
        sidebar.classList.remove("is-open");
        drawerToggle.setAttribute("aria-expanded", "false");
    });

    frame.addEventListener("load", () => {
        if (stageEmpty.classList.contains("hidden")) {
            entryStatus.textContent = "Ready";
            entryStatus.dataset.state = "ready";
        }
    });

    window.addEventListener("popstate", () => {
        const entry = entryFromUrl(entries);
        if (entry) {
            void selectEntry(entry, false);
        }
    });

    const initial = entryFromUrl(entries) ?? entries[0];
    if (initial) {
        await selectEntry(initial, false);
    }

    async function selectEntry(entry, pushHistory) {
        activeId = entry.id;
        updateActiveState(activeId);

        const kindLabel = entry.type === "standalone" ? "Standalone App" : "Interactive Example";
        entryKind.textContent = kindLabel;
        entryName.textContent = entry.name;
        entryDescription.textContent = entry.description;
        entrySourcePath.textContent = entry.source_path;
        entrySource.href = entry.source_url || "#";
        entrySource.setAttribute("aria-disabled", entry.source_url ? "false" : "true");

        const nextUrl = new URL(window.location.href);
        nextUrl.searchParams.set("example", entry.id);
        if (pushHistory) {
            history.pushState({ example: entry.id }, "", nextUrl);
        } else {
            history.replaceState({ example: entry.id }, "", nextUrl);
        }

        if (!entry.web_supported) {
            entryStatus.textContent = "Native Only";
            entryStatus.dataset.state = "native";
            entryOpen.href = "#";
            entryOpen.textContent = "仅源码可用";
            entryOpen.setAttribute("aria-disabled", "true");
            frame.src = "about:blank";
            stageEmpty.classList.remove("hidden");
            stageEmptyLabel.textContent = entry.note ? "Special Case" : "Native Only";
            stageEmptyTitle.textContent = entry.name;
            stageEmptyCopy.textContent = entry.note || "该条目不会在浏览器里执行，但仍保留在索引中供查阅和跳转源码。";
            return;
        }

        stageEmpty.classList.add("hidden");
        entryStatus.textContent = "Loading";
        entryStatus.dataset.state = "loading";

        const targetUrl = entry.type === "standalone"
            ? entry.url
            : `./viewer.html?example=${encodeURIComponent(entry.id)}`;

        entryOpen.href = targetUrl;
        entryOpen.textContent = entry.type === "standalone" ? "单独打开应用" : "单独打开示例";
        entryOpen.setAttribute("aria-disabled", "false");

        frame.src = "about:blank";
        await new Promise((resolve) => requestAnimationFrame(resolve));
        frame.src = targetUrl;
    }

    function updateActiveState(id) {
        document.querySelectorAll("[data-entry-id]").forEach((element) => {
            element.classList.toggle("is-active", element.dataset.entryId === id);
        });
    }
}

async function initViewer() {
    const manifest = await fetchManifest("./examples.json");
    const entries = manifest.flatMap((category) =>
        category.items.map((item) => ({ ...item, category: category.category })),
    );
    const params = new URLSearchParams(window.location.search);
    const exampleId = params.get("example");

    const entry = entries.find((candidate) => candidate.id === exampleId);
    const viewerTitle = document.getElementById("viewer-title");
    const viewerCopy = document.getElementById("viewer-copy");
    const viewerMeta = document.getElementById("viewer-meta");
    const overlay = document.getElementById("viewer-overlay");
    const overlayTitle = document.getElementById("viewer-overlay-title");
    const overlayCopy = document.getElementById("viewer-overlay-copy");
    const overlayStatus = document.getElementById("viewer-status");

    if (!entry) {
        overlayTitle.textContent = "Unknown entry";
        overlayCopy.textContent = "传入的示例标识不在当前清单中。";
        overlayStatus.textContent = "Manifest mismatch";
        return;
    }

    viewerTitle.textContent = entry.name;
    viewerCopy.textContent = entry.description;
    viewerMeta.textContent = `${entry.category} / ${entry.type}`;

    if (!entry.web_supported || entry.type !== "iframe") {
        overlayTitle.textContent = entry.name;
        overlayCopy.textContent = entry.note || "该条目不通过 viewer.html 挂载。";
        overlayStatus.textContent = "Not available in viewer";
        return;
    }

    overlayTitle.textContent = entry.name;
    overlayCopy.textContent = "初始化 WebGPU 运行时并加载对应的 wasm 模块。";
    overlayStatus.textContent = "Fetching module";

    const module = await import(`./wasm/${entry.id}.js`);
    overlayStatus.textContent = "Booting runtime";
    await module.default();
    overlay.classList.add("hidden");
}

async function fetchManifest(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
        throw new Error(`Failed to load manifest: ${response.status}`);
    }
    return response.json();
}

function entryFromUrl(entries) {
    const params = new URLSearchParams(window.location.search);
    const id = params.get("example");
    if (!id) {
        return null;
    }
    return entries.find((entry) => entry.id === id) || null;
}

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}