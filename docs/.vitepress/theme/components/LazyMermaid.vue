<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps<{
  id: string
  graph: string
}>()

const container = ref<HTMLElement | null>(null)
const root = ref<HTMLElement | null>(null)
const error = ref('')
const hasRendered = ref(false)
const isRendering = ref(false)
let themeObserver: MutationObserver | null = null
let viewportObserver: IntersectionObserver | null = null
let isActivated = false
let renderVersion = 0

const decodedGraph = computed(() => decodeURIComponent(props.graph))

async function renderDiagram() {
  const currentVersion = ++renderVersion
  error.value = ''

  if (!root.value) return
  isRendering.value = true

  try {
    const mermaid = (await import('mermaid')).default
    const dark = document.documentElement.classList.contains('dark')
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'loose',
      theme: dark ? 'dark' : 'default',
    })

    const { svg, bindFunctions } = await mermaid.render(`${props.id}-${currentVersion}`, decodedGraph.value)
    if (currentVersion !== renderVersion || !root.value) return

    root.value.innerHTML = svg
    bindFunctions?.(root.value)
    hasRendered.value = true
  } catch (err) {
    if (currentVersion !== renderVersion) return
    error.value = err instanceof Error ? err.message : String(err)
    if (root.value) {
      root.value.textContent = error.value
    }
  } finally {
    if (currentVersion === renderVersion) {
      isRendering.value = false
    }
  }
}

function activate() {
  if (isActivated) return
  isActivated = true
  viewportObserver?.disconnect()
  viewportObserver = null
  void renderDiagram()
}

onMounted(async () => {
  await nextTick()

  themeObserver = new MutationObserver(() => {
    if (isActivated) void renderDiagram()
  })
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class'],
  })

  if (!container.value || !('IntersectionObserver' in window)) {
    activate()
    return
  }

  viewportObserver = new IntersectionObserver(
    (entries) => {
      if (entries.some((entry) => entry.isIntersecting)) activate()
    },
    { rootMargin: '320px 0px' },
  )
  viewportObserver.observe(container.value)
})

onBeforeUnmount(() => {
  themeObserver?.disconnect()
  viewportObserver?.disconnect()
  themeObserver = null
  viewportObserver = null
  renderVersion++
})

watch(decodedGraph, () => {
  if (isActivated) void renderDiagram()
})
</script>

<template>
  <div
    ref="container"
    class="lazy-mermaid"
    :data-error="error || null"
    :data-rendered="hasRendered || null"
  >
    <div
      ref="root"
      class="lazy-mermaid__canvas"
      aria-live="polite"
      :aria-busy="isRendering"
    ></div>
  </div>
</template>

<style scoped>
.lazy-mermaid {
  width: 100%;
  min-height: 120px;
  margin: 16px 0;
  overflow-x: auto;
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
}

.lazy-mermaid[data-rendered],
.lazy-mermaid[data-error] {
  min-height: 0;
  background: transparent;
}

.lazy-mermaid__canvas {
  min-width: min-content;
}

.lazy-mermaid__canvas :deep(svg) {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
}

.lazy-mermaid[data-error] .lazy-mermaid__canvas {
  padding: 12px 14px;
  border: 1px solid var(--vp-c-danger-2);
  border-radius: 8px;
  color: var(--vp-c-danger-1);
  background: var(--vp-c-danger-soft);
  font-size: 13px;
  line-height: 1.5;
  white-space: pre-wrap;
}
</style>
