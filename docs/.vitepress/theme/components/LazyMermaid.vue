<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps<{
  id: string
  graph: string
}>()

const root = ref<HTMLElement | null>(null)
const error = ref('')
let observer: MutationObserver | null = null
let renderVersion = 0

const decodedGraph = computed(() => decodeURIComponent(props.graph))

async function renderDiagram() {
  const currentVersion = ++renderVersion
  error.value = ''

  if (!root.value) return

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
  } catch (err) {
    if (currentVersion !== renderVersion) return
    error.value = err instanceof Error ? err.message : String(err)
    if (root.value) {
      root.value.textContent = error.value
    }
  }
}

onMounted(async () => {
  await nextTick()
  await renderDiagram()

  observer = new MutationObserver(() => {
    renderDiagram()
  })
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class'],
  })
})

onBeforeUnmount(() => {
  observer?.disconnect()
  observer = null
  renderVersion++
})

watch(decodedGraph, () => {
  renderDiagram()
})
</script>

<template>
  <div class="lazy-mermaid" :data-error="error || null">
    <div ref="root" class="lazy-mermaid__canvas" aria-live="polite"></div>
  </div>
</template>

<style scoped>
.lazy-mermaid {
  width: 100%;
  margin: 16px 0;
  overflow-x: auto;
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
