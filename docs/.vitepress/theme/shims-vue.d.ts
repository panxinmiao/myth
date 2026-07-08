declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<Record<string, unknown>, Record<string, unknown>, unknown>
  export default component
}

declare module '*.css' {
  const css: string
  export default css
}

declare const __MYTH_GITHUB_PAGES__: boolean
