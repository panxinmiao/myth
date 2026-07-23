import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import HeroVideo from './components/HeroVideo.vue'
import LazyMermaid from './components/LazyMermaid.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }: { app: { component: (name: string, component: typeof LazyMermaid) => void } }) {
    app.component('LazyMermaid', LazyMermaid)
  },
  Layout() {
    return h(DefaultTheme.Layout, null, {
      // An engine-rendered demo video in the hero image area (right side).
      'home-hero-image': () => h(HeroVideo)
    })
  }
}
