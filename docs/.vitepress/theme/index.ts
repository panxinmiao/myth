import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import HeroVideo from './components/HeroVideo.vue'
import LazyMermaid from './components/LazyMermaid.vue'
import './custom.css'
import { inBrowser, withBase } from 'vitepress'

export default {
  extends: DefaultTheme,
  enhanceApp(ctx: { app: any; router: any; siteData?: { value?: { base?: string } } }) {
    ctx.app.component('LazyMermaid', LazyMermaid)

    const { router } = ctx
    if (!inBrowser) return

    const isGithubPages = __MYTH_GITHUB_PAGES__
    const defaultLocale = isGithubPages ? 'en' : 'zh'

    const base = ctx.siteData?.value?.base ?? '/'
    const routePath = (path: string) => {
      if (base !== '/' && path === base.slice(0, -1)) return '/'
      if (base !== '/' && path.startsWith(base)) return `/${path.slice(base.length)}`
      return path
    }

    const localeRoutePath = (locale: 'en' | 'zh') => {
      if (isGithubPages) return locale === 'en' ? '/' : '/zh/'
      return locale === 'en' ? '/en/' : '/'
    }

    const routeLocale = (path: string) => {
      const normalizedPath = routePath(path)
      if (isGithubPages) return normalizedPath === '/zh/' || normalizedPath.startsWith('/zh/') ? 'zh' : 'en'
      return normalizedPath === '/en/' || normalizedPath.startsWith('/en/') ? 'en' : 'zh'
    }

    const initializeLocaleRoute = () => {
      const path = routePath(window.location.pathname)

      if (path !== '/' && path !== '/index.html') return

      const savedLocale = localStorage.getItem('user-locale')
      let targetLocale = savedLocale
      if (targetLocale !== 'en' && targetLocale !== 'zh') {
        targetLocale = null
      }

      if (!targetLocale) {
        const browserLang = navigator.language || (navigator as any).userLanguage || ''
        if (browserLang.startsWith('en')) {
          targetLocale = 'en'
        } else {
          targetLocale = defaultLocale
        }
      }

      const targetPath = localeRoutePath(targetLocale as 'en' | 'zh')
      const currentPath = path === '/index.html' ? '/' : path
      if (targetPath !== currentPath) {
        router.go(withBase(targetPath))
      }
    }

    initializeLocaleRoute()

    router.onAfterRouteChanged = (to: string) => {
      localStorage.setItem('user-locale', routeLocale(to))
    }
  },
  Layout() {
    return h(DefaultTheme.Layout, null, {
      // An engine-rendered demo video in the hero image area (right side).
      'home-hero-image': () => h(HeroVideo)
    })
  }
}
