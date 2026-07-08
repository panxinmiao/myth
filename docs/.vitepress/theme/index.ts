import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import HeroVideo from './components/HeroVideo.vue'
import './custom.css'
import { inBrowser } from 'vitepress'

export default {
  extends: DefaultTheme,
  enhanceApp({ router }: { router: any }) {
    if (!inBrowser) return

    const isGithubPages = __MYTH_GITHUB_PAGES__
    const defaultLocale = isGithubPages ? 'en' : 'zh'
    const localePath = (locale: 'en' | 'zh') => {
      if (isGithubPages) return locale === 'en' ? '/' : '/zh/'
      return locale === 'en' ? '/en/' : '/'
    }

    const routeLocale = (path: string) => {
      if (isGithubPages) return path === '/zh/' || path.startsWith('/zh/') ? 'zh' : 'en'
      return path === '/en/' || path.startsWith('/en/') ? 'en' : 'zh'
    }

    const initializeLocaleRoute = () => {
      const path = router.route.path

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

      const targetPath = localePath(targetLocale as 'en' | 'zh')
      const currentPath = path === '/index.html' ? '/' : path
      if (targetPath !== currentPath) {
        router.go(targetPath)
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
