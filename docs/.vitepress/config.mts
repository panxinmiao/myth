import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineConfig } from 'vitepress'

const GITHUB_REPO = 'https://github.com/panxinmiao/myth'
const SITE_URL = 'https://panxinmiao.github.io/myth/'
const SOCIAL_IMAGE_URL = new URL('images/hero.jpg', SITE_URL).href

function normalizeBase(base: string) {
  if (!base || base === '/') return '/'
  return `/${base.replace(/^\/+|\/+$/g, '')}/`
}

const BASE = normalizeBase(process.env.DOCS_BASE ?? '/')
const publicPath = (path: string) => `${BASE}${path.replace(/^\/+/, '')}`

function pageUrl(page: string) {
  let route = page.replace(/\\/g, '/').replace(/\.md$/, '')
  if (route === 'index') route = ''
  else if (route.endsWith('/index')) route = route.slice(0, -'index'.length)
  return new URL(route, SITE_URL).href
}

function isGalleryLink(url: string) {
  const path = url.replace(/[?#].*$/, '').replace(/\\/g, '/')
  return /(?:^|\/)gallery(?:\/index)?\/?$/.test(path)
}

function sidebarEn() {
  return [
    {
      text: 'Guide',
      collapsed: false,
      items: [
        { text: 'Introduction & Vision', link: '/guide/introduction' },
        { text: 'Quick Start', link: '/guide/quick-start' },
        { text: 'Feature Overview', link: '/guide/features' },
        { text: 'Scene & Node System', link: '/guide/scene-graph' },
        { text: 'Assets, glTF & Animation', link: '/guide/assets-animation' },
        { text: 'Python Bindings', link: '/guide/python' },
      ]
    },
    {
      text: 'Architecture',
      collapsed: true,
      items: [
        { text: 'Render Paths & Frame Composer', link: '/architecture/rendering-pipeline' },
        { text: 'Render Graph', link: '/architecture/render-graph' },
        { text: 'Async Asset Pipeline', link: '/architecture/asset-pipeline' },
        { text: 'Material System', link: '/architecture/material-system' }
      ]
    },
    {
      text: 'Advanced Rendering',
      collapsed: true,
      items: [
        { text: 'PBR Materials', link: '/advanced/pbr-materials' },
        { text: 'GPU-Driven Clustered Lighting', link: '/advanced/clustered-shading' },
        { text: 'Post-Processing & Screen-Space FX', link: '/advanced/post-processing' },
        { text: 'Procedural Sky & Atmosphere', link: '/advanced/procedural-sky' },
        { text: '3D Gaussian Splatting', link: '/advanced/3dgs-integration' },
        { text: 'Custom Shaders & Post FX', link: '/advanced/custom-shader' },
        { text: 'Headless & Offscreen Rendering', link: '/advanced/headless-rendering' }
      ]
    },
  ]
}

function sidebarZh() {
  return [
    {
      text: '基础指南',
      collapsed: false,
      items: [
        { text: '简介与愿景', link: '/zh/guide/introduction' },
        { text: '快速开始', link: '/zh/guide/quick-start' },
        { text: '核心特性总览', link: '/zh/guide/features' },
        { text: '场景与节点系统', link: '/zh/guide/scene-graph' },
        { text: '资产、glTF 与动画', link: '/zh/guide/assets-animation' },
        { text: 'Python 绑定', link: '/zh/guide/python' },
      ]
    },
    {
      text: '引擎架构',
      collapsed: true,
      items: [
        { text: '渲染路径与帧合成', link: '/zh/architecture/rendering-pipeline' },
        { text: 'Render Graph 渲染图', link: '/zh/architecture/render-graph' },
        { text: '异步资源与加载管线', link: '/zh/architecture/asset-pipeline' },
        { text: '高性能材质系统', link: '/zh/architecture/material-system' }
      ]
    },
    {
      text: '进阶渲染特性',
      collapsed: true,
      items: [
        { text: 'PBR 物理材质', link: '/zh/advanced/pbr-materials' },
        { text: 'GPU-Driven 与聚类光照', link: '/zh/advanced/clustered-shading' },
        { text: '后处理与屏幕空间特效', link: '/zh/advanced/post-processing' },
        { text: '程序化天空与大气', link: '/zh/advanced/procedural-sky' },
        { text: '3DGS 高斯溅射融合渲染', link: '/zh/advanced/3dgs-integration' },
        { text: '自定义 Shader 与后处理', link: '/zh/advanced/custom-shader' },
        { text: '离屏与无头渲染', link: '/zh/advanced/headless-rendering' }
      ]
    },
  ]
}

function sidebarArticlesEn() {
  return [
    {
      text: 'Articles',
      collapsed: false,
      items: [
        { text: 'All Articles', link: '/articles/' },
        { text: 'Building an SSA-based Declarative Render Graph', link: '/articles/render-graph-design' },
      ]
    }
  ]
}

function sidebarArticlesZh() {
  return [
    {
      text: '技术文章',
      collapsed: false,
      items: [
        { text: '文章列表', link: '/zh/articles/' },
        { text: '构建基于 SSA 的声明式渲染图', link: '/zh/articles/render-graph-design' },
      ]
    }
  ]
}

const searchEn = {
  provider: 'local' as const,
  options: {
    detailedView: true as const,
    translations: {
      button: {
        buttonText: 'Search',
        buttonAriaLabel: 'Search documentation'
      },
      modal: {
        displayDetails: 'Display detailed results',
        resetButtonTitle: 'Reset search',
        backButtonTitle: 'Close search',
        noResultsText: 'No results found',
        footer: {
          selectText: 'Select',
          selectKeyAriaLabel: 'Enter',
          navigateText: 'Navigate',
          navigateUpKeyAriaLabel: 'Arrow up',
          navigateDownKeyAriaLabel: 'Arrow down',
          closeText: 'Close',
          closeKeyAriaLabel: 'Escape'
        }
      }
    }
  }
}

const searchZh = {
  provider: 'local' as const,
  options: {
    detailedView: true as const,
    translations: {
      button: {
        buttonText: '搜索',
        buttonAriaLabel: '搜索文档'
      },
      modal: {
        displayDetails: '显示详细结果',
        resetButtonTitle: '清除搜索',
        backButtonTitle: '关闭搜索',
        noResultsText: '没有找到相关结果',
        footer: {
          selectText: '选择',
          selectKeyAriaLabel: '回车键',
          navigateText: '切换结果',
          navigateUpKeyAriaLabel: '向上方向键',
          navigateDownKeyAriaLabel: '向下方向键',
          closeText: '关闭',
          closeKeyAriaLabel: 'Escape 键'
        }
      }
    }
  }
}

function getEnThemeConfig() {
  return {
    nav: [
      { text: 'Guide', link: '/guide/introduction', activeMatch: '/guide/' },
      { text: 'Architecture', link: '/architecture/rendering-pipeline', activeMatch: '/architecture/' },
      { text: 'Advanced', link: '/advanced/pbr-materials', activeMatch: '/advanced/' },
      { text: 'Articles', link: '/articles/', activeMatch: '/articles/' },
      { text: 'Gallery', link: '/gallery/', target: '_self' },
      {
        text: 'More',
        items: [
          { text: 'Examples on GitHub', link: `${GITHUB_REPO}/tree/main/examples` },
          { text: 'Python Bindings', link: `${GITHUB_REPO}/tree/main/bindings/python` },
          { text: 'Changelog', link: `${GITHUB_REPO}/blob/main/CHANGELOG.md` }
        ]
      }
    ],
    sidebar: {
      '/articles/': sidebarArticlesEn(),
      '/': sidebarEn()
    },
    outline: { label: 'On this page', level: [2, 3] as [number, number] },
    docFooter: { prev: 'Previous page', next: 'Next page' },
    lastUpdated: {
      text: 'Last updated',
      formatOptions: { dateStyle: 'medium' as const, forceLocale: true }
    },
    returnToTopLabel: 'Return to top',
    sidebarMenuLabel: 'Documentation menu',
    darkModeSwitchLabel: 'Appearance',
    lightModeSwitchTitle: 'Switch to light theme',
    darkModeSwitchTitle: 'Switch to dark theme',
    langMenuLabel: 'Change language',
    skipToContentLabel: 'Skip to content',
    notFound: {
      title: 'PAGE NOT FOUND',
      quote: 'The page may have moved. Search the docs or return home to keep exploring Myth.',
      linkLabel: 'Go to the Myth Engine home page',
      linkText: 'Take me home'
    },
    editLink: {
      pattern: `${GITHUB_REPO}/edit/main/docs/:path`,
      text: 'Edit this page on GitHub'
    },
    footer: {
      message: 'Released under the MIT / Apache-2.0 dual license',
      copyright: 'Copyright © 2026-present Pan Xinmiao'
    },
    socialLinks: [{ icon: 'github', link: GITHUB_REPO, ariaLabel: 'Myth Engine on GitHub' }],
    search: searchEn,
    externalLinkIcon: true,
    i18nRouting: true
  }
}

function getZhThemeConfig() {
  return {
    nav: [
      { text: '指南', link: '/zh/guide/introduction', activeMatch: '/zh/guide/' },
      { text: '架构', link: '/zh/architecture/rendering-pipeline', activeMatch: '/zh/architecture/' },
      { text: '进阶', link: '/zh/advanced/pbr-materials', activeMatch: '/zh/advanced/' },
      { text: '文章', link: '/zh/articles/', activeMatch: '/zh/articles/' },
      { text: 'Gallery', link: '/gallery/', target: '_self' },
      {
        text: '更多',
        items: [
          { text: 'GitHub 示例', link: `${GITHUB_REPO}/tree/main/examples` },
          { text: 'Python 绑定', link: `${GITHUB_REPO}/tree/main/bindings/python` },
          { text: '更新日志', link: `${GITHUB_REPO}/blob/main/CHANGELOG.md` }
        ]
      }
    ],
    sidebar: {
      '/zh/articles/': sidebarArticlesZh(),
      '/zh/': sidebarZh()
    },
    outline: { label: '本页大纲', level: [2, 3] as [number, number] },
    docFooter: { prev: '上一页', next: '下一页' },
    lastUpdated: {
      text: '最后更新于',
      formatOptions: { dateStyle: 'medium' as const, forceLocale: true }
    },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '文档导航',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色主题',
    darkModeSwitchTitle: '切换到深色主题',
    langMenuLabel: '切换语言',
    skipToContentLabel: '跳到主要内容',
    notFound: {
      title: '页面未找到',
      quote: '页面可能已经移动。你可以搜索文档，或返回首页继续探索 Myth。',
      linkLabel: '返回 Myth Engine 首页',
      linkText: '返回首页'
    },
    editLink: {
      pattern: `${GITHUB_REPO}/edit/main/docs/:path`,
      text: '在 GitHub 上编辑此页'
    },
    footer: {
      message: '基于 MIT / Apache-2.0 双协议发布',
      copyright: 'Copyright © 2026-present Pan Xinmiao'
    },
    socialLinks: [{ icon: 'github', link: GITHUB_REPO, ariaLabel: '在 GitHub 上查看 Myth Engine' }],
    search: searchZh,
    externalLinkIcon: true,
    i18nRouting: true
  }
}

export default defineConfig({
  title: 'Myth Engine',
  description: 'A developer-focused, cross-platform Rust rendering engine for real-time 3D across native, WebGPU, and Python.',
  appearance: 'dark',
  base: BASE,
  lastUpdated: true,
  cleanUrls: true,
  // Gallery is assembled into dist/ after the VitePress build.
  ignoreDeadLinks: [isGalleryLink],
  outDir: '../dist',

  vite: {
    build: {
      emptyOutDir: true
    }
  },

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: publicPath('/favicon.svg') }],
    ['meta', { name: 'application-name', content: 'Myth Engine' }],
    ['meta', { name: 'theme-color', content: '#f6f6f7', media: '(prefers-color-scheme: light)' }],
    ['meta', { name: 'theme-color', content: '#1b1b1f', media: '(prefers-color-scheme: dark)' }],
    ['meta', { property: 'og:site_name', content: 'Myth Engine' }]
  ],

  sitemap: {
    hostname: SITE_URL
  },

  transformHead({ page, siteConfig, pageData, title, description }) {
    if (page === '404.md' || pageData.isNotFound) {
      return [['meta', { name: 'robots', content: 'noindex, nofollow' }]]
    }

    const normalizedPage = page.replace(/\\/g, '/')
    const isZh = normalizedPage.startsWith('zh/')
    const englishPage = isZh ? normalizedPage.slice('zh/'.length) : normalizedPage
    const chinesePage = isZh ? normalizedPage : `zh/${normalizedPage}`
    const canonicalUrl = pageUrl(normalizedPage)
    const englishUrl = pageUrl(englishPage)
    const chineseUrl = pageUrl(chinesePage)
    const hasEnglish = existsSync(resolve(siteConfig.srcDir, englishPage))
    const hasChinese = existsSync(resolve(siteConfig.srcDir, chinesePage))
    const pageType = /(^|\/)(articles|releases)\//.test(normalizedPage) ? 'article' : 'website'

    const tags: Array<[string, Record<string, string>]> = [
      ['link', { rel: 'canonical', href: canonicalUrl }],
      ['meta', { property: 'og:type', content: pageType }],
      ['meta', { property: 'og:title', content: title }],
      ['meta', { property: 'og:description', content: description }],
      ['meta', { property: 'og:url', content: canonicalUrl }],
      ['meta', { property: 'og:image', content: SOCIAL_IMAGE_URL }],
      ['meta', { property: 'og:image:type', content: 'image/jpeg' }],
      ['meta', { property: 'og:image:width', content: '1728' }],
      ['meta', { property: 'og:image:height', content: '972' }],
      ['meta', { property: 'og:image:alt', content: 'Myth Engine real-time rendering showcase' }],
      ['meta', { property: 'og:locale', content: isZh ? 'zh_CN' : 'en_US' }],
      ['meta', { property: 'og:locale:alternate', content: isZh ? 'en_US' : 'zh_CN' }],
      ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
      ['meta', { name: 'twitter:title', content: title }],
      ['meta', { name: 'twitter:description', content: description }],
      ['meta', { name: 'twitter:image', content: SOCIAL_IMAGE_URL }],
      ['meta', { name: 'twitter:image:alt', content: 'Myth Engine real-time rendering showcase' }]
    ]

    if (hasEnglish) {
      tags.push(['link', { rel: 'alternate', hreflang: 'en-US', href: englishUrl }])
      tags.push(['link', { rel: 'alternate', hreflang: 'x-default', href: englishUrl }])
    }
    if (hasChinese) {
      tags.push(['link', { rel: 'alternate', hreflang: 'zh-CN', href: chineseUrl }])
    }

    return tags
  },

  markdown: {
    config(md) {
      const defaultFence = md.renderer.rules.fence?.bind(md.renderer.rules)
      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const language = token.info.trim().split(/\s+/, 1)[0]
        if (language !== 'mermaid') {
          return defaultFence ? defaultFence(tokens, idx, options, env, self) : self.renderToken(tokens, idx, options)
        }

        const id = `mermaid-${idx}`
        const graph = encodeURIComponent(token.content)
        return `<LazyMermaid id="${id}" graph="${graph}"></LazyMermaid>`
      }
    }
  },

  // VitePress discovers the local-search indexer from the global theme
  // config at build time. Locale configs below then provide translated UI.
  themeConfig: {
    search: { provider: 'local' }
  },

  locales: {
    root: {
      label: 'English',
      lang: 'en-US',
      description: 'A developer-focused, cross-platform Rust rendering engine for real-time 3D across native, WebGPU, and Python.',
      themeConfig: getEnThemeConfig()
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      description: '面向开发者、覆盖原生平台、WebGPU 与 Python 的跨平台 Rust 实时 3D 渲染引擎。',
      themeConfig: getZhThemeConfig()
    }
  }
})
