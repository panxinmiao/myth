import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

const isGithub = process.env.GITHUB_ACTIONS === 'true'

const zhBase = isGithub ? '/zh' : ''
const enBase = isGithub ? '' : '/en'

// Shared, language-agnostic configuration
const GITHUB_REPO = 'https://github.com/panxinmiao/myth'

// Deployment base. Local dev/preview uses '/'; CI sets DOCS_BASE='/myth/'
// so the site is served from https://panxinmiao.github.io/myth/.
const BASE = process.env.DOCS_BASE ?? '/'

// The Gallery is built as a static sub-app under <base>/gallery/.
// `target: '_self'` forces a full-page navigation out of the VitePress SPA.
const GALLERY_LINK = `${BASE}gallery/`

// ── Chinese sidebar ──────────────────────────────────────────────
function sidebarZh() {
  return [
    {
      text: '基础指南',
      collapsed: false,
      items: [
        { text: '简介与愿景', link: `${zhBase}/guide/introduction` },
        { text: '核心特性总览', link: `${zhBase}/guide/features` },
        { text: '快速开始', link: `${zhBase}/guide/quick-start` },
        { text: '场景与节点系统', link: `${zhBase}/guide/scene-graph` },
        { text: '资产、glTF 与动画', link: `${zhBase}/guide/assets-animation` },
        { text: 'Python 绑定', link: `${zhBase}/guide/python` }
      ]
    },
    {
      text: '引擎架构',
      collapsed: false,
      items: [
        { text: '渲染路径与帧合成', link: `${zhBase}/architecture/rendering-pipeline` },
        { text: 'Render Graph 渲染图', link: `${zhBase}/architecture/render-graph` },
        { text: '异步资源与加载管线', link: `${zhBase}/architecture/asset-pipeline` },
        { text: '高性能材质系统', link: `${zhBase}/architecture/material-system` }
      ]
    },
    {
      text: '进阶渲染特性',
      collapsed: false,
      items: [
        { text: 'PBR 物理材质', link: `${zhBase}/advanced/pbr-materials` },
        { text: 'GPU-Driven 与聚类光照', link: `${zhBase}/advanced/clustered-shading` },
        { text: '后处理与屏幕空间特效', link: `${zhBase}/advanced/post-processing` },
        { text: '程序化天空与大气', link: `${zhBase}/advanced/procedural-sky` },
        { text: '3DGS 高斯溅射融合渲染', link: `${zhBase}/advanced/3dgs-integration` },
        { text: '自定义 Shader 与后处理', link: `${zhBase}/advanced/custom-shader` },
        { text: '离屏与无头渲染', link: `${zhBase}/advanced/headless-rendering` }
      ]
    }
  ]
}

// ── Chinese articles sidebar ────────────────────────────────
function sidebarArticlesZh() {
  return [
    {
      text: '技术文章',
      collapsed: false,
      items: [
        { text: '文章列表', link: `${zhBase}/articles/` },
        { text: '构建基于 SSA 的声明式渲染图', link: `${zhBase}/articles/render-graph-design` }
      ]
    }
  ]
}

// ── English sidebar ─────────────────────────────────────────────────────
function sidebarEn() {
  return [
    {
      text: 'Guide',
      collapsed: false,
      items: [
        { text: 'Introduction & Vision', link: `${enBase}/guide/introduction` },
        { text: 'Feature Overview', link: `${enBase}/guide/features` },
        { text: 'Quick Start', link: `${enBase}/guide/quick-start` },
        { text: 'Scene & Node System', link: `${enBase}/guide/scene-graph` },
        { text: 'Assets, glTF & Animation', link: `${enBase}/guide/assets-animation` },
        { text: 'Python Bindings', link: `${enBase}/guide/python` }
      ]
    },
    {
      text: 'Architecture',
      collapsed: false,
      items: [
        { text: 'Render Paths & Frame Composer', link: `${enBase}/architecture/rendering-pipeline` },
        { text: 'Render Graph', link: `${enBase}/architecture/render-graph` },
        { text: 'Async Asset Pipeline', link: `${enBase}/architecture/asset-pipeline` },
        { text: 'Material System', link: `${enBase}/architecture/material-system` }
      ]
    },
    {
      text: 'Advanced Rendering',
      collapsed: false,
      items: [
        { text: 'PBR Materials', link: `${enBase}/advanced/pbr-materials` },
        { text: 'GPU-Driven Clustered Lighting', link: `${enBase}/advanced/clustered-shading` },
        { text: 'Post-Processing & Screen-Space FX', link: `${enBase}/advanced/post-processing` },
        { text: 'Procedural Sky & Atmosphere', link: `${enBase}/advanced/procedural-sky` },
        { text: '3D Gaussian Splatting', link: `${enBase}/advanced/3dgs-integration` },
        { text: 'Custom Shaders & Post FX', link: `${enBase}/advanced/custom-shader` },
        { text: 'Headless & Offscreen Rendering', link: `${enBase}/advanced/headless-rendering` }
      ]
    }
  ]
}

// ── English articles sidebar ───────────────────────────────
function sidebarArticlesEn() {
  return [
    {
      text: 'Articles',
      collapsed: false,
      items: [
        { text: 'All Articles', link: `${enBase}/articles/` },
        { text: 'Building an SSA-based Declarative Render Graph', link: `${enBase}/articles/render-graph-design` }
      ]
    }
  ]
}

function getZhThemeConfig() {
  return {
    nav: [
      { text: '指南', link: `${zhBase}/guide/introduction`, activeMatch: `${zhBase}/guide/` },
      { text: '架构', link: `${zhBase}/architecture/rendering-pipeline`, activeMatch: `${zhBase}/architecture/` },
      { text: '进阶', link: `${zhBase}/advanced/pbr-materials`, activeMatch: `${zhBase}/advanced/` },
      { text: '文章', link: `${zhBase}/articles/`, activeMatch: `${zhBase}/articles/` },
      { text: 'Gallery', link: GALLERY_LINK, target: '_self' },
      {
        text: '更多',
        items: [
          { text: 'Examples 示例', link: `${GITHUB_REPO}/tree/main/examples` },
          { text: 'Python 绑定', link: `${GITHUB_REPO}/tree/main/bindings/python` },
          { text: 'Change Log', link: `${GITHUB_REPO}/tree/main/CHANGELOG.md` },
        ]
      }
    ],
    sidebar: {
      [`${zhBase}/articles/`]: sidebarArticlesZh(),
      [`${zhBase}/`]: sidebarZh()
    },
    outline: { label: '本页大纲', level: [2, 3] as [number, number] },
    docFooter: { prev: '上一页', next: '下一页' },
    lastUpdatedText: '最后更新于',
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    editLink: {
      pattern: `${GITHUB_REPO}/edit/main/docs/:path`,
      text: '在 GitHub 上编辑此页'
    },
    footer: {
      message: '基于 MIT / Apache-2.0 双协议发布',
      copyright: 'Copyright © 2026-present Pan Xinmiao'
    }
  }
}

function getEnThemeConfig() {
  return {
    nav: [
      { text: 'Guide', link: `${enBase}/guide/introduction`, activeMatch: `${enBase}/guide/` },
      { text: 'Architecture', link: `${enBase}/architecture/rendering-pipeline`, activeMatch: `${enBase}/architecture/` },
      { text: 'Advanced', link: `${enBase}/advanced/pbr-materials`, activeMatch: `${enBase}/advanced/` },
      { text: 'Articles', link: `${enBase}/articles/`, activeMatch: `${enBase}/articles/` },
      { text: 'Gallery', link: GALLERY_LINK, target: '_self' },
      {
        text: 'More',
        items: [
          { text: 'Examples', link: `${GITHUB_REPO}/tree/main/examples` },
          { text: 'Python Bindings', link: `${GITHUB_REPO}/tree/main/bindings/python` },
          { text: 'Change Log', link: `${GITHUB_REPO}/tree/main/CHANGELOG.md` },
        ]
      }
    ],
    sidebar: {
      [`${enBase}/articles/`]: sidebarArticlesEn(),
      [`${enBase}/`]: sidebarEn()
    },
    outline: { label: 'On this page', level: [2, 3] as [number, number] },
    editLink: {
      pattern: `${GITHUB_REPO}/edit/main/docs/:path`,
      text: 'Edit this page on GitHub'
    },
    footer: {
      message: 'Released under the MIT / Apache-2.0 dual license',
      copyright: 'Copyright © 2026-present Pan Xinmiao'
    }
  }
}

// ── 导出 VitePress 配置 ──────────────────────────────────────────────────
export default withMermaid(
  defineConfig({
    title: 'Myth Engine',
    description: '极致性能的轻量级 Rust 渲染引擎 · A high-performance, lightweight Rust rendering engine',

    appearance: 'dark',

    base: BASE,
    lastUpdated: true,
    cleanUrls: true,
    ignoreDeadLinks: !isGithub,
    outDir: '../dist',

    vite: {
      define: {
        __MYTH_GITHUB_PAGES__: JSON.stringify(isGithub)
      },
      build: {
        emptyOutDir: false
      }
    },

    head: [
      ['meta', { name: 'theme-color', content: '#4a6f9f' }]
    ],

    rewrites: isGithub
      ? (id: string) => (id.startsWith('en/') ? id.slice(3) : `zh/${id}`)
      : undefined,

    themeConfig: {
      socialLinks: [{ icon: 'github', link: GITHUB_REPO }],
      search: { provider: 'local' }
    },

    locales: isGithub
      ? {
          root: {
            label: 'English',
            lang: 'en-US',
            themeConfig: getEnThemeConfig()
          },
          zh: {
            label: '简体中文',
            lang: 'zh-CN',
            link: '/zh/',
            themeConfig: getZhThemeConfig()
          }
        }
      : {
          root: {
            label: '简体中文',
            lang: 'zh-CN',
            themeConfig: getZhThemeConfig()
          },
          en: {
            label: 'English',
            lang: 'en-US',
            link: '/en/',
            themeConfig: getEnThemeConfig()
          }
        }
  })
)
