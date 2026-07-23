#!/usr/bin/env node

import { readdir, readFile } from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(scriptDir, '..')
const docsDir = path.resolve(projectRoot, 'docs')
const localOrigin = 'https://docs.example.invalid'
const productionOrigin = new URL(process.env.DOCS_ORIGIN ?? 'https://panxinmiao.github.io').origin
const canonicalBase = normalizeBase(process.env.DOCS_CANONICAL_BASE ?? '/myth/')
const errors = new Set()

const options = parseArguments(process.argv.slice(2))
const distDir = path.resolve(projectRoot, options.dist)
const base = normalizeBase(options.base)

const contentSections = [
  'advanced',
  'architecture',
  'articles',
  'guide',
  'reference',
  'releases',
]
const englishSectionPattern = new RegExp(`^/(?:${contentSections.join('|')})(?:/|$)`)

function parseArguments(args) {
  const positional = []
  let sourceOnly = false
  let requireGallery = false

  for (const argument of args) {
    if (argument === '--source-only') {
      sourceOnly = true
    } else if (argument === '--require-gallery') {
      requireGallery = true
    } else if (argument.startsWith('--')) {
      throw new Error(`Unknown option: ${argument}`)
    } else {
      positional.push(argument)
    }
  }

  if (positional.length > 2) {
    throw new Error('Usage: check-doc-site.mjs [dist] [base] [--source-only] [--require-gallery]')
  }

  return {
    dist: positional[0] ?? 'dist',
    base: positional[1] ?? process.env.DOCS_BASE ?? '/',
    sourceOnly,
    requireGallery,
  }
}

function normalizeBase(value) {
  const withLeadingSlash = value.startsWith('/') ? value : `/${value}`
  return withLeadingSlash.endsWith('/') ? withLeadingSlash : `${withLeadingSlash}/`
}

function toPosix(relativePath) {
  return relativePath.split(path.sep).join('/')
}

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true })
  const files = []

  for (const entry of entries) {
    const absolutePath = path.join(directory, entry.name)
    if (entry.isDirectory()) {
      files.push(...await walk(absolutePath))
    } else if (entry.isFile()) {
      files.push(absolutePath)
    }
  }

  return files
}

function pageRoute(relativeHtmlPath) {
  const relativePath = toPosix(relativeHtmlPath)

  if (relativePath === 'index.html') return '/'
  if (relativePath.endsWith('/index.html')) {
    return `/${relativePath.slice(0, -'index.html'.length)}`
  }

  return `/${relativePath.slice(0, -'.html'.length)}`
}

function withBase(route, selectedBase = base) {
  if (route === '/') return selectedBase
  return `${selectedBase.slice(0, -1)}${route}`
}

function routeAliases(relativeHtmlPath) {
  const route = pageRoute(relativeHtmlPath)
  const aliases = [route]

  if (route === '/') return aliases
  if (route.endsWith('/')) {
    aliases.push(route.slice(0, -1))
  }

  return aliases
}

function pageTargetAliases(page) {
  if (!isGalleryContent(page.route)) {
    return page.relativePath === '404.html'
      ? ['/404.html']
      : routeAliases(page.relativePath)
  }

  const rawTarget = `/${page.relativePath}`
  const aliases = new Set([page.route, rawTarget])
  if (rawTarget.endsWith('/index.html')) {
    const directoryTarget = rawTarget.slice(0, -'index.html'.length)
    aliases.add(directoryTarget)
    aliases.add(directoryTarget.slice(0, -1))
  }
  return [...aliases]
}

function builtPageRoute(relativeHtmlPath) {
  const relativePath = toPosix(relativeHtmlPath)

  if (relativePath === '404.html') return '/404.html'
  if (!relativePath.startsWith('gallery/')) return pageRoute(relativePath)
  if (relativePath === 'gallery/index.html') return '/gallery/'
  return `/${relativePath}`
}

function isIgnoredUrl(rawUrl) {
  return rawUrl === ''
    || /^(?:about|blob|data|javascript|mailto|tel):/i.test(rawUrl)
}

function isGalleryMount(sitePath) {
  return sitePath === '/gallery' || sitePath === '/gallery/'
}

function isGalleryContent(sitePath) {
  return sitePath === '/gallery' || sitePath.startsWith('/gallery/')
}

function decodeUrlPart(value) {
  try {
    return decodeURIComponent(value)
  } catch {
    return null
  }
}

function unpackSitePath(pathname, selectedBase) {
  const decodedPathname = decodeUrlPart(pathname)
  if (decodedPathname === null) {
    return { error: 'invalid URL encoding' }
  }

  if (selectedBase === '/') {
    return { sitePath: decodedPathname || '/' }
  }

  const baseWithoutSlash = selectedBase.slice(0, -1)
  if (decodedPathname === baseWithoutSlash) {
    return { sitePath: '/' }
  }
  if (!decodedPathname.startsWith(selectedBase)) {
    return { error: `escapes configured base ${selectedBase}` }
  }

  const relativeTarget = decodedPathname.slice(selectedBase.length)
  return { sitePath: relativeTarget === '' ? '/' : `/${relativeTarget}` }
}

function parseTagAttributes(tag) {
  const attributes = {}
  const pattern = /([:@\w-]+)\s*=\s*(["'])(.*?)\2/g
  for (const match of tag.matchAll(pattern)) {
    attributes[match[1].toLowerCase()] = match[3]
  }
  return attributes
}

function findTags(html, name) {
  const pattern = new RegExp(`<${name}\\b[^>]*>`, 'gi')
  return [...html.matchAll(pattern)].map(match => parseTagAttributes(match[0]))
}

function recordHtmlError(page, subject, message) {
  errors.add(`${page.relativePath}: ${message}: ${subject}`)
}

function stripInlineCode(line) {
  let output = ''

  for (let index = 0; index < line.length;) {
    if (line[index] !== '`') {
      output += line[index]
      index += 1
      continue
    }

    let openerEnd = index
    while (line[openerEnd] === '`') openerEnd += 1
    const delimiterLength = openerEnd - index
    let closerStart = openerEnd
    let closerEnd = -1

    while (closerStart < line.length) {
      closerStart = line.indexOf('`', closerStart)
      if (closerStart === -1) break
      let candidateEnd = closerStart
      while (line[candidateEnd] === '`') candidateEnd += 1
      if (candidateEnd - closerStart === delimiterLength) {
        closerEnd = candidateEnd
        break
      }
      closerStart = candidateEnd
    }

    if (closerEnd === -1) {
      output += line.slice(index, openerEnd)
      index = openerEnd
    } else {
      index = closerEnd
    }
  }

  return output
}

function stripMarkdownCode(markdown) {
  const withoutComments = markdown
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<(pre|code)\b[^>]*>[\s\S]*?<\/\1\s*>/gi, '')
  const output = []
  let fence = null

  for (const line of withoutComments.split(/\r?\n/)) {
    const marker = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/)
    if (fence !== null) {
      if (marker
        && marker[1][0] === fence[0]
        && marker[1].length >= fence.length
        && marker[2].trim() === '') {
        fence = null
      }
      output.push('')
      continue
    }

    if (marker && (marker[1][0] === '~' || !marker[2].includes('`'))) {
      fence = marker[1]
      output.push('')
      continue
    }

    output.push(/^(?: {4,}|\t)/.test(line) ? '' : stripInlineCode(line))
  }

  return output.join('\n')
}

async function checkSourceTopology() {
  const markdownFiles = (await walk(docsDir)).filter(file => file.endsWith('.md'))

  for (const markdownFile of markdownFiles) {
    const relativePath = toPosix(path.relative(projectRoot, markdownFile))
    const isChinese = relativePath.startsWith('docs/zh/')
    const markdown = stripMarkdownCode(await readFile(markdownFile, 'utf8'))
    const linkPatterns = [
      /!?\[[^\]]*\]\(\s*<?([^\s)>]+)>?(?:\s+["'][^"']*["'])?\s*\)/g,
      /^\s*\[[^\]]+\]:\s*<?([^\s>]+)>?/gm,
      /\bhref\s*=\s*["']([^"']+)["']/gi,
      /<(\/[^\s>]+)>/g,
    ]

    for (const pattern of linkPatterns) {
      for (const match of markdown.matchAll(pattern)) {
        const rawUrl = match[1].trim()
        if (!rawUrl.startsWith('/') || rawUrl.startsWith('//')) continue

        let sourcePath
        try {
          sourcePath = decodeURIComponent(new URL(rawUrl, localOrigin).pathname)
        } catch {
          errors.add(`${relativePath}: invalid source link: ${rawUrl}`)
          continue
        }

        if (sourcePath === '/en' || sourcePath.startsWith('/en/')) {
          errors.add(`${relativePath}: uses removed /en locale path: ${rawUrl}`)
        }

        if (isChinese && englishSectionPattern.test(sourcePath)) {
          errors.add(`${relativePath}: Chinese source links to the English content tree: ${rawUrl}`)
        }
      }
    }
  }
}

function canonicalUrl(route) {
  return new URL(withBase(route, canonicalBase), productionOrigin).href
}

function englishCounterpartRoute(route) {
  if (route === '/zh' || route === '/zh/') return '/'
  return route.startsWith('/zh/') ? route.slice('/zh'.length) : route
}

function chineseCounterpartRoute(route) {
  const englishRoute = englishCounterpartRoute(route)
  return englishRoute === '/' ? '/zh/' : `/zh${englishRoute}`
}

function routeVariants(route) {
  if (route === '/') return [route]
  return route.endsWith('/') ? [route, route.slice(0, -1)] : [route, `${route}/`]
}

function matchingMetaTags(metaTags, key) {
  return metaTags.filter(tag => [tag.property, tag.name]
    .some(attribute => attribute?.toLowerCase() === key))
}

function singleMetaContent(page, metaTags, key) {
  const matches = matchingMetaTags(metaTags, key)
  if (matches.length !== 1) {
    recordHtmlError(page, String(matches.length), `expected exactly one ${key} meta tag, found`)
    return null
  }
  return matches[0].content
}

function validateSeo(page, validTargets, pageByTarget) {
  if (page.relativePath === '404.html' || isGalleryContent(page.route)) return

  const expected = canonicalUrl(page.route)
  const linkTags = findTags(page.html, 'link')
  const metaTags = findTags(page.html, 'meta')
  const canonicalTags = linkTags.filter(tag => tag.rel?.toLowerCase() === 'canonical')

  if (canonicalTags.length !== 1) {
    recordHtmlError(page, String(canonicalTags.length), 'expected exactly one canonical link, found')
  } else if (canonicalTags[0].href !== expected) {
    recordHtmlError(page, canonicalTags[0].href, `canonical URL should be ${expected}`)
  }

  const englishRoute = englishCounterpartRoute(page.route)
  const chineseRoute = chineseCounterpartRoute(page.route)
  const expectedAlternates = new Map()
  if (pageByTarget.has(englishRoute)) {
    expectedAlternates.set('en-US', canonicalUrl(englishRoute))
    expectedAlternates.set('x-default', canonicalUrl(englishRoute))
  }
  if (pageByTarget.has(chineseRoute)) {
    expectedAlternates.set('zh-CN', canonicalUrl(chineseRoute))
  }

  const alternateTags = linkTags.filter(tag => tag.rel?.toLowerCase() === 'alternate' && tag.hreflang !== undefined)
  for (const alternate of alternateTags) {
    validateCanonicalTarget(page, alternate.href, validTargets, `hreflang ${alternate.hreflang}`)
    if (!expectedAlternates.has(alternate.hreflang)) {
      recordHtmlError(page, alternate.hreflang, 'unexpected hreflang value')
    }
  }

  for (const [language, expectedUrl] of expectedAlternates) {
    const matches = alternateTags.filter(tag => tag.hreflang === language)
    if (matches.length !== 1) {
      recordHtmlError(page, String(matches.length), `expected exactly one ${language} hreflang link, found`)
    } else if (matches[0].href !== expectedUrl) {
      recordHtmlError(page, matches[0].href, `${language} hreflang URL should be ${expectedUrl}`)
    }
  }

  const ogUrl = singleMetaContent(page, metaTags, 'og:url')
  if (ogUrl !== null && ogUrl !== expected) {
    recordHtmlError(page, ogUrl ?? '(missing)', `og:url should be ${expected}`)
  }

  for (const key of ['og:image', 'twitter:image']) {
    const image = singleMetaContent(page, metaTags, key)
    if (image === null) continue
    if (!image) {
      recordHtmlError(page, '(missing)', `${key} is required`)
      continue
    }
    validateCanonicalTarget(page, image, validTargets, key)
  }
}

function validateCanonicalTarget(page, rawUrl, validTargets, label) {
  let target
  try {
    target = new URL(rawUrl)
  } catch {
    recordHtmlError(page, rawUrl, `${label} is not an absolute URL`)
    return
  }

  if (target.origin !== productionOrigin) {
    recordHtmlError(page, rawUrl, `${label} uses the wrong origin`)
    return
  }

  const unpacked = unpackSitePath(target.pathname, canonicalBase)
  if (unpacked.error === 'invalid URL encoding') {
    recordHtmlError(page, rawUrl, `${label} has invalid URL encoding`)
    return
  }
  if (unpacked.error) {
    recordHtmlError(page, rawUrl, `${label} escapes canonical base ${canonicalBase}`)
    return
  }

  const { sitePath } = unpacked
  if (!validTargets.has(sitePath) && !isGalleryMount(sitePath)) {
    recordHtmlError(page, rawUrl, `${label} target does not exist`)
  }
}

function validateInternalUrl(page, rawUrl, validTargets, pageByTarget, forbiddenHtmlTargets) {
  // HTML files can contain markup inside JavaScript template literals. The
  // final URL is assembled at runtime, so it has no static document target.
  if (/\$\{[^}]+\}/.test(rawUrl)) return

  if (isIgnoredUrl(rawUrl)) return

  let target
  try {
    target = new URL(rawUrl, `${localOrigin}${withBase(page.route)}`)
  } catch {
    recordHtmlError(page, rawUrl, 'invalid URL')
    return
  }

  const isGeneratedInternal = target.origin === localOrigin
  const isProductionInternal = target.origin === productionOrigin
  if (!isGeneratedInternal && !isProductionInternal) return

  const selectedBase = isGeneratedInternal ? base : canonicalBase
  const unpacked = unpackSitePath(target.pathname, selectedBase)
  if (unpacked.error) {
    if (isGeneratedInternal && !options.requireGallery) {
      const directoryRoute = page.route.endsWith('/') ? page.route : `${page.route}/`
      const galleryCandidate = new URL(rawUrl, `${localOrigin}${withBase(directoryRoute)}`)
      const galleryTarget = unpackSitePath(galleryCandidate.pathname, base)
      if (!galleryTarget.error && isGalleryMount(galleryTarget.sitePath)) return
    }
    recordHtmlError(page, rawUrl, unpacked.error)
    return
  }
  const { sitePath } = unpacked

  if (forbiddenHtmlTargets.has(sitePath)) {
    recordHtmlError(page, rawUrl, 'uses an .html URL while cleanUrls is enabled')
    return
  }

  if (sitePath === '/en' || sitePath.startsWith('/en/')) {
    recordHtmlError(page, rawUrl, 'uses removed /en locale path')
    return
  }

  if ((page.route === '/zh' || page.route.startsWith('/zh/')) && englishSectionPattern.test(sitePath)) {
    const counterpart = englishCounterpartRoute(page.route)
    if (!routeVariants(counterpart).includes(sitePath)) {
      recordHtmlError(page, rawUrl, 'Chinese page links to the English content tree')
      return
    }
  }

  const galleryIsExternalMount = !options.requireGallery && isGalleryMount(sitePath)
  if (!validTargets.has(sitePath) && !galleryIsExternalMount) {
    recordHtmlError(page, rawUrl, 'target does not exist in the built site')
    return
  }

  if (!target.hash || galleryIsExternalMount) return
  const fragment = decodeUrlPart(target.hash.slice(1))
  if (fragment === null) {
    recordHtmlError(page, rawUrl, 'invalid fragment encoding')
    return
  }
  if (fragment === '') return

  const targetPage = pageByTarget.get(sitePath)
  if (targetPage && !targetPage.ids.has(fragment)) {
    recordHtmlError(page, rawUrl, `fragment #${fragment} does not exist on ${sitePath}`)
  }
}

function recordGalleryError(subject, message) {
  errors.add(`gallery/examples.json: ${message}: ${subject}`)
}

function requireGalleryFile(builtFiles, relativePath, label) {
  if (!builtFiles.has(relativePath)) {
    recordGalleryError(relativePath, `${label} is missing`)
  }
}

async function checkGalleryManifest(validTargets, builtFiles) {
  requireGalleryFile(builtFiles, 'gallery/index.html', 'Gallery entry point')
  requireGalleryFile(builtFiles, 'gallery/examples.json', 'Gallery manifest')
  if (!builtFiles.has('gallery/examples.json')) return

  let manifest
  try {
    manifest = JSON.parse(await readFile(path.join(distDir, 'gallery', 'examples.json'), 'utf8'))
  } catch (error) {
    recordGalleryError(error.message, 'manifest is not valid JSON')
    return
  }

  if (!Array.isArray(manifest)) {
    recordGalleryError(typeof manifest, 'manifest root must be an array')
    return
  }
  if (manifest.length === 0) {
    recordGalleryError('0', 'manifest must contain at least one category')
  }

  const categoryNames = new Set()
  const itemIds = new Set()

  for (const [categoryIndex, category] of manifest.entries()) {
    const categoryLabel = `category[${categoryIndex}]`
    if (category === null || typeof category !== 'object' || Array.isArray(category)) {
      recordGalleryError(categoryLabel, 'category must be an object')
      continue
    }

    if (typeof category.category !== 'string' || category.category.trim() === '') {
      recordGalleryError(categoryLabel, 'category name must be a non-empty string')
    } else if (categoryNames.has(category.category)) {
      recordGalleryError(category.category, 'category name is duplicated')
    } else {
      categoryNames.add(category.category)
    }

    if (!Array.isArray(category.items)) {
      recordGalleryError(categoryLabel, 'category items must be an array')
      continue
    }
    if (category.items.length === 0) {
      recordGalleryError(categoryLabel, 'category must contain at least one item')
    }

    for (const [itemIndex, item] of category.items.entries()) {
      const itemLabel = `${categoryLabel}.items[${itemIndex}]`
      if (item === null || typeof item !== 'object' || Array.isArray(item)) {
        recordGalleryError(itemLabel, 'item must be an object')
        continue
      }

      for (const field of ['id', 'name', 'description', 'source_path']) {
        if (typeof item[field] !== 'string' || item[field].trim() === '') {
          recordGalleryError(itemLabel, `${field} must be a non-empty string`)
        }
      }

      const safeId = typeof item.id === 'string'
        && /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(item.id)
      if (typeof item.id === 'string') {
        if (itemIds.has(item.id)) {
          recordGalleryError(item.id, 'item id is duplicated')
        } else {
          itemIds.add(item.id)
        }
      }
      if (!safeId) {
        recordGalleryError(item.id ?? itemLabel, 'item id is not a safe artifact name')
      }

      if (item.type !== 'iframe' && item.type !== 'standalone') {
        recordGalleryError(item.type ?? itemLabel, 'item type must be iframe or standalone')
      }
      if (typeof item.web_supported !== 'boolean') {
        recordGalleryError(itemLabel, 'web_supported must be a boolean')
      }

      if (item.web_supported !== true) continue

      if (item.type === 'iframe') {
        requireGalleryFile(builtFiles, 'gallery/viewer.html', `viewer for ${item.id}`)
        if (safeId) {
          requireGalleryFile(builtFiles, `gallery/wasm/${item.id}.js`, `JavaScript artifact for ${item.id}`)
          requireGalleryFile(builtFiles, `gallery/wasm/${item.id}_bg.wasm`, `WebAssembly artifact for ${item.id}`)
        }
        continue
      }

      if (item.type !== 'standalone') continue
      if (typeof item.url !== 'string' || item.url.trim() === '') {
        recordGalleryError(item.id ?? itemLabel, 'web standalone item must have a local url')
        continue
      }

      let target
      try {
        target = new URL(item.url, `${localOrigin}${withBase('/gallery/')}`)
      } catch {
        recordGalleryError(item.url, `standalone url for ${item.id} is invalid`)
        continue
      }
      if (target.origin !== localOrigin) {
        recordGalleryError(item.url, `standalone url for ${item.id} must be local`)
        continue
      }

      const unpacked = unpackSitePath(target.pathname, base)
      if (unpacked.error) {
        recordGalleryError(item.url, `standalone url for ${item.id} ${unpacked.error}`)
      } else if (!isGalleryContent(unpacked.sitePath)) {
        recordGalleryError(item.url, `standalone url for ${item.id} escapes the Gallery`)
      } else if (!validTargets.has(unpacked.sitePath)) {
        recordGalleryError(item.url, `standalone artifact for ${item.id} does not exist`)
      }
    }
  }
}

async function checkSitemap(validTargets, pages) {
  const sitemapPath = path.join(distDir, 'sitemap.xml')
  let sitemap
  try {
    sitemap = await readFile(sitemapPath, 'utf8')
  } catch {
    errors.add('sitemap.xml: missing from built site')
    return
  }

  const locations = [...sitemap.matchAll(/<loc>(.*?)<\/loc>/g)].map(match => match[1].replace(/&amp;/g, '&'))
  const sitemapRoutes = new Set()

  for (const location of locations) {
    let target
    try {
      target = new URL(location)
    } catch {
      errors.add(`sitemap.xml: invalid URL: ${location}`)
      continue
    }

    if (target.origin !== productionOrigin) {
      errors.add(`sitemap.xml: URL escapes ${productionOrigin}${canonicalBase}: ${location}`)
      continue
    }

    const unpacked = unpackSitePath(target.pathname, canonicalBase)
    if (unpacked.error) {
      errors.add(`sitemap.xml: URL escapes ${productionOrigin}${canonicalBase}: ${location}`)
      continue
    }

    const { sitePath } = unpacked
    sitemapRoutes.add(sitePath)

    if (sitePath === '/en' || sitePath.startsWith('/en/')) {
      errors.add(`sitemap.xml: uses removed /en locale path: ${location}`)
    } else if (!validTargets.has(sitePath)) {
      errors.add(`sitemap.xml: target does not exist in built site: ${location}`)
    }
  }

  for (const page of pages) {
    if (page.relativePath === '404.html' || isGalleryContent(page.route)) continue
    if (!sitemapRoutes.has(page.route)) {
      errors.add(`sitemap.xml: missing page route: ${page.route}`)
    }
  }
}

async function checkBuiltSite() {
  let files
  try {
    files = await walk(distDir)
  } catch (error) {
    throw new Error(`Cannot read built site at ${distDir}. Run the docs build first.\n${error.message}`)
  }

  const htmlFiles = files.filter(file => file.endsWith('.html'))
  if (htmlFiles.length === 0) {
    throw new Error(`No HTML files found in ${distDir}. Run the docs build first.`)
  }

  const validTargets = new Set(['/'])
  const forbiddenHtmlTargets = new Set()
  const builtFiles = new Set()
  const pageByTarget = new Map()
  const pages = []

  for (const file of files) {
    const relativePath = toPosix(path.relative(distDir, file))
    const rawTarget = `/${relativePath}`
    builtFiles.add(relativePath)
    if (!relativePath.endsWith('.html')
      || relativePath.startsWith('gallery/')
      || relativePath === '404.html') {
      validTargets.add(rawTarget)
    } else {
      forbiddenHtmlTargets.add(rawTarget)
    }
  }

  for (const htmlFile of htmlFiles) {
    const relativePath = path.relative(distDir, htmlFile)
    const html = await readFile(htmlFile, 'utf8')
    const ids = new Set()
    for (const match of html.matchAll(/\b(?:id|name)\s*=\s*["']([^"']+)["']/gi)) {
      const decoded = decodeUrlPart(match[1])
      ids.add(decoded ?? match[1])
    }

    const page = { relativePath: toPosix(relativePath), route: builtPageRoute(relativePath), html, ids }
    pages.push(page)
    for (const alias of pageTargetAliases(page)) {
      validTargets.add(alias)
      pageByTarget.set(alias, page)
    }
  }

  if (options.requireGallery) {
    await checkGalleryManifest(validTargets, builtFiles)
  }

  for (const page of pages) {
    const attributePattern = /\b(?:href|src)\s*=\s*["']([^"']+)["']/gi
    for (const match of page.html.matchAll(attributePattern)) {
      validateInternalUrl(page, match[1].trim(), validTargets, pageByTarget, forbiddenHtmlTargets)
    }
    validateSeo(page, validTargets, pageByTarget)
  }

  await checkSitemap(validTargets, pages)
}

await checkSourceTopology()
if (!options.sourceOnly) {
  await checkBuiltSite()
}

if (errors.size > 0) {
  console.error(`Documentation checks failed with ${errors.size} error(s):`)
  for (const error of [...errors].sort()) {
    console.error(`- ${error}`)
  }
  process.exitCode = 1
} else if (options.sourceOnly) {
  console.log('Documentation source topology passed.')
} else {
  const gallery = options.requireGallery ? ', Gallery required' : ''
  console.log(`Documentation checks passed (${base}, ${path.relative(projectRoot, distDir) || '.'}${gallery}).`)
}
