<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'

// The VP9 WebM carries a real alpha channel. Browsers that cannot reliably
// render VP9 alpha (notably Safari) keep the transparent PNG poster instead.
const videoSrcWebm = withBase('/media/demo.webm')
const poster = withBase('/images/hero.png')

const shouldAnimate = ref(false)
let motionPreference: MediaQueryList | null = null

function supportsTransparentHeroVideo() {
  const userAgent = navigator.userAgent
  const isSafari = /Safari/i.test(userAgent) && !/(Chrome|Chromium|CriOS|Edg|OPR|Android)/i.test(userAgent)
  const video = document.createElement('video')
  const supportsVp9 = video.canPlayType('video/webm; codecs="vp9"') !== ''
  return supportsVp9 && !isSafari
}

function updateMotionPreference() {
  shouldAnimate.value = Boolean(
    motionPreference && !motionPreference.matches && supportsTransparentHeroVideo()
  )
}

onMounted(() => {
  motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)')
  updateMotionPreference()
  motionPreference.addEventListener?.('change', updateMotionPreference)
})

onBeforeUnmount(() => {
  motionPreference?.removeEventListener?.('change', updateMotionPreference)
  motionPreference = null
})
</script>

<template>
  <div class="hero-video" aria-hidden="true">
    <div class="hero-video__glow"></div>
    <img
      class="hero-video__media hero-video__poster"
      :src="poster"
      alt=""
      width="768"
      height="768"
    >
    <video
      v-if="shouldAnimate"
      class="hero-video__media"
      :poster="poster"
      aria-hidden="true"
      tabindex="-1"
      autoplay
      muted
      loop
      playsinline
      preload="metadata"
      @error="shouldAnimate = false"
    >
      <source :src="videoSrcWebm" type='video/webm; codecs="vp9"'>
    </video>
  </div>
</template>

<style scoped>
.hero-video {
  position: relative;
  width: 100%;
  aspect-ratio: 1 / 1;
  margin: 0 auto;
  background: transparent;
}

/* The media has a transparent background, so no frame/shadow/fill — it must
   blend straight into the page in both light and dark mode. */
.hero-video__media {
  position: absolute;
  inset: 0;
  z-index: 1;
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: transparent;
}

.hero-video__poster {
  z-index: 0;
}

/* Soft brand-colored aura behind the character, echoing the VitePress hero
   blur. Kept subtle so it reads well on both themes. */
.hero-video__glow {
  position: absolute;
  inset: 8% 6% 2%;
  z-index: 0;
  border-radius: 50%;
  filter: blur(80px);
  opacity: 0.4;
  background-image: linear-gradient(
    -45deg,
    var(--vp-c-brand-3) 30%,
    var(--vp-c-brand-1) 70%
    /* #bd34fe 50%,
    #47caff 50% */
  );
}

@media (prefers-reduced-motion: reduce) {
  .hero-video__glow {
    filter: blur(64px);
    opacity: 0.28;
  }
}
</style>
