<script setup lang="ts">
const props = defineProps<{
  text: string
  streaming?: boolean
}>()


const isStreaming = computed(
  () => Boolean(
    props.streaming,
  ),
)


const displayText = ref(
  props.text,
)

const hasSoftTail = computed(
  () => (
    isStreaming.value
    && displayText.value.length > 48
  ),
)


let animationFrame:
  number
  | null = null


function revealCount(
  remaining: number,
): number {
  if (!isStreaming.value) {
    if (remaining > 160) {
      return 36
    }

    if (remaining > 60) {
      return 18
    }

    return 10
  }

  if (remaining > 500) {
    return 32
  }

  if (remaining > 220) {
    return 18
  }

  if (remaining > 100) {
    return 10
  }

  if (remaining > 40) {
    return 6
  }

  if (remaining > 16) {
    return 4
  }

  return 2
}


function cancelReveal() {
  if (
    animationFrame === null
    || !import.meta.client
  ) {
    return
  }

  window.cancelAnimationFrame(
    animationFrame,
  )

  animationFrame = null
}


function revealNextFrame() {
  animationFrame = null

  const target = props.text
  const current = displayText.value

  if (current === target) {
    return
  }

  if (!target.startsWith(current)) {
    displayText.value = target
    return
  }

  const remaining = (
    target.length
    - current.length
  )

  const count = revealCount(
    remaining,
  )

  displayText.value = target.slice(
    0,
    current.length + count,
  )

  scheduleReveal()
}


function scheduleReveal() {
  if (!import.meta.client) {
    displayText.value = props.text
    return
  }

  if (animationFrame !== null) {
    return
  }

  if (
    displayText.value
    === props.text
  ) {
    return
  }

  animationFrame = (
    window.requestAnimationFrame(
      revealNextFrame,
    )
  )
}


watch(
  () => props.text,
  () => {
    scheduleReveal()
  },
)


watch(
  () => props.streaming,
  () => {
    scheduleReveal()
  },
)


onMounted(
  () => {
    scheduleReveal()
  },
)


onBeforeUnmount(
  () => {
    cancelReveal()
  },
)
</script>

<template>
  <div
    class="markdown-message"
    :class="{
      streaming: isStreaming,
      'soft-tail': hasSoftTail,
    }"
  >
    <Markdown
      :value="displayText"
      :streaming="isStreaming"
      :caret="isStreaming"
    />
  </div>
</template>

<style scoped>
.markdown-message {
  position: relative;
  min-width: 0;
  color: var(--text-soft);
  overflow-wrap: anywhere;
}


/*------------------------STREAMING-------------------------*/

.markdown-message.streaming {
  animation:
    stream-message-enter
    240ms
    cubic-bezier(.2,.8,.2,1)
    both;
}

.markdown-message.soft-tail {
  -webkit-mask-image:
    linear-gradient(
      to bottom,
      #000 0,
      #000 calc(100% - 14px),
      rgba(0, 0, 0, 0.72) 100%
    );

  mask-image:
    linear-gradient(
      to bottom,
      #000 0,
      #000 calc(100% - 14px),
      rgba(0, 0, 0, 0.72) 100%
    );
}

.markdown-message.streaming::after {
  content: '';
  position: absolute;
  right: -10px;
  bottom: -3px;
  width: 74px;
  height: 24px;
  border-radius: 50%;
  background:
    radial-gradient(
      ellipse at center,
      rgba(54, 211, 153, 0.09),
      transparent 68%
    );
  filter: blur(7px);
  opacity: 0.55;
  pointer-events: none;
  animation:
    stream-tail-glow
    1.4s
    ease-in-out
    infinite;
}


@keyframes stream-message-enter {
  from {
    opacity: 0;
    filter: blur(2px);
    transform: translateY(3px);
  }

  to {
    opacity: 1;
    filter: blur(0);
    transform: translateY(0);
  }
}


@keyframes stream-tail-glow {
  0%,
  100% {
    opacity: 0.2;
    transform:
      translateX(-3px)
      scaleX(0.9);
  }

  50% {
    opacity: 0.62;
    transform:
      translateX(3px)
      scaleX(1.08);
  }
}


/*------------------------PARAGRAPHS-------------------------*/

.markdown-message :deep(p) {
  margin: 0 0 12px;
  color: var(--text-soft);
  font-size: 14px;
  font-weight: 420;
  line-height: 1.72;
}

.markdown-message :deep(p:last-child) {
  margin-bottom: 0;
}


/*------------------------STRONG-------------------------*/

.markdown-message :deep(strong) {
  color: var(--text);
  font-weight: 650;
  letter-spacing: -0.008em;
}


/*------------------------EMPHASIS-------------------------*/

.markdown-message :deep(em) {
  color: #c5d6cd;
}


/*------------------------LISTS-------------------------*/

.markdown-message :deep(ul),
.markdown-message :deep(ol) {
  margin: 10px 0 15px;
  padding-left: 23px;
}

.markdown-message :deep(li) {
  margin: 7px 0;
  padding-left: 2px;
  color: var(--text-soft);
  line-height: 1.68;
}

.markdown-message :deep(li::marker) {
  color: var(--green);
}

.markdown-message :deep(li > p) {
  margin: 0;
}


/*------------------------HEADINGS-------------------------*/

.markdown-message :deep(h1),
.markdown-message :deep(h2),
.markdown-message :deep(h3),
.markdown-message :deep(h4) {
  color: var(--text);
  font-weight: 620;
  letter-spacing: -0.025em;
}

.markdown-message :deep(h1) {
  margin: 20px 0 10px;
  font-size: 22px;
}

.markdown-message :deep(h2) {
  margin: 18px 0 9px;
  font-size: 18px;
}

.markdown-message :deep(h3) {
  margin: 16px 0 8px;
  font-size: 15px;
}


/*------------------------INLINE CODE-------------------------*/

.markdown-message :deep(code) {
  padding: 2px 5px;
  border: 1px solid var(--border);
  border-radius: 5px;
  background: var(--surface-2);
  color: #a7dfc7;
  font-family: var(--font-mono);
  font-size: 0.88em;
}


/*------------------------CODE BLOCKS-------------------------*/

.markdown-message :deep(pre) {
  margin: 14px 0;
  padding: 14px;
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: #090d0c;
}

.markdown-message :deep(pre code) {
  padding: 0;
  border: 0;
  background: transparent;
  color: #c5d5cd;
}


/*------------------------BLOCKQUOTES-------------------------*/

.markdown-message :deep(blockquote) {
  margin: 14px 0;
  padding: 3px 0 3px 13px;
  border-left:
    2px solid
    rgba(54, 211, 153, 0.45);
  color: var(--text-muted);
}


/*------------------------LINKS-------------------------*/

.markdown-message :deep(a) {
  color: #72dcb1;
  text-decoration-color:
    rgba(114, 220, 177, 0.35);
  text-underline-offset: 3px;
}
</style>
