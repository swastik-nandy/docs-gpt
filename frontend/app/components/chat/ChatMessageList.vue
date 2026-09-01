<script setup lang="ts">
import {
  AlertCircle,
  Sparkles,
} from 'lucide-vue-next'

import MarkdownMessage from './MarkdownMessage.vue'

import type {
  ChatMessage,
} from '~/types/chat'


const props = defineProps<{
  messages: ChatMessage[]
}>()


const viewport = ref<
  HTMLElement
  | null
>(null)


watch(
  () => props.messages.map(
    message => (
      `${message.id}:${message.text.length}`
    ),
  ),

  async () => {
    await nextTick()

    viewport.value?.scrollTo({
      top:
        viewport.value.scrollHeight,

      behavior: 'smooth',
    })
  },

  {
    deep: true,
  },
)
</script>

<template>
  <div
    ref="viewport"
    class="message-viewport"
  >
    <TransitionGroup
      name="message"
      tag="div"
      class="message-list"
    >
      <article
        v-for="message in messages"
        :key="message.id"
        class="message-row"
        :class="message.role"
      >
        <div
          v-if="message.role === 'assistant'"
          class="assistant-mark"
        >
          <Sparkles
            :size="14"
            :stroke-width="1.9"
          />
        </div>

        <div
          class="message-content"
          :class="{
            streaming:
              message.status === 'streaming',
            errored:
              message.status === 'error',
          }"
        >
          <div
            v-if="
              message.role === 'assistant'
              && !message.text
              && message.status === 'streaming'
            "
            class="thinking"
          >
            <span />
            <span />
            <span />
          </div>

          <div
            v-else
            class="message-text"
          >
            <MarkdownMessage
              v-if="message.role === 'assistant'"
              :text="message.text"
              :streaming="
                message.status === 'streaming'
              "
            />

            <template v-else>
              {{ message.text }}
            </template>
          </div>

          <div
            v-if="message.status === 'error'"
            class="error-label"
          >
            <AlertCircle :size="12" />

            Response interrupted
          </div>
        </div>
      </article>
    </TransitionGroup>
  </div>
</template>

<style scoped>
.message-viewport {
  position: absolute;
  inset: 0;
  overflow-x: hidden;
  overflow-y: auto;
  padding: 36px 24px 190px;
  scroll-behavior: smooth;
}

.message-list {
  display: flex;
  width: min(760px, 100%);
  margin: 0 auto;
  flex-direction: column;
  gap: 30px;
}

.message-row {
  display: flex;
  width: 100%;
  gap: 12px;
}

.message-row.user {
  justify-content: flex-end;
}

.message-row.assistant {
  align-items: flex-start;
}

.message-content {
  min-width: 0;
  color: var(--text-soft);
  font-size: 14px;
  font-weight: 420;
  line-height: 1.72;
}

.user .message-content {
  max-width: 76%;
  padding: 10px 14px;
  border: 1px solid rgba(94, 233, 181, 0.1);
  border-radius:
    14px
    14px
    4px
    14px;
  background:
    linear-gradient(
      145deg,
      rgba(31, 66, 53, 0.65),
      rgba(18, 39, 32, 0.62)
    );
  color: #dcece4;
  box-shadow:
    0 8px 28px rgba(0, 0, 0, 0.12);
}

.assistant .message-content {
  width: calc(100% - 40px);
  padding-top: 4px;
}

.assistant-mark {
  display: grid;
  width: 29px;
  height: 29px;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid rgba(94, 233, 181, 0.13);
  border-radius: 8px;
  background: rgba(54, 211, 153, 0.055);
  color: var(--green);
}

.message-text {
  white-space: pre-wrap;
  word-break: break-word;
}

.stream-cursor {
  display: inline-block;
  width: 5px;
  height: 15px;
  margin-left: 3px;
  border-radius: 2px;
  vertical-align: -2px;
  background: var(--green);
  animation:
    cursor-pulse
    900ms
    ease-in-out
    infinite;
}

.thinking {
  display: flex;
  height: 26px;
  align-items: center;
  gap: 4px;
}

.thinking span {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--green-soft);
  animation:
    thinking
    1.1s
    ease-in-out
    infinite;
}

.thinking span:nth-child(2) {
  animation-delay: 120ms;
}

.thinking span:nth-child(3) {
  animation-delay: 240ms;
}

.error-label {
  display: flex;
  align-items: center;
  gap: 5px;
  margin-top: 8px;
  color: #c97983;
  font-size: 10px;
}

.message-enter-active {
  transition:
    opacity 300ms ease,
    transform 360ms
    cubic-bezier(.2,.8,.2,1);
}

.message-enter-from {
  opacity: 0;
  transform: translateY(8px);
}

@keyframes cursor-pulse {
  0%,
  100% {
    opacity: 0.25;
  }

  50% {
    opacity: 1;
  }
}

@keyframes thinking {
  0%,
  60%,
  100% {
    opacity: 0.25;
    transform: translateY(0);
  }

  30% {
    opacity: 1;
    transform: translateY(-3px);
  }
}
</style>
