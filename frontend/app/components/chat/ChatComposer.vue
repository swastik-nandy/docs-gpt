<script setup lang="ts">
import {
  ArrowUp,
  LoaderCircle,
  Paperclip,
  Sparkles,
} from 'lucide-vue-next'


const props = defineProps<{
  busy?: boolean
}>()


const emit = defineEmits<{
  submit: [message: string]
}>()


const message = ref('')
const textarea = ref<
  HTMLTextAreaElement
  | null
>(null)


function resizeTextarea() {
  const element = textarea.value

  if (!element) {
    return
  }

  element.style.height = 'auto'

  element.style.height = (
    `${Math.min(
      element.scrollHeight,
      180,
    )}px`
  )
}


watch(
  message,
  async () => {
    await nextTick()
    resizeTextarea()
  },
)


function submit() {
  const text = message.value.trim()

  if (
    !text
    || props.busy
  ) {
    return
  }

  emit(
    'submit',
    text,
  )

  message.value = ''
}
</script>

<template>
  <div class="composer-shell">
    <div class="composer">
      <textarea
        ref="textarea"
        v-model="message"
        rows="1"
        placeholder="Ask Docs GPT anything about your knowledge..."
        @keydown.enter.exact.prevent="submit"
      />

      <div class="composer-actions">
        <div class="composer-left">
          <button
            class="composer-icon"
            type="button"
          >
            <Paperclip :size="16" />
          </button>

          <div class="context-pill">
            <Sparkles :size="12" />

            <span>
              Knowledge grounded
            </span>
          </div>
        </div>

        <button
          class="send-button"
          type="button"
          :disabled="
            !message.trim()
            || busy
          "
          @click="submit"
        >
          <LoaderCircle
            v-if="busy"
            :size="15"
            class="spinner"
          />

          <ArrowUp
            v-else
            :size="16"
            :stroke-width="2.2"
          />
        </button>
      </div>
    </div>

    <p class="composer-hint">
      Enter to send · Shift + Enter for a new line
    </p>
  </div>
</template>

<style scoped>
.composer-shell {
  width: min(760px, calc(100% - 48px));
  margin: 0 auto;
}

.composer {
  padding: 13px 13px 10px;
  border: 1px solid var(--border-strong);
  border-radius: 15px;
  background:
    linear-gradient(
      180deg,
      rgba(18, 27, 24, 0.97),
      rgba(12, 18, 16, 0.99)
    );
  box-shadow:
    0 24px 80px rgba(0, 0, 0, 0.3),
    inset 0 1px rgba(255, 255, 255, 0.025);
  transition:
    border-color 180ms ease,
    box-shadow 180ms ease;
}

.composer:focus-within {
  border-color: rgba(94, 233, 181, 0.25);
  box-shadow:
    0 24px 80px rgba(0, 0, 0, 0.3),
    0 0 0 3px rgba(54, 211, 153, 0.035);
}

textarea {
  display: block;
  width: 100%;
  min-height: 48px;
  max-height: 180px;
  resize: none;
  overflow-y: auto;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--text);
  font-size: 14px;
  font-weight: 430;
  line-height: 1.55;
}

textarea::placeholder {
  color: #5d6b64;
}

.composer-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.composer-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.composer-icon {
  display: grid;
  width: 30px;
  height: 30px;
  cursor: pointer;
  place-items: center;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: var(--text-muted);
}

.composer-icon:hover {
  background: var(--surface-3);
  color: var(--text-soft);
}

.context-pill {
  display: flex;
  align-items: center;
  gap: 5px;
  color: var(--text-faint);
  font-size: 10px;
}

.context-pill svg {
  color: var(--green-soft);
}

.send-button {
  display: grid;
  width: 32px;
  height: 32px;
  cursor: pointer;
  place-items: center;
  border: 1px solid rgba(94, 233, 181, 0.25);
  border-radius: 9px;
  background: var(--green);
  color: #06271d;
  box-shadow:
    0 4px 18px rgba(54, 211, 153, 0.14);
  transition:
    transform 150ms ease,
    opacity 150ms ease,
    background 150ms ease;
}

.send-button:hover:not(:disabled) {
  background: var(--green-bright);
  transform: translateY(-1px);
}

.send-button:disabled {
  cursor: default;
  opacity: 0.28;
}

.spinner {
  animation: spin 850ms linear infinite;
}

.composer-hint {
  margin-top: 8px;
  color: var(--text-faint);
  font-size: 9.5px;
  text-align: center;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
