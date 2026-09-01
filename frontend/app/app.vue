<script setup lang="ts">
import {
  CircleHelp,
  Command,
  Settings,
} from 'lucide-vue-next'


useHead({
  htmlAttrs: {
    lang: 'en',
  },
})


const {
  sources,
  refresh: refreshKnowledge,
} = useKnowledgeSources()


const {
  conversationId,
  messages,
  hasMessages,
  isGenerating,
  connectionState,
  sendMessage,
  closeSocket,
} = useChat()


const sourceFileInput = ref<HTMLInputElement | null>(
  null,
)


const {
  uploading: uploadingSources,
  uploadSources,
} = useSourceUpload()


function openUpload() {
  if (uploadingSources.value) {
    return
  }

  sourceFileInput.value?.click()
}


type UploadStage =
  | 'uploading'
  | 'indexing'
  | 'building'
  | 'queued'


const uploadOverlayOpen = ref(false)

const uploadStage = ref<UploadStage>(
  'uploading',
)


function delay(
  milliseconds: number,
) {
  return new Promise<void>(
    resolve => setTimeout(
      resolve,
      milliseconds,
    ),
  )
}


async function handleSourceSelection(
  event: Event,
) {
  const input = event.target as HTMLInputElement

  const files = Array.from(
    input.files ?? [],
  )

  input.value = ''

  if (files.length === 0) {
    return
  }

  const startedAt = Date.now()

  uploadStage.value = 'uploading'
  uploadOverlayOpen.value = true

  const indexingTimer = window.setTimeout(
    () => {
      uploadStage.value = 'indexing'
    },
    1600,
  )

  const buildingTimer = window.setTimeout(
    () => {
      uploadStage.value = 'building'
    },
    3400,
  )

  try {
    const result = await uploadSources(
      files,
    )

    console.info(
      'Sources queued:',
      result,
    )

    const minimumDuration = 5600

    const elapsed = (
      Date.now() - startedAt
    )

    if (elapsed < minimumDuration) {
      await delay(
        minimumDuration - elapsed,
      )
    }

    uploadStage.value = 'queued'

    await delay(700)

    uploadOverlayOpen.value = false

    await refreshKnowledge()
  }

  catch (error) {
    window.clearTimeout(
      indexingTimer,
    )

    window.clearTimeout(
      buildingTimer,
    )

    uploadOverlayOpen.value = false

    const message = (
      error instanceof Error
        ? error.message
        : 'Unable to upload sources.'
    )

    window.alert(message)
  }

  finally {
    window.clearTimeout(
      indexingTimer,
    )

    window.clearTimeout(
      buildingTimer,
    )
  }
}


const statusLabel = computed(
  () => {
    if (isGenerating.value) {
      return 'Thinking'
    }

    if (
      connectionState.value
      === 'connecting'
    ) {
      return 'Connecting'
    }

    if (
      connectionState.value
      === 'error'
    ) {
      return 'Connection issue'
    }

    if (
      connectionState.value
      === 'connected'
    ) {
      return 'Connected'
    }

    return 'Ready'
  },
)


onBeforeUnmount(
  () => {
    closeSocket()
  },
)
</script>

<template>
  <div class="app-shell">
    <KnowledgeBuildOverlay
      :open="uploadOverlayOpen"
      :stage="uploadStage"
    />

    <input
      ref="sourceFileInput"
      class="source-file-input"
      type="file"
      accept=".md,.mdx"
      multiple
      @change="handleSourceSelection"
    >

    <KnowledgeSidebar
      :sources="sources"
      @refresh="refreshKnowledge"
      @upload="openUpload"
    />

    <main class="chat-shell">
      <header class="chat-header">
        <div class="header-copy">
          <span class="conversation-label">
            {{
              conversationId
                ? 'Conversation'
                : 'New conversation'
            }}
          </span>

          <span class="conversation-status">
            <span
              class="status-dot"
              :class="{
                thinking: isGenerating,
              }"
            />

            {{ statusLabel }}
          </span>
        </div>

        <div class="header-actions">
          <button
            class="header-button"
            type="button"
          >
            <CircleHelp :size="15" />
          </button>

          <button
            class="header-button"
            type="button"
          >
            <Settings :size="15" />
          </button>

          <div class="keyboard-hint">
            <Command :size="11" />
            K
          </div>
        </div>
      </header>

      <section class="chat-stage">
        <div class="ambient-grid" />

        <div
          v-if="!hasMessages"
          class="welcome-position"
        >
          <ChatWelcome
            @select="sendMessage"
          />
        </div>

        <ChatMessageList
          v-else
          :messages="messages"
        />

        <div class="composer-position">
          <ChatComposer
            :busy="isGenerating"
            @submit="sendMessage"
          />
        </div>
      </section>
    </main>
  </div>
</template>

<style scoped>
.source-file-input {
  display: none;
}

.app-shell {
  display: flex;
  width: 100vw;
  height: 100dvh;
  overflow: hidden;
}

.chat-shell {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
  background:
    radial-gradient(
      circle at 52% 8%,
      rgba(45, 212, 163, 0.045),
      transparent 34%
    ),
    var(--bg);
}

.chat-header {
  position: relative;
  z-index: 4;
  display: flex;
  height: 62px;
  flex: 0 0 auto;
  align-items: center;
  justify-content: space-between;
  padding: 0 21px;
  border-bottom:
    1px solid
    rgba(151, 190, 171, 0.07);
  background:
    rgba(8, 12, 11, 0.72);
  backdrop-filter: blur(15px);
}

.header-copy {
  display: flex;
  align-items: center;
  gap: 12px;
}

.conversation-label {
  color: var(--text-soft);
  font-size: 12px;
  font-weight: 560;
}

.conversation-status {
  display: flex;
  align-items: center;
  gap: 5px;
  color: var(--text-faint);
  font-size: 9.5px;
}

.status-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--green);
  box-shadow:
    0 0 8px
    rgba(54, 211, 153, 0.55);
}

.status-dot.thinking {
  animation:
    status-pulse
    1.2s
    ease-in-out
    infinite;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 5px;
}

.header-button {
  display: grid;
  width: 30px;
  height: 30px;
  cursor: pointer;
  place-items: center;
  border: 1px solid transparent;
  border-radius: 8px;
  background: transparent;
  color: var(--text-muted);
}

.header-button:hover {
  border-color: var(--border);
  background: var(--surface-1);
  color: var(--text-soft);
}

.keyboard-hint {
  display: flex;
  height: 25px;
  align-items: center;
  gap: 3px;
  margin-left: 5px;
  padding: 0 7px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-1);
  color: var(--text-faint);
  font-family: var(--font-mono);
  font-size: 9px;
}

.chat-stage {
  position: relative;
  min-height: 0;
  flex: 1;
  overflow: hidden;
}

.ambient-grid {
  position: absolute;
  inset: 0;
  opacity: 0.22;
  background-image:
    linear-gradient(
      rgba(135, 170, 154, 0.025) 1px,
      transparent 1px
    ),
    linear-gradient(
      90deg,
      rgba(135, 170, 154, 0.025) 1px,
      transparent 1px
    );
  background-size: 44px 44px;
  mask-image:
    radial-gradient(
      ellipse at center,
      black,
      transparent 72%
    );
  pointer-events: none;
}

.welcome-position {
  position: absolute;
  inset: 0 0 150px;
  display: grid;
  place-items: center;
}

.composer-position {
  position: absolute;
  z-index: 3;
  right: 0;
  bottom: 22px;
  left: 0;
}

@keyframes status-pulse {
  0%,
  100% {
    opacity: 0.35;
    box-shadow:
      0 0 5px
      rgba(54, 211, 153, 0.3);
  }

  50% {
    opacity: 1;
    box-shadow:
      0 0 13px
      rgba(54, 211, 153, 0.7);
  }
}
</style>
