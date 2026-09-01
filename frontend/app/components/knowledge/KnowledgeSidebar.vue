<script setup lang="ts">
import {
  BookOpen,
  FileText,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
} from 'lucide-vue-next'

import type { KnowledgeSource } from '~/types/knowledge'

const props = defineProps<{
  sources: KnowledgeSource[]
}>()

defineEmits<{
  refresh: []
  upload: []
}>()


const searchQuery = ref('')

const filteredSources = computed(
  () => {
    const query = searchQuery.value
      .trim()
      .toLowerCase()

    if (!query) {
      return props.sources
    }

    return props.sources.filter(
      source => (
        source.filename
          .toLowerCase()
          .includes(query)
      ),
    )
  },
)
</script>

<template>
  <aside class="knowledge-sidebar">
    <div class="brand-row">
      <div class="brand-mark">
        <Sparkles :size="16" :stroke-width="2" />
      </div>

      <div class="brand-copy">
        <span class="brand-name">
          Docs GPT
        </span>

        <span class="brand-label">
          Knowledge Assistant
        </span>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="knowledge-heading">
        <div>
          <div class="eyebrow">
            Knowledge
          </div>

          <div class="knowledge-title-row">
            <h2>
              Sources
            </h2>

            <span class="source-count">
              {{ sources.length }}
            </span>
          </div>
        </div>

        <button
          class="icon-button"
          type="button"
          title="Refresh knowledge"
          @click="$emit('refresh')"
        >
          <RefreshCw :size="15" />
        </button>
      </div>

      <div class="search-shell">
        <Search :size="14" />

        <input
          v-model="searchQuery"
          type="text"
          placeholder="Search sources"
        >
      </div>
    </div>

    <div class="indexed-summary">
      <div class="summary-icon">
        <BookOpen :size="16" />
      </div>

      <div class="summary-copy">
        <span class="summary-value">
          {{ sources.length }} indexed
        </span>

        <span class="summary-label">
          Available to the assistant
        </span>
      </div>

      <span class="live-dot" />
    </div>

    <div class="source-scroll">
      <TransitionGroup
        name="knowledge-card"
        tag="div"
        class="source-stack"
      >
        <button
          v-for="source in filteredSources"
          :key="source.id"
          class="source-card"
          type="button"
        >
          <div class="file-icon">
            <FileText
              :size="15"
              :stroke-width="1.8"
            />
          </div>

          <div class="file-content">
            <div class="file-name">
              {{ source.filename }}
            </div>

            <div class="file-meta">
              <span>
                {{ source.chunks }} chunks
              </span>

              <span class="meta-dot" />

              <span>
                Indexed
              </span>
            </div>
          </div>

          <span class="source-status" />
        </button>
      </TransitionGroup>

      <div
        v-if="filteredSources.length === 0"
        class="empty-source-state"
      >
        No matching sources
      </div>
    </div>

    <div class="sidebar-footer">
      <button
        type="button"
        class="add-source-button"
        @click="$emit('upload')"
      >
        <Plus :size="16" />

        <span>
          Add sources
        </span>
      </button>

      <div class="footer-note">
        Markdown and MDX
      </div>
    </div>
  </aside>
</template>

<style scoped>
.knowledge-sidebar {
  display: flex;
  width: var(--sidebar-width);
  min-width: var(--sidebar-width);
  height: 100dvh;
  flex-direction: column;
  border-right: 1px solid var(--border);
  background:
    linear-gradient(
      180deg,
      rgba(13, 19, 17, 0.98),
      rgba(8, 13, 11, 0.98)
    );
  box-shadow:
    22px 0 70px rgba(0, 0, 0, 0.12);
}

.brand-row {
  display: flex;
  height: 76px;
  align-items: center;
  gap: 11px;
  padding: 0 20px;
  border-bottom: 1px solid rgba(151, 190, 171, 0.07);
}

.brand-mark {
  display: grid;
  width: 34px;
  height: 34px;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid rgba(94, 233, 181, 0.18);
  border-radius: 10px;
  background:
    linear-gradient(
      145deg,
      rgba(54, 211, 153, 0.17),
      rgba(54, 211, 153, 0.04)
    );
  color: var(--green-bright);
  box-shadow:
    inset 0 1px rgba(255, 255, 255, 0.05),
    0 8px 24px rgba(0, 0, 0, 0.18);
}

.brand-copy {
  display: flex;
  min-width: 0;
  flex-direction: column;
}

.brand-name {
  color: var(--text);
  font-size: 15px;
  font-weight: 650;
  letter-spacing: -0.025em;
}

.brand-label {
  margin-top: 1px;
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 450;
}

.sidebar-section {
  padding: 20px 16px 12px;
}

.knowledge-heading {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  padding: 0 3px 13px;
}

.eyebrow {
  margin-bottom: 4px;
  color: var(--text-faint);
  font-size: 10px;
  font-weight: 650;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.knowledge-title-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.knowledge-title-row h2 {
  color: var(--text);
  font-size: 15px;
  font-weight: 590;
}

.source-count {
  display: grid;
  min-width: 22px;
  height: 19px;
  place-items: center;
  padding: 0 6px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 600;
}

.icon-button {
  display: grid;
  width: 30px;
  height: 30px;
  cursor: pointer;
  place-items: center;
  border: 1px solid transparent;
  border-radius: 8px;
  background: transparent;
  color: var(--text-muted);
  transition:
    color 160ms ease,
    background 160ms ease,
    border-color 160ms ease,
    transform 160ms ease;
}

.icon-button:hover {
  border-color: var(--border);
  background: var(--surface-2);
  color: var(--green-bright);
}

.icon-button:active {
  transform: scale(0.94);
}

.search-shell {
  display: flex;
  height: 36px;
  align-items: center;
  gap: 9px;
  padding: 0 11px;
  border: 1px solid var(--border);
  border-radius: 9px;
  background: rgba(8, 13, 11, 0.65);
  color: var(--text-faint);
  transition:
    border-color 160ms ease,
    background 160ms ease,
    box-shadow 160ms ease;
}

.search-shell:focus-within {
  border-color: rgba(94, 233, 181, 0.23);
  background: var(--surface-1);
  box-shadow:
    0 0 0 3px rgba(54, 211, 153, 0.05);
}

.search-shell input {
  width: 100%;
  min-width: 0;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--text-soft);
  font-size: 12px;
}

.search-shell input::placeholder {
  color: var(--text-faint);
}

.indexed-summary {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 2px 16px 12px;
  padding: 11px;
  border: 1px solid rgba(54, 211, 153, 0.1);
  border-radius: 10px;
  background:
    linear-gradient(
      135deg,
      rgba(54, 211, 153, 0.07),
      rgba(54, 211, 153, 0.015)
    );
}

.summary-icon {
  display: grid;
  width: 30px;
  height: 30px;
  flex: 0 0 auto;
  place-items: center;
  border-radius: 8px;
  background: rgba(54, 211, 153, 0.08);
  color: var(--green);
}

.summary-copy {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
}

.summary-value {
  color: var(--text-soft);
  font-size: 11px;
  font-weight: 590;
}

.summary-label {
  margin-top: 2px;
  overflow: hidden;
  color: var(--text-faint);
  font-size: 10px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.live-dot,
.source-status {
  display: block;
  width: 6px;
  height: 6px;
  flex: 0 0 auto;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 9px rgba(54, 211, 153, 0.55);
}

.source-scroll {
  position: relative;
  width: 100%;
  min-height: 0;
  flex: 1 1 auto;
  overflow-x: hidden;
  overflow-y: auto;
  overscroll-behavior: contain;
  scrollbar-gutter: stable;
}

.source-stack {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 0 12px 24px;
}

.source-card {
  position: relative;
  display: flex;
  width: 100%;
  min-width: 0;
  cursor: pointer;
  align-items: center;
  gap: 10px;
  padding: 9px 10px;
  overflow: hidden;
  border: 1px solid transparent;
  border-radius: 9px;
  background: transparent;
  text-align: left;
  transition:
    border-color 180ms ease,
    background 180ms ease,
    transform 180ms cubic-bezier(.2,.8,.2,1),
    box-shadow 180ms ease;
}

.source-card:hover {
  border-color: var(--border);
  background:
    linear-gradient(
      90deg,
      rgba(54, 211, 153, 0.045),
      rgba(17, 25, 23, 0.72)
    );
  box-shadow:
    inset 2px 0 rgba(54, 211, 153, 0.45);
  transform: translateX(2px);
}

.file-icon {
  display: grid;
  width: 29px;
  height: 29px;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--surface-2);
  color: #68b99a;
}

.file-content {
  min-width: 0;
  flex: 1;
}

.file-name {
  overflow: hidden;
  color: var(--text-soft);
  font-size: 11.5px;
  font-weight: 510;
  letter-spacing: -0.01em;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 3px;
  color: var(--text-faint);
  font-size: 9.5px;
}

.meta-dot {
  width: 2px;
  height: 2px;
  border-radius: 50%;
  background: var(--text-faint);
}

.source-status {
  width: 5px;
  height: 5px;
  opacity: 0.72;
}

.empty-source-state {
  margin: 20px 16px;
  padding: 22px 12px;
  border: 1px dashed var(--border);
  border-radius: 10px;
  color: var(--text-faint);
  font-size: 11px;
  text-align: center;
}

.sidebar-footer {
  padding: 12px 16px 17px;
  border-top: 1px solid rgba(151, 190, 171, 0.07);
}

.add-source-button {
  display: flex;
  width: 100%;
  height: 38px;
  cursor: pointer;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border: 1px solid rgba(94, 233, 181, 0.2);
  border-radius: 9px;
  background:
    linear-gradient(
      180deg,
      rgba(54, 211, 153, 0.14),
      rgba(54, 211, 153, 0.08)
    );
  color: #bdf9dd;
  font-size: 11.5px;
  font-weight: 580;
  box-shadow:
    inset 0 1px rgba(255, 255, 255, 0.035);
  transition:
    background 160ms ease,
    border-color 160ms ease,
    transform 160ms ease;
}

.add-source-button:hover {
  border-color: rgba(94, 233, 181, 0.32);
  background:
    linear-gradient(
      180deg,
      rgba(54, 211, 153, 0.2),
      rgba(54, 211, 153, 0.11)
    );
}

.add-source-button:active {
  transform: scale(0.985);
}

.footer-note {
  margin-top: 9px;
  color: var(--text-faint);
  font-size: 9.5px;
  text-align: center;
}

.knowledge-card-enter-active,
.knowledge-card-leave-active,
.knowledge-card-move {
  transition:
    opacity 360ms ease,
    transform 420ms cubic-bezier(.2,.8,.2,1);
}

.knowledge-card-enter-from {
  opacity: 0;
  transform: translateY(-9px) scale(0.97);
}

.knowledge-card-leave-to {
  opacity: 0;
  transform: translateX(-12px) scale(0.96);
}

.knowledge-card-leave-active {
  position: absolute;
}
</style>
