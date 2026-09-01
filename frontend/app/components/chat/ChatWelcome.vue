<script setup lang="ts">
import {
  ArrowUpRight,
  Database,
  MessageSquareText,
  Sparkles,
} from 'lucide-vue-next'

const suggestions = [
  'How do I get started?',
  'Explain the indexed architecture',
  'What do these docs say about search?',
]

const emit = defineEmits<{
  select: [message: string]
}>()
</script>

<template>
  <div class="welcome">
    <div class="assistant-orb">
      <Sparkles
        :size="22"
        :stroke-width="1.8"
      />
    </div>

    <div class="welcome-copy">
      <div class="welcome-eyebrow">
        <Database :size="12" />

        <span>
          Grounded in your knowledge
        </span>
      </div>

      <h1>
        What can I help you find?
      </h1>

      <p>
        Ask questions across your indexed documentation.
        Docs GPT retrieves supporting context before it answers.
      </p>
    </div>

    <div class="suggestions">
      <button
        v-for="suggestion in suggestions"
        :key="suggestion"
        type="button"
        class="suggestion-card"
        @click="emit('select', suggestion)"
      >
        <MessageSquareText
          :size="15"
          class="suggestion-icon"
        />

        <span>
          {{ suggestion }}
        </span>

        <ArrowUpRight
          :size="13"
          class="suggestion-arrow"
        />
      </button>
    </div>
  </div>
</template>

<style scoped>
.welcome {
  display: flex;
  width: min(660px, calc(100% - 48px));
  flex-direction: column;
  align-items: center;
  animation:
    welcome-in 620ms
    cubic-bezier(.2,.8,.2,1)
    both;
}

.assistant-orb {
  position: relative;
  display: grid;
  width: 52px;
  height: 52px;
  place-items: center;
  border: 1px solid rgba(94, 233, 181, 0.17);
  border-radius: 15px;
  background:
    radial-gradient(
      circle at 35% 25%,
      rgba(94, 233, 181, 0.16),
      rgba(54, 211, 153, 0.045) 55%,
      rgba(13, 19, 17, 0.8)
    );
  color: var(--green-bright);
  box-shadow:
    0 20px 55px rgba(0, 0, 0, 0.25),
    0 0 45px rgba(54, 211, 153, 0.04),
    inset 0 1px rgba(255, 255, 255, 0.06);
}

.assistant-orb::after {
  position: absolute;
  width: 72px;
  height: 72px;
  border: 1px solid rgba(54, 211, 153, 0.04);
  border-radius: 21px;
  content: '';
}

.welcome-copy {
  margin-top: 25px;
  text-align: center;
}

.welcome-eyebrow {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  margin-bottom: 9px;
  color: #6f9e89;
  font-size: 10.5px;
  font-weight: 560;
}

.welcome-copy h1 {
  color: var(--text);
  font-size: clamp(27px, 3vw, 38px);
  font-weight: 590;
  letter-spacing: -0.045em;
  line-height: 1.08;
}

.welcome-copy p {
  width: min(510px, 100%);
  margin: 13px auto 0;
  color: var(--text-muted);
  font-size: 13px;
  font-weight: 410;
  line-height: 1.65;
}

.suggestions {
  display: grid;
  width: 100%;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-top: 31px;
}

.suggestion-card {
  display: flex;
  min-height: 78px;
  cursor: pointer;
  flex-direction: column;
  align-items: flex-start;
  padding: 12px;
  border: 1px solid var(--border);
  border-radius: 11px;
  background:
    linear-gradient(
      180deg,
      rgba(17, 25, 23, 0.63),
      rgba(12, 18, 16, 0.5)
    );
  color: var(--text-muted);
  text-align: left;
  transition:
    transform 180ms cubic-bezier(.2,.8,.2,1),
    border-color 180ms ease,
    background 180ms ease,
    color 180ms ease;
}

.suggestion-card:hover {
  border-color: rgba(94, 233, 181, 0.15);
  background: rgba(22, 32, 29, 0.7);
  color: var(--text-soft);
  transform: translateY(-2px);
}

.suggestion-icon {
  margin-bottom: 10px;
  color: #5a9f83;
}

.suggestion-card span {
  flex: 1;
  font-size: 10.5px;
  font-weight: 520;
  line-height: 1.4;
}

.suggestion-arrow {
  align-self: flex-end;
  color: var(--text-faint);
}

@keyframes welcome-in {
  from {
    opacity: 0;
    transform: translateY(10px) scale(0.985);
  }

  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

@media (max-width: 800px) {
  .suggestions {
    grid-template-columns: 1fr;
  }
}
</style>
