<script setup lang="ts">
import {
  Check,
  Database,
  FileUp,
  Sparkles,
} from 'lucide-vue-next'


type UploadStage =
  | 'uploading'
  | 'indexing'
  | 'building'
  | 'queued'


const props = defineProps<{
  open: boolean
  stage: UploadStage
}>()


const stageIndex = computed(
  () => {
    switch (props.stage) {
      case 'uploading':
        return 0

      case 'indexing':
        return 1

      case 'building':
        return 2

      case 'queued':
        return 3
    }
  },
)


const stageText = computed(
  () => {
    switch (props.stage) {
      case 'uploading':
        return {
          eyebrow: 'Uploading',
          title: 'Uploading your sources',
          description:
            'Sending your Markdown files securely.',
        }

      case 'indexing':
        return {
          eyebrow: 'Indexing',
          title: 'Preparing documents',
          description:
            'Getting your sources ready for indexing.',
        }

      case 'building':
        return {
          eyebrow: 'Knowledge',
          title: 'Building knowledge base',
          description:
            'Connecting your sources to the retrieval pipeline.',
        }

      case 'queued':
        return {
          eyebrow: 'Ready',
          title: 'Sources queued',
          description:
            'Your knowledge build is ready to continue.',
        }
    }
  },
)
</script>


<template>
  <Teleport to="body">
    <Transition name="knowledge-overlay">
      <div
        v-if="open"
        class="knowledge-backdrop"
      >
        <div class="knowledge-card">
          <div class="visual">
            <div
              class="orbit orbit-one"
              :class="{
                complete: stage === 'queued',
              }"
            />

            <div
              class="orbit orbit-two"
              :class="{
                complete: stage === 'queued',
              }"
            />

            <div
              class="core"
              :class="{
                complete: stage === 'queued',
              }"
            >
              <Check
                v-if="stage === 'queued'"
                :size="25"
                :stroke-width="2"
              />

              <Sparkles
                v-else
                :size="23"
                :stroke-width="1.65"
              />
            </div>

            <span class="particle particle-one" />
            <span class="particle particle-two" />
            <span class="particle particle-three" />
          </div>


          <div class="copy">
            <div class="eyebrow">
              {{ stageText.eyebrow }}
            </div>

            <Transition
              name="stage-copy"
              mode="out-in"
            >
              <div
                :key="stage"
                class="stage-copy"
              >
                <h2>
                  {{ stageText.title }}
                </h2>

                <p>
                  {{ stageText.description }}
                </p>
              </div>
            </Transition>
          </div>


          <div class="steps">
            <div
              class="step"
              :class="{
                active: stageIndex >= 0,
              }"
            >
              <div class="step-icon">
                <FileUp :size="14" />
              </div>

              <span>
                Upload
              </span>
            </div>

            <div
              class="connector"
              :class="{
                active: stageIndex >= 1,
              }"
            />

            <div
              class="step"
              :class="{
                active: stageIndex >= 1,
              }"
            >
              <div class="step-icon">
                <Database :size="14" />
              </div>

              <span>
                Index
              </span>
            </div>

            <div
              class="connector"
              :class="{
                active: stageIndex >= 2,
              }"
            />

            <div
              class="step"
              :class="{
                active: stageIndex >= 2,
              }"
            >
              <div class="step-icon">
                <Sparkles :size="14" />
              </div>

              <span>
                Knowledge
              </span>
            </div>
          </div>


          <div class="progress-track">
            <div
              class="progress-value"
              :class="`stage-${stage}`"
            />
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>


<style scoped>
.knowledge-backdrop {
  position: fixed;
  z-index: 9999;
  inset: 0;
  display: grid;
  place-items: center;
  padding: 24px;
  background:
    radial-gradient(
      circle at 50% 42%,
      rgba(54, 211, 153, 0.055),
      transparent 32%
    ),
    rgba(3, 6, 5, 0.78);
  backdrop-filter: blur(10px);
}

.knowledge-card {
  width: min(430px, 100%);
  padding: 38px 34px 30px;
  border: 1px solid rgba(110, 231, 183, 0.14);
  border-radius: 18px;
  background:
    linear-gradient(
      180deg,
      rgba(17, 25, 22, 0.98),
      rgba(8, 13, 11, 0.99)
    );
  box-shadow:
    0 40px 100px rgba(0, 0, 0, 0.5),
    inset 0 1px rgba(255, 255, 255, 0.025);
}

.visual {
  position: relative;
  width: 104px;
  height: 104px;
  margin: 0 auto 26px;
}

.core {
  position: absolute;
  z-index: 3;
  top: 50%;
  left: 50%;
  display: grid;
  width: 48px;
  height: 48px;
  place-items: center;
  border: 1px solid rgba(110, 231, 183, 0.26);
  border-radius: 15px;
  background:
    linear-gradient(
      145deg,
      rgba(54, 211, 153, 0.2),
      rgba(54, 211, 153, 0.07)
    );
  color: #80eac1;
  box-shadow:
    0 0 34px rgba(54, 211, 153, 0.09);
  transform: translate(-50%, -50%);
  animation: core-pulse 2s ease-in-out infinite;
}

.core.complete {
  color: #b6f5da;
  animation: complete-pop 450ms ease forwards;
}

.orbit {
  position: absolute;
  top: 50%;
  left: 50%;
  border-style: solid;
  border-color:
    rgba(54, 211, 153, 0.34)
    transparent
    rgba(54, 211, 153, 0.08)
    transparent;
  border-radius: 50%;
}

.orbit-one {
  width: 78px;
  height: 78px;
  margin: -39px;
  border-width: 1px;
  animation: orbit-spin 2.1s linear infinite;
}

.orbit-two {
  width: 102px;
  height: 102px;
  margin: -51px;
  border-width: 1px;
  opacity: 0.42;
  animation: orbit-spin-reverse 4s linear infinite;
}

.orbit.complete {
  animation-play-state: paused;
  border-color: rgba(54, 211, 153, 0.12);
}

.particle {
  position: absolute;
  z-index: 4;
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: #6ee7b7;
  box-shadow:
    0 0 9px rgba(110, 231, 183, 0.85);
}

.particle-one {
  top: 13px;
  left: 25px;
  animation: particle-float 2.7s ease-in-out infinite;
}

.particle-two {
  right: 13px;
  bottom: 31px;
  animation: particle-float 2.2s ease-in-out infinite 400ms;
}

.particle-three {
  bottom: 7px;
  left: 37px;
  opacity: 0.6;
  animation: particle-float 3s ease-in-out infinite 700ms;
}

.copy {
  min-height: 92px;
  text-align: center;
}

.eyebrow {
  margin-bottom: 7px;
  color: #63d9ad;
  font-size: 10px;
  font-weight: 650;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}

.stage-copy h2 {
  margin: 0;
  color: var(--text, #edf5f1);
  font-size: 19px;
  font-weight: 620;
  letter-spacing: -0.02em;
}

.stage-copy p {
  margin: 7px 0 0;
  color: var(--text-muted, #809089);
  font-size: 11.5px;
  line-height: 1.5;
}

.steps {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  margin-top: 24px;
}

.step {
  display: flex;
  width: 66px;
  align-items: center;
  flex-direction: column;
  gap: 6px;
  color: #48564f;
  transition: color 350ms ease;
}

.step.active {
  color: #82d8b8;
}

.step-icon {
  display: grid;
  width: 30px;
  height: 30px;
  place-items: center;
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 9px;
  background: rgba(255, 255, 255, 0.018);
  transition:
    border-color 350ms ease,
    background 350ms ease,
    box-shadow 350ms ease;
}

.step.active .step-icon {
  border-color: rgba(54, 211, 153, 0.2);
  background: rgba(54, 211, 153, 0.07);
  box-shadow:
    0 0 18px rgba(54, 211, 153, 0.045);
}

.step span {
  font-size: 9px;
  font-weight: 560;
}

.connector {
  width: 42px;
  height: 1px;
  margin-top: 15px;
  background: rgba(255, 255, 255, 0.065);
  transition: background 450ms ease;
}

.connector.active {
  background:
    linear-gradient(
      90deg,
      rgba(54, 211, 153, 0.7),
      rgba(54, 211, 153, 0.23)
    );
}

.progress-track {
  height: 2px;
  margin-top: 26px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.05);
}

.progress-value {
  height: 100%;
  border-radius: inherit;
  background:
    linear-gradient(
      90deg,
      rgba(54, 211, 153, 0.5),
      #69e3b7
    );
  box-shadow:
    0 0 12px rgba(54, 211, 153, 0.28);
  transition:
    width 700ms cubic-bezier(.22,.8,.3,1);
}

.progress-value.stage-uploading {
  width: 22%;
}

.progress-value.stage-indexing {
  width: 52%;
}

.progress-value.stage-building {
  width: 82%;
}

.progress-value.stage-queued {
  width: 100%;
}

.stage-copy-enter-active,
.stage-copy-leave-active {
  transition:
    opacity 180ms ease,
    transform 180ms ease;
}

.stage-copy-enter-from {
  opacity: 0;
  transform: translateY(5px);
}

.stage-copy-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}

.knowledge-overlay-enter-active,
.knowledge-overlay-leave-active {
  transition: opacity 230ms ease;
}

.knowledge-overlay-enter-active .knowledge-card,
.knowledge-overlay-leave-active .knowledge-card {
  transition:
    opacity 230ms ease,
    transform 260ms cubic-bezier(.2,.8,.2,1);
}

.knowledge-overlay-enter-from,
.knowledge-overlay-leave-to {
  opacity: 0;
}

.knowledge-overlay-enter-from .knowledge-card {
  opacity: 0;
  transform: translateY(9px) scale(0.98);
}

.knowledge-overlay-leave-to .knowledge-card {
  opacity: 0;
  transform: translateY(-5px) scale(0.99);
}

@keyframes orbit-spin {
  to {
    transform: rotate(360deg);
  }
}

@keyframes orbit-spin-reverse {
  to {
    transform: rotate(-360deg);
  }
}

@keyframes core-pulse {
  0%,
  100% {
    transform:
      translate(-50%, -50%)
      scale(1);
  }

  50% {
    transform:
      translate(-50%, -50%)
      scale(1.05);
  }
}

@keyframes particle-float {
  0%,
  100% {
    transform: translateY(0);
    opacity: 0.4;
  }

  50% {
    transform: translateY(-6px);
    opacity: 1;
  }
}

@keyframes complete-pop {
  0% {
    transform:
      translate(-50%, -50%)
      scale(0.9);
  }

  60% {
    transform:
      translate(-50%, -50%)
      scale(1.08);
  }

  100% {
    transform:
      translate(-50%, -50%)
      scale(1);
  }
}
</style>
