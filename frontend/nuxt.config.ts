export default defineNuxtConfig({
  compatibilityDate: '2026-08-01',

  devtools: {
    enabled: true,
  },

  modules: [
    'reka-ui/nuxt',
    'motion-v/nuxt',
    '@comark/nuxt',
  ],

  css: [
    '~/assets/css/main.css',
  ],

  runtimeConfig: {
    backendBaseUrl: 'http://127.0.0.1:8000',

    public: {
      backendWebSocketUrl: 'ws://127.0.0.1:8000',
    },
  },

  app: {
    head: {
      title: 'Docs GPT',
      meta: [
        {
          name: 'description',
          content: 'AI-powered knowledge assistant',
        },
        {
          name: 'theme-color',
          content: '#080c0b',
        },
      ],
    },
  },
})
