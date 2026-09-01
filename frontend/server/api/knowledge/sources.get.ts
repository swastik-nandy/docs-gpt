import type {
  KnowledgeSourcesResponse,
} from '../../../app/types/knowledge'


export default defineEventHandler(
  async (event): Promise<KnowledgeSourcesResponse> => {
    const config = useRuntimeConfig(event)

    const backendBaseUrl = String(
      config.backendBaseUrl,
    ).replace(/\/+$/, '')

    try {
      return await $fetch<KnowledgeSourcesResponse>(
        `${backendBaseUrl}/api/v1/knowledge/sources`,
      )
    }

    catch (error) {
      console.error(
        'Knowledge API request failed:',
        error,
      )

      throw createError({
        statusCode: 502,
        statusMessage: 'Knowledge service unavailable',
      })
    }
  },
)
