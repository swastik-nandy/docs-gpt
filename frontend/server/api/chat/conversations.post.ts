import type {
  CreateConversationResponse,
} from '../../../app/types/chat'


export default defineEventHandler(
  async (event): Promise<CreateConversationResponse> => {
    const config = useRuntimeConfig(event)

    const backendBaseUrl = String(
      config.backendBaseUrl,
    ).replace(/\/+$/, '')

    try {
      return await $fetch<CreateConversationResponse>(
        `${backendBaseUrl}/api/v1/chat/conversations`,
        {
          method: 'POST',
        },
      )
    }

    catch (error) {
      console.error(
        'Conversation creation failed:',
        error,
      )

      throw createError({
        statusCode: 502,
        statusMessage: 'Chat service unavailable',
      })
    }
  },
)
