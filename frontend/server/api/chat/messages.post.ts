import type {
  CreateMessageResponse,
} from '../../../app/types/chat'


interface MessageRequest {
  conversation_id: string
  message: string
}


export default defineEventHandler(
  async (event): Promise<CreateMessageResponse> => {
    const config = useRuntimeConfig(event)

    const backendBaseUrl = String(
      config.backendBaseUrl,
    ).replace(/\/+$/, '')

    const body = await readBody<MessageRequest>(
      event,
    )

    try {
      return await $fetch<CreateMessageResponse>(
        `${backendBaseUrl}/api/v1/chat/messages`,
        {
          method: 'POST',
          body,
        },
      )
    }

    catch (error: any) {
      console.error(
        'Chat message submission failed:',
        error,
      )

      throw createError({
        statusCode:
          error?.response?.status
          || 502,

        statusMessage:
          error?.data?.detail
          || 'Message submission failed',
      })
    }
  },
)
