import type {
  ChatMessage,
  ChatStreamEvent,
  CreateConversationResponse,
  CreateMessageResponse,
} from '~/types/chat'


export function useChat() {
  const config = useRuntimeConfig()

  const conversationId = useState<string | null>(
    'chat-conversation-id',
    () => null,
  )

  const messages = useState<ChatMessage[]>(
    'chat-messages',
    () => [],
  )

  const isGenerating = useState(
    'chat-generating',
    () => false,
  )

  const connectionState = useState<
    'idle'
    | 'connecting'
    | 'connected'
    | 'error'
  >(
    'chat-connection-state',
    () => 'idle',
  )

  const errorMessage = useState<string | null>(
    'chat-error',
    () => null,
  )

  const socket = shallowRef<WebSocket | null>(
    null,
  )

  const socketConversationId = ref<
    string | null
  >(
    null,
  )

  let activeAssistantId:
    string
    | null = null


  function localId(
    prefix: string,
  ): string {
    if (
      import.meta.client
      && typeof crypto !== 'undefined'
      && 'randomUUID' in crypto
    ) {
      return `${prefix}_${crypto.randomUUID()}`
    }

    return (
      `${prefix}_`
      + `${Date.now()}_`
      + Math.random()
        .toString(16)
        .slice(2)
    )
  }


  async function ensureConversation():
    Promise<string> {
    if (conversationId.value) {
      return conversationId.value
    }

    const conversation = (
      await $fetch<CreateConversationResponse>(
        '/api/chat/conversations',
        {
          method: 'POST',
        },
      )
    )

    conversationId.value = (
      conversation.conversation_id
    )

    return conversation.conversation_id
  }


  function findActiveAssistant():
    ChatMessage | undefined {
    if (!activeAssistantId) {
      return undefined
    }

    return messages.value.find(
      message => (
        message.id === activeAssistantId
      ),
    )
  }


  function ensureAssistantMessage():
    ChatMessage {
    const existing = findActiveAssistant()

    if (existing) {
      return existing
    }

    const message: ChatMessage = {
      id: localId('assistant'),
      role: 'assistant',
      text: '',
      status: 'streaming',
    }

    messages.value.push(
      message,
    )

    activeAssistantId = message.id

    return message
  }


  function handleStreamEvent(
    event: ChatStreamEvent,
  ) {
    const type = event.event_type

    if (type === 'stream.started') {
      isGenerating.value = true
      ensureAssistantMessage()
      return
    }

    if (type === 'answer.delta') {
      const assistant = (
        ensureAssistantMessage()
      )

      assistant.text += (
        event.text
        || ''
      )

      return
    }

    if (type === 'answer.replace') {
      const assistant = (
        ensureAssistantMessage()
      )

      assistant.text = (
        event.text
        || ''
      )

      return
    }

    if (type === 'stream.completed') {
      const assistant = (
        findActiveAssistant()
      )

      if (assistant) {
        assistant.status = 'complete'
      }

      isGenerating.value = false
      activeAssistantId = null
      return
    }

    if (type === 'stream.error') {
      const assistant = (
        ensureAssistantMessage()
      )

      const payloadMessage = (
        typeof event.payload?.message === 'string'
          ? event.payload.message
          : null
      )

      if (!assistant.text) {
        assistant.text = (
          payloadMessage
          || 'The response could not be completed.'
        )
      }

      assistant.status = 'error'

      errorMessage.value = (
        payloadMessage
        || 'Generation failed.'
      )

      isGenerating.value = false
      activeAssistantId = null
    }
  }


  async function connectSocket(
    targetConversationId: string,
  ): Promise<void> {
    if (!import.meta.client) {
      return
    }

    if (
      socket.value
      && socket.value.readyState
        === WebSocket.OPEN
      && socketConversationId.value
        === targetConversationId
      && connectionState.value
        === 'connected'
    ) {
      return
    }

    closeSocket()

    connectionState.value = 'connecting'
    errorMessage.value = null

    const baseUrl = String(
      config.public.backendWebSocketUrl,
    ).replace(/\/+$/, '')

    const url = (
      `${baseUrl}`
      + '/api/v1/chat/ws/'
      + encodeURIComponent(
        targetConversationId,
      )
    )

    await new Promise<void>(
      (resolve, reject) => {
        const ws = new WebSocket(
          url,
        )

        socket.value = ws
        socketConversationId.value = (
          targetConversationId
        )

        let ready = false

        const timeout = window.setTimeout(
          () => {
            if (ready) {
              return
            }

            ws.close()

            connectionState.value = (
              'error'
            )

            reject(
              new Error(
                'WebSocket connection timed out.',
              ),
            )
          },
          10000,
        )

        ws.onmessage = (
          message,
        ) => {
          let event: ChatStreamEvent

          try {
            event = JSON.parse(
              String(message.data),
            )
          }

          catch {
            return
          }

          if (
            event.event_type
            === 'connection.ready'
          ) {
            ready = true

            window.clearTimeout(
              timeout,
            )

            connectionState.value = (
              'connected'
            )

            resolve()
            return
          }

          handleStreamEvent(
            event,
          )
        }

        ws.onerror = () => {
          if (!ready) {
            window.clearTimeout(
              timeout,
            )

            connectionState.value = (
              'error'
            )

            reject(
              new Error(
                'WebSocket connection failed.',
              ),
            )
          }
        }

        ws.onclose = () => {
          window.clearTimeout(
            timeout,
          )

          socket.value = null
          socketConversationId.value = null

          if (
            connectionState.value
            !== 'error'
          ) {
            connectionState.value = (
              'idle'
            )
          }
        }
      },
    )
  }


  async function sendMessage(
    rawMessage: string,
  ): Promise<void> {
    const text = rawMessage.trim()

    if (
      !text
      || isGenerating.value
    ) {
      return
    }

    errorMessage.value = null

    try {
      const id = (
        await ensureConversation()
      )

      await connectSocket(
        id,
      )

      const userMessage: ChatMessage = {
        id: localId('user'),
        role: 'user',
        text,
        status: 'complete',
      }

      const assistantMessage:
        ChatMessage = {
          id: localId('assistant'),
          role: 'assistant',
          text: '',
          status: 'streaming',
        }

      messages.value.push(
        userMessage,
        assistantMessage,
      )

      activeAssistantId = (
        assistantMessage.id
      )

      isGenerating.value = true

      const submission = (
        await $fetch<CreateMessageResponse>(
          '/api/chat/messages',
          {
            method: 'POST',

            body: {
              conversation_id: id,
              message: text,
            },
          },
        )
      )

      userMessage.id = (
        submission.user_turn_id
      )
    }

    catch (error: any) {
      console.error(
        'Chat send failed:',
        error,
      )

      const assistant = (
        findActiveAssistant()
      )

      if (assistant) {
        assistant.status = 'error'

        if (!assistant.text) {
          assistant.text = (
            'I could not start that response.'
          )
        }
      }

      isGenerating.value = false
      activeAssistantId = null

      errorMessage.value = (
        error?.data?.statusMessage
        || error?.message
        || 'Chat request failed.'
      )
    }
  }


  function closeSocket() {
    if (socket.value) {
      socket.value.close()
    }

    socket.value = null
    socketConversationId.value = null
  }


  const hasMessages = computed(
    () => messages.value.length > 0,
  )


  return {
    conversationId,
    messages,
    hasMessages,
    isGenerating,
    connectionState,
    errorMessage,
    sendMessage,
    closeSocket,
  }
}
