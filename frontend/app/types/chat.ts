export type ChatRole =
  | 'user'
  | 'assistant'

export type ChatMessageStatus =
  | 'complete'
  | 'streaming'
  | 'error'


export interface ChatMessage {
  id: string
  role: ChatRole
  text: string
  status: ChatMessageStatus
}


export interface CreateConversationResponse {
  conversation_id: string
  status: string
}


export interface CreateMessageResponse {
  conversation_id: string
  user_turn_id: string
  status: string
}


export interface ChatStreamEvent {
  event_type: string
  event_id?: string
  sequence?: number
  timestamp?: number
  conversation_id?: string
  turn_id?: string | null
  text?: string
  payload?: Record<string, unknown>
}
