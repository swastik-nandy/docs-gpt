export interface KnowledgeSourceRecord {
  knowledge_source_id: string
  filename: string
  media_type: string
  size_bytes: number
  status: string
  indexed_chunk_count: number
  indexed_at: string | null
  created_at: string
}

export interface KnowledgeSourcesResponse {
  total: number
  sources: KnowledgeSourceRecord[]
}

export interface KnowledgeSource {
  id: string
  filename: string
  chunks: number
  mediaType: string
  sizeBytes: number
  status: string
  indexedAt: string | null
  createdAt: string
}
