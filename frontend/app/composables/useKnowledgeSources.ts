import type {
  KnowledgeSource,
  KnowledgeSourcesResponse,
} from '~/types/knowledge'


export function useKnowledgeSources() {
  const {
    data,
    status,
    error,
    refresh,
  } = useFetch<KnowledgeSourcesResponse>(
    '/api/knowledge/sources',
    {
      key: 'knowledge-sources',

      default: () => ({
        total: 0,
        sources: [],
      }),
    },
  )

  const sources = computed<KnowledgeSource[]>(
    () => {
      return data.value.sources.map(
        source => ({
          id: source.knowledge_source_id,
          filename: source.filename,
          chunks: source.indexed_chunk_count,
          mediaType: source.media_type,
          sizeBytes: source.size_bytes,
          status: source.status,
          indexedAt: source.indexed_at,
          createdAt: source.created_at,
        }),
      )
    },
  )

  const total = computed(
    () => data.value.total,
  )

  const pending = computed(
    () => status.value === 'pending',
  )

  return {
    sources,
    total,
    pending,
    error,
    refresh,
  }
}
