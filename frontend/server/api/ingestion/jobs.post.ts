import { proxyRequest } from 'h3'


export default defineEventHandler(
  async (event) => {
    const config = useRuntimeConfig(event)

    const backendBaseUrl = String(
      config.backendBaseUrl,
    ).replace(/\/+$/, '')

    return proxyRequest(
      event,
      `${backendBaseUrl}/api/v1/ingestion/jobs`,
    )
  },
)
