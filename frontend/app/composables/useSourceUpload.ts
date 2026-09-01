interface CreateIngestionJobResponse {
  tenant_id: string
  job_id: string
  status: 'queued'
  upload_count: number
  source_count: number
  duplicate_count: number
  source_root: string
}


const MAX_UPLOAD_FILES = 10

const ALLOWED_EXTENSIONS = [
  '.md',
  '.mdx',
]


function isAllowedSourceFile(
  file: File,
): boolean {
  const filename = file.name.toLowerCase()

  return ALLOWED_EXTENSIONS.some(
    extension => filename.endsWith(extension),
  )
}


export function useSourceUpload() {
  const uploading = ref(false)

  const uploadError = ref<string | null>(
    null,
  )


  async function uploadSources(
    files: File[],
  ): Promise<CreateIngestionJobResponse> {
    uploadError.value = null

    if (files.length === 0) {
      throw new Error(
        'Select at least one .md or .mdx file.',
      )
    }

    if (files.length > MAX_UPLOAD_FILES) {
      throw new Error(
        `You can upload at most ${MAX_UPLOAD_FILES} files at once.`,
      )
    }

    const invalidFiles = files.filter(
      file => !isAllowedSourceFile(file),
    )

    if (invalidFiles.length > 0) {
      throw new Error(
        `Only .md and .mdx files are allowed: ${invalidFiles
          .map(file => file.name)
          .join(', ')}`,
      )
    }

    const formData = new FormData()

    for (const file of files) {
      formData.append(
        'files',
        file,
        file.name,
      )
    }

    uploading.value = true

    try {
      return await $fetch<CreateIngestionJobResponse>(
        '/api/ingestion/jobs',
        {
          method: 'POST',
          body: formData,
        },
      )
    }

    catch (error) {
      uploadError.value = (
        error instanceof Error
          ? error.message
          : 'Source upload failed.'
      )

      throw error
    }

    finally {
      uploading.value = false
    }
  }


  return {
    uploading,
    uploadError,
    uploadSources,
  }
}
