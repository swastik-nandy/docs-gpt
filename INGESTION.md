# Ingestion Pipeline Guide

This guide explains how to set up and run the ingestion pipeline after cloning the repository.

The main goal is to run this flow:

```text
test_data/*.md/*.mdx
        ↓
FastAPI upload endpoint
        ↓
artifacts/source
        ↓
data transformation worker
        ↓
artifacts/chunked
        ↓
embedding/indexing worker
        ↓
Qdrant
        ↓
artifacts/indexed
```

The repository root is:

```text
emb-docs-gpt/
```

The backend root is:

```text
emb-docs-gpt/backend/
```

---

## 1. Requirements

Required tools:

```text
Git
Python 3.12.13
uv
Docker
Docker Compose v2
```

Check versions:

### Linux / macOS

```bash
git --version
python3 --version
uv --version
docker --version
docker compose version
```

### Windows PowerShell

```powershell
git --version
python --version
uv --version
docker --version
docker compose version
```

Expected Python version:

```text
Python 3.12.13
```

---

## 2. Clone the repository

```bash
git clone <repo-url>
cd emb-docs-gpt
```

If the repository is already cloned:

```bash
cd emb-docs-gpt
git pull
```

---

## 3. Install uv

### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart terminal if `uv` is not detected immediately.

Verify:

```bash
uv --version
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell if `uv` is not detected immediately.

Verify:

```powershell
uv --version
```

---

## 4. Install backend dependencies

Move into backend:

### Linux / macOS

```bash
cd emb-docs-gpt/backend
```

### Windows PowerShell

```powershell
cd emb-docs-gpt\backend
```

Install all dependencies from `pyproject.toml`:

```bash
uv sync
```

`uv sync` reads the backend `pyproject.toml` and installs the project dependencies.

No separate `requirements.txt` step is needed.

Verify project Python:

```bash
uv run python --version
```

Expected:

```text
Python 3.12.13
```

---

## 5. Important config files

API config:

```text
backend/app/config/api.toml
```

Embedding config:

```text
backend/rag/ingestion/embeddings/settings/embeddings.toml
```

Qdrant compose file:

```text
backend/compose.qdrant.yml
```

---

## 6. Upload limits

Configured in:

```text
app/config/api.toml
```

Current limits:

```text
max_files        = 50
max_file_bytes   = 50 MB
max_total_bytes  = 250 MB
```

Supported file types:

```text
.md
.mdx
.markdown
.mdown
.mkd
.mdwn
```

---

## 7. Multitenancy

Current local tenant config:

```toml
[tenancy]
default_tenant_id = "tnt_local_000000000001"
allow_client_tenant_id = false
```

This means all local uploads use:

```text
tnt_local_000000000001
```

The client does not pass tenant IDs.

Artifact paths look like:

```text
rag/ingestion/artifacts/source/tnt_local_000000000001/<job_id>/
rag/ingestion/artifacts/chunked/tnt_local_000000000001/<job_id>/
rag/ingestion/artifacts/indexed/tnt_local_000000000001/<job_id>/
```

`<job_id>` is generated automatically for each upload request.

---

## 8. Embedding model

The embedding worker uses:

```text
BAAI/bge-m3
```

This is a strong retrieval model for RAG because it supports multilingual retrieval and high-quality document embeddings.

Configured in:

```text
rag/ingestion/embeddings/settings/embeddings.toml
```

Current setting:

```toml
device = "auto"
```

Device behavior:

```text
CUDA GPU available      -> uses cuda
Apple Silicon available -> uses mps
Otherwise              -> uses cpu
```

CPU-only laptops can run the demo.

For small `test_data/`, CPU is fine.

For large ingestion jobs, GPU is recommended because embedding generation is the slowest stage.

First embedding run may take longer because the model can be downloaded and cached locally.

---

## 9. Current limitations

The current ingestion pipeline does not have semantic caching yet.

That means repeated chunks can still hit the embedding model again.

Current behavior:

```text
chunk
  ↓
embedding model
  ↓
Qdrant upsert
```

Not implemented yet:

```text
Semantic cache
Redis cache
Redis queue
Distributed job leasing
External orchestration
```

The current system uses local filesystem polling workers.

---

## 10. Qdrant storage

Qdrant runs through Docker Compose.

The Qdrant container stores data inside:

```text
/qdrant/storage
```

The host machine stores it at:

```text
backend/rag/vector_store/
```

So vectors are persisted locally in:

```text
rag/vector_store/
```

Stopping or removing the container does not delete vectors.

Deleting `rag/vector_store/` deletes local Qdrant data.

---

## 11. Start Qdrant

Open Terminal 1.

### Linux / macOS

```bash
cd emb-docs-gpt/backend
docker compose -f compose.qdrant.yml up -d
```

### Windows PowerShell

```powershell
cd emb-docs-gpt\backend
docker compose -f compose.qdrant.yml up -d
```

Check running containers:

```bash
docker ps
```

Check only Qdrant:

### Linux / macOS

```bash
docker ps | grep qdrant
```

### Windows PowerShell

```powershell
docker ps
```

Qdrant should be available at:

```text
http://localhost:6333
```

Check Qdrant:

```bash
curl http://localhost:6333
```

---

## 12. Stop Qdrant

Stop container but keep it available for later:

```bash
docker compose -f compose.qdrant.yml stop
```

Stop and remove container:

```bash
docker compose -f compose.qdrant.yml down
```

This does not remove vectors because they are stored in:

```text
rag/vector_store/
```

---

## 13. Reset Qdrant data

Danger: this deletes local vector data.

### Linux / macOS

```bash
cd emb-docs-gpt/backend

docker compose -f compose.qdrant.yml down
rm -rf rag/vector_store/*
docker compose -f compose.qdrant.yml up -d
```

### Windows PowerShell

```powershell
cd emb-docs-gpt\backend

docker compose -f compose.qdrant.yml down
Remove-Item -Recurse -Force .\rag\vector_store\*
docker compose -f compose.qdrant.yml up -d
```

---

## 14. Start FastAPI server

Open Terminal 2.

### Linux / macOS

```bash
cd emb-docs-gpt/backend

uv run uvicorn app.main:app --reload
```

### Windows PowerShell

```powershell
cd emb-docs-gpt\backend

uv run uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

API upload endpoint:

```text
POST /api/v1/ingestion/jobs
```

Job status endpoint:

```text
GET /api/v1/ingestion/jobs/{job_id}
```

---

## 15. Start data transformation worker

Open Terminal 3.

### Linux / macOS

```bash
cd emb-docs-gpt/backend

uv run python -m rag.ingestion.data_transformation.runner.cli --watch
```

### Windows PowerShell

```powershell
cd emb-docs-gpt\backend

uv run python -m rag.ingestion.data_transformation.runner.cli --watch
```

This worker watches:

```text
rag/ingestion/artifacts/source/
```

It looks for:

```text
_READY
```

Then it writes:

```text
rag/ingestion/artifacts/chunked/
```

Outputs include:

```text
parser/*.ir.json
chunks/*.chunks.jsonl
reports/*.json
_DONE or _FAILED
```

---

## 16. Start embedding/indexing worker

Open Terminal 4.

### Linux / macOS

```bash
cd emb-docs-gpt/backend

uv run python -m rag.ingestion.embeddings.cli --watch
```

### Windows PowerShell

```powershell
cd emb-docs-gpt\backend

uv run python -m rag.ingestion.embeddings.cli --watch
```

This worker watches:

```text
rag/ingestion/artifacts/chunked/
```

It looks for:

```text
_DONE
```

Then it:

```text
reads chunks
generates embeddings
upserts vectors into Qdrant
writes indexed reports
```

Outputs go to:

```text
rag/ingestion/artifacts/indexed/
```

Success marker:

```text
_INDEXED
```

Failure marker:

```text
_INDEX_FAILED
```

---

## 17. Upload test data

After Qdrant, FastAPI, transformation worker, and embedding worker are running, upload files from:

```text
backend/test_data/
```

The upload must go through FastAPI so the real artifact creation code is tested.

---

### Linux upload command

```bash
cd emb-docs-gpt/backend

curl -X POST http://localhost:8000/api/v1/ingestion/jobs \
  $(find test_data -type f \( -name "*.md" -o -name "*.mdx" \) -printf '-F files=@%p ')
```

---

### macOS upload command

```bash
cd emb-docs-gpt/backend

files=()
while IFS= read -r file; do
  files+=("-F" "files=@${file}")
done < <(find test_data -type f \( -name "*.md" -o -name "*.mdx" \))

curl -X POST http://localhost:8000/api/v1/ingestion/jobs "${files[@]}"
```

---

### Windows PowerShell upload command

```powershell
cd emb-docs-gpt\backend

$files = Get-ChildItem .\test_data -Recurse -Include *.md,*.mdx

$args = @()

foreach ($file in $files) {
    $args += "-F"
    $args += "files=@$($file.FullName)"
}

curl.exe -X POST http://localhost:8000/api/v1/ingestion/jobs @args
```

---

## 18. Expected upload response

Example:

```json
{
  "tenant_id": "tnt_local_000000000001",
  "job_id": "dabc1a5a1901472eade4ecee12c6d846",
  "status": "queued",
  "upload_count": 3,
  "source_count": 3,
  "duplicate_count": 0
}
```

Save the `job_id`.

---

## 19. Check job status

```bash
curl http://localhost:8000/api/v1/ingestion/jobs/<job_id>
```

Example:

```bash
curl http://localhost:8000/api/v1/ingestion/jobs/dabc1a5a1901472eade4ecee12c6d846
```

Possible statuses:

```text
queued
succeeded
failed
not_found
```

---

## 20. Verify source artifacts

```bash
find rag/ingestion/artifacts/source -maxdepth 5 -type f | sort
```

Expected shape:

```text
rag/ingestion/artifacts/source/
└── tnt_local_000000000001/
    └── <job_id>/
        ├── manifest.json
        ├── _READY
        └── files/
            └── *.md / *.mdx
```

Windows PowerShell:

```powershell
Get-ChildItem .\rag\ingestion\artifacts\source -Recurse
```

---

## 21. Verify chunked artifacts

```bash
find rag/ingestion/artifacts/chunked -maxdepth 6 -type f | sort
```

Expected shape:

```text
rag/ingestion/artifacts/chunked/
└── tnt_local_000000000001/
    └── <job_id>/
        ├── manifest.json
        ├── _DONE
        ├── parser/
        │   └── *.ir.json
        ├── chunks/
        │   └── *.chunks.jsonl
        └── reports/
            └── *.json
```

Windows PowerShell:

```powershell
Get-ChildItem .\rag\ingestion\artifacts\chunked -Recurse
```

---

## 22. Verify indexed artifacts

```bash
find rag/ingestion/artifacts/indexed -maxdepth 6 -type f | sort
```

Expected shape:

```text
rag/ingestion/artifacts/indexed/
└── tnt_local_000000000001/
    └── <job_id>/
        ├── _INDEXED
        └── reports/
            └── *.json
```

Windows PowerShell:

```powershell
Get-ChildItem .\rag\ingestion\artifacts\indexed -Recurse
```

---

## 23. Complete demo flow

Run these first:

```text
Terminal 1 -> Qdrant
Terminal 2 -> FastAPI
Terminal 3 -> Data transformation worker
Terminal 4 -> Embedding/indexing worker
```

Then run upload command.

After upload, the pipeline continues automatically:

```text
curl upload
    ↓
FastAPI writes artifacts/source/.../_READY
    ↓
Transformation worker writes artifacts/chunked/.../_DONE
    ↓
Embedding worker upserts to Qdrant
    ↓
Embedding worker writes artifacts/indexed/.../_INDEXED
```

---

## 24. Troubleshooting

### FastAPI is not reachable

Check:

```bash
curl http://localhost:8000/health
```

If it fails, restart:

```bash
uv run uvicorn app.main:app --reload
```

---

### Qdrant is not reachable

Check:

```bash
docker ps
curl http://localhost:6333
```

Restart:

```bash
docker compose -f compose.qdrant.yml down
docker compose -f compose.qdrant.yml up -d
```

---

### Upload returns no files

Check that files exist:

### Linux / macOS

```bash
find test_data -type f \( -name "*.md" -o -name "*.mdx" \)
```

### Windows PowerShell

```powershell
Get-ChildItem .\test_data -Recurse -Include *.md,*.mdx
```

---

### Transformation does not start

Check source artifacts:

```bash
find rag/ingestion/artifacts/source -maxdepth 5 -type f | sort
```

Confirm `_READY` exists.

Confirm worker is running:

```bash
uv run python -m rag.ingestion.data_transformation.runner.cli --watch
```

---

### Embedding does not start

Check chunked artifacts:

```bash
find rag/ingestion/artifacts/chunked -maxdepth 6 -type f | sort
```

Confirm `_DONE` exists.

Confirm worker is running:

```bash
uv run python -m rag.ingestion.embeddings.cli --watch
```

---

### CPU machine feels slow

This is expected for large files or many chunks.

CPU-only machines are supported.

For larger ingestion batches, use GPU if available.

---

## 25. Success checklist

```text
[ ] uv sync completed
[ ] Qdrant running
[ ] FastAPI health check returns ok
[ ] Transformation worker running
[ ] Embedding worker running
[ ] curl upload returns job_id
[ ] artifacts/source contains _READY
[ ] artifacts/chunked contains _DONE
[ ] artifacts/indexed contains _INDEXED
[ ] Qdrant data exists in rag/vector_store/
```
