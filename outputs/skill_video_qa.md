---
name: video-qa
description: Ingest a local MP4 or YouTube video into a scene-level multi-vector index, then answer natural-language questions with grounded timestamps, scene citations, and frame previews.
phase: 19
lesson: 12
---

# Video QA

Use this skill when the user wants to index, search, or ask questions about a video and needs answers that can be verified against exact timestamps.

## Pipeline

```text
local MP4 / YouTube URL
        ↓
scene detection + keyframes
        ↓
Whisper transcription + word timestamps
        ↓
per-scene visual captions
        ↓
Qdrant multi-vector index
  ├─ caption embedding
  ├─ transcript embedding
  └─ keyframe embedding
        ↓
three-stream retrieval + RRF
        ↓
answer synthesis
        ↓
temporal grounding
        ↓
answer + timestamps + scene citations + frame previews
```

## Expected project scripts

```text
src/video_qa/
├─ scenes.py
├─ transcribe.py
├─ caption_scenes.py
├─ build_index.py
├─ search.py
├─ answer.py
└─ temporal_ground.py
```

## Core rules

1. Accept either a local MP4 or one YouTube URL.
2. Reuse an index only when its metadata belongs to the exact selected video.
3. If no matching index exists, run the complete ingestion sequence:
   `scenes.py → transcribe.py → caption_scenes.py → build_index.py`.
4. Ask questions with `answer.py`.
5. Pass the evidence JSON produced by `answer.py` into `temporal_ground.py`.
6. Return only claims supported by visual or transcript evidence.
7. Positive answers should include timestamped scene citations.
8. Unsupported questions should not receive fabricated timestamps.
9. Prefer the temporally grounded interval over a broad scene interval when a valid grounded window exists.
10. Surface frame-preview paths when grounding produces them.
11. Never hardcode a known scene or inject answer-specific keywords merely to make a test pass.

## Windows PowerShell setup

Run from the project root.

```powershell
$python = (Get-Command python).Source
```

## Step 1 — Resolve the video

### Local MP4

```powershell
$video = (Resolve-Path "C:\path\to\video.mp4").Path
```

### YouTube

Download one video at up to 720p. FFmpeg must be available when audio and video streams need merging.

```powershell
New-Item -ItemType Directory -Force .\data\raw\youtube | Out-Null

python -m yt_dlp `
    --no-playlist `
    -f "bv*[height<=720]+ba/b[height<=720]/best" `
    --merge-output-format mp4 `
    -o ".\data\raw\youtube\%(id)s.%(ext)s" `
    "YOUTUBE_URL"
```

Only download and process videos the user is permitted to use.

## Step 2 — Reuse or create an index

Search for metadata:

```powershell
Get-ChildItem `
    .\data\processed `
    -Recurse `
    -Filter index_metadata.json
```

Do not select the newest metadata merely because it is newest. Confirm it belongs to the selected video.

If there is no matching index, ingest the video.

## Step 3 — Detect scenes and extract keyframes

```powershell
$ingestStart = Get-Date

& $python `
    .\src\video_qa\scenes.py `
    "$video" `
    --output .\data\processed
```

Locate the new manifest:

```powershell
$sceneManifest = Get-ChildItem `
    .\data\processed `
    -Recurse `
    -Filter scenes.json |
    Where-Object {
        $_.LastWriteTime -ge $ingestStart.AddSeconds(-5)
    } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
```

Fail clearly if `$sceneManifest` is empty.

## Step 4 — Transcribe with word timestamps

```powershell
$transcribeStart = Get-Date

& $python `
    .\src\video_qa\transcribe.py `
    --manifest "$sceneManifest" `
    --model small.en `
    --device cpu `
    --compute-type int8 `
    --language en
```

For non-English videos, use the appropriate language code or omit `--language` for automatic detection.

```powershell
$processedVideoDir = Split-Path $sceneManifest -Parent

$transcriptManifest = Get-ChildItem `
    "$processedVideoDir" `
    -Filter "scenes_with_transcript*.json" |
    Where-Object {
        $_.LastWriteTime -ge $transcribeStart.AddSeconds(-5)
    } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
```

## Step 5 — Caption scene keyframes

```powershell
$captionStart = Get-Date

& $python `
    .\src\video_qa\caption_scenes.py `
    --manifest "$transcriptManifest" `
    --model gemini-3.1-flash-lite
```

```powershell
$captionManifest = Get-ChildItem `
    "$processedVideoDir" `
    -Filter "scenes_with_captions*.json" |
    Where-Object {
        $_.LastWriteTime -ge $captionStart.AddSeconds(-5)
    } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
```

Do not proceed to indexing if caption generation failed.

## Step 6 — Build the Qdrant multi-vector index

```powershell
$videoId = Split-Path "$processedVideoDir" -Leaf
$collection = "video_$videoId" -replace '[^A-Za-z0-9_-]', '_'

& $python `
    .\src\video_qa\build_index.py `
    --manifest "$captionManifest" `
    --qdrant-path .\data\qdrant `
    --collection "$collection" `
    --device cpu `
    --recreate
```

Resolve metadata:

```powershell
$indexMetadata = Join-Path `
    "$processedVideoDir" `
    "index_metadata.json"

$indexMetadata = (Resolve-Path $indexMetadata).Path
```

Each scene should expose the named vectors:

```text
caption
transcript
keyframe
```

## Step 7 — Retrieve and synthesize an answer

```powershell
$question = "USER QUESTION"
$answerOutput = Join-Path "$processedVideoDir" "last_answer.json"

& $python `
    .\src\video_qa\answer.py `
    "$question" `
    --index-metadata "$indexMetadata" `
    --model gemini-3.1-flash-lite `
    --device cpu `
    --per-stream 10 `
    --top-retrieved 5 `
    --neighbor-radius 1 `
    --output "$answerOutput"
```

Read the structured result:

```powershell
$answer = Get-Content `
    "$answerOutput" `
    -Raw |
    ConvertFrom-Json
```

Expected fields include:

```text
question
answerable
answer
confidence
citations
limitations
evidence_context_path
```

## Step 8 — Ground the answer temporally

Use the exact evidence file reported by `answer.py`.

```powershell
$evidenceContext = $answer.evidence_context_path
$groundingOutput = Join-Path "$processedVideoDir" "grounding_output"

& $python `
    .\src\video_qa\temporal_ground.py `
    "$question" `
    --evidence "$evidenceContext" `
    --index-metadata "$indexMetadata" `
    --model gemini-3.1-flash-lite `
    --video "$video" `
    --output-directory "$groundingOutput" `
    --maximum-windows 5
```

Locate the newest grounding result:

```powershell
$lastGrounding = Get-ChildItem `
    "$groundingOutput" `
    -Recurse `
    -Filter last_grounding.json |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
```

Accepted windows contain:

```text
answerable = true
grounded_start_ms
grounded_end_ms
scene_numbers
evidence
confidence
selected_frame_paths
```

## Final response format

### Answerable

```text
Answer: <direct supported answer>
Confidence: <high|medium|low>

Citations:
- Scene <n>, <grounded start>–<grounded end>

Frame previews:
- <path>
- <path>
```

### Unsupported

```text
Answer: The available video evidence does not support a reliable answer.
Confidence: <confidence>

No grounded timestamp citation is available.
```

Do not turn a rejected grounding window into a citation.

## Timestamp formatting

Use:

```text
MM:SS.mmm
```

or for videos over one hour:

```text
HH:MM:SS.mmm
```

Examples:

```text
310583 ms → 05:10.583
383320 ms → 06:23.320
```

## Evidence rules

- Visual claims require visual support from captions, keyframes, or sampled frames.
- Dialogue claims require transcript support.
- Character identity must be supported within the local scene context.
- Consecutive scenes may be treated as one local interaction when temporal continuity supports it.
- Do not transfer identity from a distant unrelated scene.
- Negative answers should remain conservative.
- Never invent off-screen events, causes, identities, counts, or dialogue.

## Retrieval rules

The query system searches:

```text
caption
transcript
keyframe
```

and merges them with reciprocal rank fusion.

Keep retrieval generic. Do not introduce test-specific logic such as forcing a known scene for one question.

A genuine retrieval miss should be reported as a limitation rather than hidden with hardcoded query expansion.

## Failure handling

### FFmpeg missing

```powershell
ffmpeg -version
ffprobe -version
```

If unavailable, install FFmpeg or expose it on `PATH` before retrying a YouTube download.

### `build_index.py` requests `--manifest`

`build_index.py` is only the final ingestion stage. It must receive `scenes_with_captions.json` from `caption_scenes.py`.

### No matching index

Never reuse another video's metadata. Ingest the selected video.

### Qdrant directory locked

Stop other processes using the same local Qdrant path, then retry.

### Relevant scene is not retrieved

Do not hardcode the expected scene. Report the semantic retrieval miss as a known limitation. Possible future improvements include lexical retrieval, reranking, or finer-grained transcript embeddings.

## Quality checklist

Before answering, verify:

- the selected video and `index_metadata.json` correspond to each other;
- `answer.py` completed successfully;
- the evidence context belongs to the current question;
- temporal grounding completed successfully;
- every positive timestamp comes from accepted evidence;
- frame-preview paths exist before presenting them;
- unsupported questions receive no invented citation;
- no test-specific retrieval shortcut was used.

## Known limitation

Scene-level dense transcript embeddings can miss semantically related dialogue when the question wording differs substantially from the transcript wording. Preserve this as an explicit limitation rather than overfitting a known test case.
