# Video Understanding Pipeline

A video question-answering pipeline that can take a local MP4 or YouTube video, index its visual and spoken content, and answer questions with timestamped evidence.

## Features

* Local MP4 and YouTube input
* YouTube downloading with `yt-dlp`
* Scene detection and keyframe extraction
* Speech transcription with Faster Whisper
* Visual scene captioning with Gemini
* Multi-vector search using Qdrant
* Caption, transcript, and keyframe retrieval
* Timestamped answers
* Temporal grounding and frame previews
* Streamlit demo interface

## How it works

```text
Video
  ↓
Scene detection + keyframes
  ↓
Whisper transcription
  ↓
Scene captioning
  ↓
Qdrant indexing
  ↓
Multi-stream retrieval
  ↓
Answer generation
  ↓
Temporal grounding
```

Each scene is indexed using three types of information:

* visual caption
* transcript
* keyframe

Search results from the three streams are combined using Reciprocal Rank Fusion.

## Setup

Clone the repository and create a virtual environment:

```powershell
git clone <REPOSITORY-URL>
cd video-understanding-pipeline

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

Set your Gemini API key:

```powershell
$env:GEMINI_API_KEY = "API_KEY"
```

YouTube videos are downloaded using `yt-dlp`, which is installed through `requirements.txt`.

## Run the app

```powershell
python -m streamlit run app.py
```

The app lets you:

1. choose a local MP4, upload one, or paste a YouTube URL
2. run the ingestion pipeline
3. ask questions about the video
4. view the answer together with timestamps and supporting frames

For a new video, ingestion runs:

```text
scenes.py → transcribe.py → caption_scenes.py → build_index.py
```

After the index is created, the video is ready for question answering.

## Project structure

```text
.
├── src/
│   └── video_qa/
│       ├── scenes.py
│       ├── transcribe.py
│       ├── caption_scenes.py
│       ├── build_index.py
│       ├── search.py
│       ├── answer.py
│       └── temporal_ground.py
│
├── outputs/
│   └── skill-video-qa.md
│
├── app.py
├── evaluate.py
├── eval_cases.json
├── requirements.txt
└── README.md
```

Generated videos, processed scenes, Qdrant data, and runtime outputs are excluded from Git.

## Example questions

```text
What does the character say about following his passions?

Where are the two characters standing?

When does the fight begin?

What object is the character holding?
```

A typical result includes an answer, confidence score, and a grounded timestamp such as:

```text
Answer:
The two characters are standing outdoors beside a canal.

Timestamp:
05:10.583 – 05:13.022
```

## Evaluation

The repository includes a small evaluation suite:

```text
evaluate.py
eval_cases.json
```

To run:

```powershell
python evaluate.py
```

The tests cover dialogue retrieval, visual questions, answerability, and temporal grounding.

## Reusable AI Skill

```text
outputs/skill-video-qa.md
```

## Known limitation

Retrieval currently works at the scene level. Because a full scene transcript is represented by one embedding, short pieces of dialogue can sometimes be missed when the user's wording is very different from the original transcript. Possible improvements include smaller transcript chunks, hybrid search, or reranking.

## Tech stack

Python, Streamlit, Qdrant, Faster Whisper, Sentence Transformers, CLIP, Gemini, OpenCV, yt-dlp

## Acknowledgement

Built for the Video Understanding Pipeline capstone in AI Engineering from Scratch.
