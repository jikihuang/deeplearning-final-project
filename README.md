# Digital Human Chat: Multimodal Foundation Model Demo

## Project Overview

This project is a live digital-human chat system for a final presentation demo.

The system allows a user to type a message, generates a natural language response, converts the response into speech with a custom voice, and optionally produces a talking-avatar video. The goal is to demonstrate an end-to-end multimodal foundation model pipeline in a simple interactive interface.

## Foundation Models Used

This project uses three foundation models or foundation-model-based systems:

### 1. Qwen2.5-7B-Instruct

Qwen2.5-7B-Instruct is used as the language model. It receives the user message and generates the assistant's response through a local vLLM OpenAI-compatible API server.

### 2. Boson Higgs Audio TTS

Boson Higgs Audio TTS is used for text-to-speech generation. The project supports both preset voices and custom voice IDs, allowing the digital human to speak with a personalized voice.

### 3. MuseTalk V1.5

MuseTalk V1.5 is used for audio-driven talking-avatar video generation. The synthesized speech audio is used to generate a talking digital-human video.

## Main Features

* Text-based chat interface.
* Voice Mode: generates an audio response only.
* Video Mode: generates a talking-avatar video response.
* Custom avatar image upload.
* Custom Boson voice support.
* Local Qwen vLLM server.
* Persistent MuseTalk worker server.
* Gradio-based live demo interface.

## Project Structure

```text
.
├── app.py
├── musetalk_worker_server.py
├── README.md
├── .env.example
├── .gitignore
├── requirements-app.txt
├── requirements-worker.txt
├── scripts/
│   ├── start_qwen.sh
│   ├── start_musetalk_worker.sh
│   ├── start_app.sh
│   └── check_system.sh
└── voice_samples/
    └── .gitkeep
```

## Environment Requirements

The project was developed and tested in a Linux Docker environment with NVIDIA GPUs.

Recommended environment:

* Python 3.10+
* CUDA-compatible GPU
* Conda
* ffmpeg
* vLLM
* Gradio
* MuseTalk V1.5
* Boson API key

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/jikihuang/deeplearning-final-project.git
cd deeplearning-final-project
```

### 2. Create `.env`

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
nano .env
```

Example:

```env
BOSON_API_KEY=your_boson_api_key_here
BOSON_VOICE=default

LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_API_KEY=local-qwen
LLM_MODEL=qwen2.5-7b-instruct

MUSETALK_DIR=/workspace/MuseTalk
MUSETALK_CUDA_VISIBLE_DEVICES=1
MUSETALK_WORKER_URL=http://127.0.0.1:8890
MUSETALK_WORKER_PORT=8890

GRADIO_SERVER_PORT=8888
```

Do not upload `.env` to GitHub.

## Install Dependencies

### App Dependencies

```bash
conda activate digitalhuman
pip install -r requirements-app.txt
```

### Worker Dependencies

```bash
conda activate digitalhuman
pip install -r requirements-worker.txt
```

MuseTalk should be installed separately. In this project, MuseTalk is expected to be located at:

```text
/workspace/MuseTalk
```

The MuseTalk V1.5 model files should be prepared under:

```text
/workspace/MuseTalk/models/musetalkV15/
```

## Running the Demo

This project uses three running services:

1. Qwen vLLM server
2. MuseTalk worker server
3. Gradio web app

Open three terminals.

### Terminal 1: Start Qwen vLLM

```bash
bash scripts/start_qwen.sh
```

The Qwen API should run at:

```text
http://127.0.0.1:8001/v1
```

### Terminal 2: Start MuseTalk Worker

```bash
bash scripts/start_musetalk_worker.sh
```

Check the worker:

```bash
curl http://127.0.0.1:8890/health
```

### Terminal 3: Start Gradio App

```bash
bash scripts/start_app.sh
```

Open the browser:

```text
http://<server-ip>:8888
```

## Usage

### Voice Mode

In Voice Mode, the user enters a message and the system generates:

1. A text response using Qwen.
2. A speech response using Boson TTS.

The right-side avatar remains idle while the audio response plays.

### Video Mode

In Video Mode, the user enters a message and the system generates:

1. A text response using Qwen.
2. A speech response using Boson TTS.
3. A talking-avatar video using MuseTalk.

The generated video is displayed in the right-side digital-human panel.

## Example Demo Prompts

English:

```text
Hi, can you briefly introduce this final project?
```

```text
Can you explain what foundation models are used in this demo?
```

Korean:

```text
안녕하세요. 이 프로젝트를 간단히 소개해 줄 수 있나요?
```

Chinese:

```text
你好，请简单介绍一下这个数字人项目。
```

## Reproducibility Notes

This repository contains the main source code and setup instructions. Large model weights, API keys, generated outputs, private voice samples, and temporary files are not included.


## Project Summary

This project demonstrates a practical multimodal digital-human assistant by combining:

* LLM-based response generation,
* custom voice synthesis,
* audio-driven talking-avatar video generation.

The final result is an interactive digital human that can answer user questions in both voice and video modes.
