# Learnaroo

Kid-friendly lesson to short animated video generator

Learnaroo is an experimental platform that converts simple learning topics into short, story-driven animated videos for children. The focus is on curiosity, visual reasoning, and narration rather than textbook-style instruction.

---

## Overview

Learnaroo takes a topic such as “Gravity” and generates a four-beat narrative lesson. Each beat is converted into narration and a short animated scene, which are then stitched together into a final video.

The project explores how large language models and generative video systems can be combined to create engaging educational content for children.

---

## Motivation

Most educational content for children relies heavily on definitions and explanations. This often results in low engagement and poor retention.

Learnaroo experiments with an alternative approach:

* Curiosity-first storytelling
* Action and visual discovery
* Minimal explicit definitions

---

## System Architecture

The system is designed as a modular pipeline:

* Script and narration generation using a Groq-hosted LLM
* Narration synthesis using macOS Text-to-Speech
* Image-to-video generation using BytePlus Seedance
* Audio-video synchronization and stitching using FFmpeg
* Interactive UI built with Streamlit

Each component is loosely coupled to allow independent iteration and replacement.

---

## Current Status

* End-to-end pipeline is functional
* Scene generation, narration, and video stitching are working
* Video generation reliability depends on ByteCloud availability
* Narration quality and storytelling depth are the primary areas under active improvement

This project is in active development and is not intended as a production-ready system.

---

## Running the Project Locally

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

```bash
cp .env.example .env
```

Populate the following values:

```env
ARK_API_KEY=your_byteplus_key
BASE_URL=https://ark.ap-southeast.bytepluses.com/api/v3
MODEL=seedance-1-5-pro-251215
GROQ_API_KEY=your_groq_key
```

### Run the application

```bash
streamlit run app_min_byteplus.py
```

---

## Repository Structure

```
learnaroo/
├── app_min_byteplus.py
├── core/
│   ├── pipeline.py
│   ├── script_groq.py
│   └── __init__.py
├── tools/
│   └── byteplus_client.py
├── assets/
│   └── mascots/
├── requirements.txt
├── .env.example
└── README.md
```

---

## Planned Improvements

* Improve story-driven narration with real-world prompts and dialogue
* Experiment with more creative language models and prompting strategies
* Add narration quality evaluation and automatic rewrite passes
* Improve reliability and cost control for video generation

---

## Notes

* This repository contains an experimental demo
* Generated media outputs are intentionally excluded from version control
* Video generation may fail if ByteCloud credits are unavailable

---
