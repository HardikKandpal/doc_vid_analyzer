---
title: Legal Document Analysis API
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# Legal Document Analysis API

This API provides tools for analyzing legal documents, videos, and audio files. It uses NLP models to extract insights, summarize content, and answer legal questions.

## Features

- Document analysis (PDF)
- Video and audio transcription and analysis
- Legal question answering
- Risk assessment and visualization
- Contract clause analysis

## Deployment

This API is deployed on Hugging Face Spaces.

## API Endpoints

- `/analyze_document` - Analyze legal documents
- `/analyze_legal_video` - Analyze legal videos
- `/analyze_legal_audio` - Analyze legal audio
- `/ask_legal_question` - Ask questions about legal documents

## Technologies

- FastAPI
- Hugging Face Transformers
- SpaCy
- PyTorch
- MoviePy