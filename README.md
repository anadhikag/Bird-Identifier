# Bird Identification Assistant

> An end-to-end AI-powered bird identification and ecological information system using Computer Vision, Retrieval-Augmented Generation (RAG), and LLMs.

---

## Overview

Bird Identification Assistant is a multimodal AI application that identifies bird species from photographs and generates reliable ecological information using Retrieval-Augmented Generation (RAG) and Groq (qwen3-32b).

Unlike traditional image classifiers that only predict a label, this project combines modern Computer Vision and Generative AI to create an interactive bird ecology assistant.

---

# Features

### Computer Vision

- Bird species classification using **EfficientNetV2-S**
- Transfer learning from ImageNet
- Fine-grained classification on **CUB-200-2011**
- Top-1 and Top-5 predictions
- Confidence scores
- Grad-CAM explainability (planned)

---

### Retrieval-Augmented Generation

- Species-specific knowledge base
- Semantic search using Sentence Transformers
- FAISS vector database
- Context-aware prompt construction
- Hallucination reduction through grounded retrieval

---

### Large Language Model

Groq (qwen3-32b) generates:

- Ecological descriptions
- Habitat information
- Migration patterns
- Conservation status
- Interactive question answering

---

## System Architecture

```text
Image
   │
   ▼
Crop Bird
   │
   ▼
EfficientNetV2-S
   │
   ▼
Species Prediction
   │
   ▼
Knowledge Retrieval (FAISS)
   │
   ▼
Prompt Engineering
   │
   ▼
qwen3-32b   
   ▼
Ecological Report + Chat
```

---

# Dataset

## Species Classification

Dataset:

- CUB-200-2011

Contains:

- 200 bird species
- ~11,800 images
- Official train/test split
- Bounding boxes
- Part annotations

---

# Technology Stack

## Computer Vision

- PyTorch
- torchvision
- EfficientNetV2-S
- OpenCV

## NLP / RAG

- Sentence Transformers
- FAISS
- qwen3-32b

## Frontend

- React + Vite
- Framer

## Backend

- FastAPI 

## Cloud

- Microsoft Azure

---

# Goals

This project aims to demonstrate:

- Fine-grained image classification
- Modern transfer learning techniques
- Explainable AI
- Retrieval-Augmented Generation
- Production-oriented AI architecture
- End-to-end AI system design

---

# License

This project is being developed for academic and educational purposes.
