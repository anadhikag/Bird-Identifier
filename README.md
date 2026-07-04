# Bird Identification Assistant

> An end-to-end AI-powered bird identification and ecological information system using Computer Vision, Retrieval-Augmented Generation (RAG), and Large Language Models.

---

## Overview

Bird Identification Assistant is a multimodal AI application that identifies bird species from photographs and generates reliable ecological information using Retrieval-Augmented Generation (RAG) and Google's Gemini.

Unlike traditional image classifiers that only predict a label, this project combines modern Computer Vision and Generative AI to create an interactive bird ecology assistant.

The project is being developed in three stages:

- **Stage 1:** Computer Vision + RAG + Streamlit
- **Stage 2:** FastAPI backend + React frontend
- **Stage 3:** Cloud deployment on Microsoft Azure

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

Google Gemini 2.5 Flash generates:

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
Bird Detection (YOLOv8)
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
Gemini 2.5 Flash
   │
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

## Bird Detection 

Dataset:

- Open Images V7 (Bird class)

---

# Technology Stack

## Computer Vision

- PyTorch
- torchvision
- EfficientNetV2-S
- YOLOv8
- OpenCV

## NLP / RAG

- Sentence Transformers
- FAISS
- Google Gemini 2.5 Flash

## Frontend

- Streamlit (Stage 1)
- React + Vite (Stage 2)

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
