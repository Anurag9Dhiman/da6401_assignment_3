# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Links

- **GitHub Repository:** [https://github.com/Anurag9Dhiman/da6401_assignment_3](https://github.com/Anurag9Dhiman/da6401_assignment_3)
- **W&B Report:** [https://wandb.ai/anuragdhiman666-indian-institute-of-technology-madras/da6401-a3/reports/da6401_assignment_3--VmlldzoxNjg2OTE3Ng](https://wandb.ai/anuragdhiman666-indian-institute-of-technology-madras/da6401-a3/reports/da6401_assignment_3--VmlldzoxNjg2OTE3Ng)

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
