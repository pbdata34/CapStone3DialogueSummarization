# Dialogue Summarization Project

By Pierre Hunter
## Colab Links
https://colab.research.google.com/drive/1_15DkZ77TXXPlISLBt3Wy9RuCe-iaGax?usp=sharing  (Final Notebook)

https://colab.research.google.com/drive/1_15DkZ77TXXPlISLBt3Wy9RuCe-iaGax?usp=sharing  (MVP Version)

https://docs.google.com/document/d/1lraBc32xHi8S1llUEGWYnbC0Vliyx5JHJ3ZkijKev-k/edit?tab=t.0 (FlowChart)

https://docs.google.com/document/d/1r-OA6Uwz8Of7pHK76fwS8OVAWsB81Cgey_Jaw0J_eGQ/edit?tab=t.0 (Project Pitch)

## Problem Statement and Business Context

Users are overwhelmed by long group chat conversations and need quick summaries. This project builds an AI system for Acme Communications to automatically summarize dialogues, helping users stay informed without reading every message.

## Technical Approach and Methodology

Used BART (facebook/bart-base) transformer model on SAMSum dataset. Trained on 5,000 dialogues for 3 epochs using Google Colab T4 GPU. Settings: learning rate 5e-5, batch size 8, beam search generation.

## How to Run

**Requirements:**
pip install transformers==4.35.2 datasets evaluate rouge-score accelerate==0.26.1

**Steps:**
1. Open `CAPSTONE3Notebook.ipynb` in Google Colab
2. Enable T4 GPU (Runtime → Change runtime type → GPU)
3. Run all cells in order
4. Training takes ~45 minutes

## Results and Evaluation

ROUGE scores: ROUGE-1 (0.41), ROUGE-2 (0.19), ROUGE-L (0.35). Target was 0.40 ROUGE-L. Summaries are accurate and capture main points but slightly below target.

## Discussion of Limitations and Future Work

Main limitation: trained on 5,000 samples instead of full 14,731 due to GPU time. This caused the 0.35 vs 0.40 gap. Future work: train on full dataset, try larger models, add more epochs.

## References

SAMSum Corpus: https://huggingface.co/datasets/knkarthick/dialogsum


