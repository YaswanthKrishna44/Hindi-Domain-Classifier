# Hindi-Domain-Classifier
# 📊 Hindi Domain Classifier: Comparative Study

This project explores multi-task domain classification for Hindi text using state-of-the-art Transformer architectures.

## 🏆 Final Results
I compared two major architectures using **PEFT (LoRA)** and **4-bit Quantization**.

| Model | Accuracy | Strengths |
| :--- | :--- | :--- |
| **NLLB-200 (600M)** | **79.06%** | Best overall accuracy; superior cross-lingual transfer. |
| **XLM-RoBERTa-Large** | **78.00%** | High precision in Wikinews; robust semantic understanding. |

## 🛠️ Technical Implementation
- **Architecture:** Fine-tuned NLLB-200 Encoder and XLM-R Large backbones.
- **Optimization:** Implemented **Mean Pooling** for sentence embeddings and **Label Smoothing** to handle domain overlap.
- **Hardware:** Optimized for single T4 GPU usage via **QLoRA**.
- **Strategy:** Transitioned from Zero-Shot to **Few-Shot adaptation** to bridge the English-Hindi linguistic gap.
