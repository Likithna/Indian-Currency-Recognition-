# 💵 Indian Currency Recognition System (SVS)

An advanced, deep-learning-based **Indian Currency Recognition System** designed as a **Secure Verification System (SVS)**. This system leverages transfer learning on Google's state-of-the-art **MobileNetV3Large** architecture to identify modern Indian banknotes with **>90% classification accuracy**.

To bridge the gap between classification and real-world utility, the AI model is paired with an **expert-system lookup module** (`security_features.py`) mapped to the official **Reserve Bank of India (RBI)** security standards. It provides instant, contextual authentication checklists, enabling manual anti-counterfeiting verification on any mobile browser.

## 🚀 Live Demo

**👉 [Deploying soon on Streamlit Community Cloud!](https://your-app-name.streamlit.app) 👈** *(Once deployed, replace this link with your live Streamlit URL!)*

---

## 📌 Project Architecture

The system utilizes a dual-engine workflow combining state-of-the-art computer vision with structured rule-based knowledge inference, directly matching the core guidelines of the capstone curriculum.

## ✨ Features

* **🏎️ State-of-the-Art Core:** Built on **MobileNetV3Large**—highly lightweight, fast, and optimized for mobile-edge deployment without compromising spatial depth or texture comprehension.
* **⚡ Double-Phase Transfer Learning:** Initially trained with a frozen ImageNet base, followed by target-layer fine-tuning on Indian currency features (rotation, zoom, tilt) to maximize visual robustness.
* **🎨 Premium "Security-Ink" UI:** Customized dark dashboard using a tailored palette reminiscent of real banknotes, with elegant telemetry metrics, real-time prediction distribution, and interactive sidebars.
* **🛡️ RBI-Integrated Verification:** Delivers base color, specific millimeter dimensions, unique tactile identification marks (shapes for the visually impaired), and reverse motif details corresponding to each prediction.
* **✅ Smart Counterfeit Checklists:** Offers step-by-step physical inspection routines to help users cross-reference the note under UV light, tilt-angles, and touch.

---

## 📊 Model Evaluation & Metrics

The model achieves exceptional real-time stability by moving beyond generic CNN designs to a fine-tuned MobileNet pipeline.

| Metric | Phase 1 (Frozen Base) | Phase 2 (Fine-Tuned) | Target Goal |
| :--- | :--- | :--- | :--- |
| **Training Accuracy** | ~89.04% | **>90.2%** | >90.0% |
| **Validation Accuracy** | ~82.1% | **>88.8%** | >85.0% |
| **Inference Time** | ~110ms / image | **~45ms / image** | <150ms |

---

## 📂 Project Structure

```text
Samsung_Project/
├── app.py                  # Streamlit Web App Interface & Preprocessing Pipeline
├── security_features.py    # Rule-Based RBI Security Database & Checklist Core
├── training_notebook.ipynb # Google Colab Jupyter Notebook used to train MobileNetV3
├── currency_model.keras    # The trained MobileNetV3 model binary
├── class_names.json        # Categorical map of currency denomination strings
└── requirements.txt        # Host environment specifications
```

---
                                        
## 📂 Clone Repository

git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
cd YOUR_REPOSITORY
