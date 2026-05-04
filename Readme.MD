# CN-Detect — Chest X-ray Disease Classifier

A hybrid deep learning web application for classifying chest X-ray images into four categories:
- 🦠 COVID-19
- ✅ Normal
- 🫁 Pneumonia
- ⚠️ Tuberculosis

## Model Architecture
- **EfficientNet-B4** — CNN backbone for local texture features
- **Swin-Tiny** — Transformer for global lung structure
- **CAFS Module** — Novel Cross-Attention Feature Selection Fusion

## Performance
- Test Accuracy: **95.15%**
- Tuberculosis Recall: **1.00**
- Pneumonia Recall: **1.00**

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Research
Final Year Design Project — Daffodil International University, 2025