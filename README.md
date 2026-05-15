# ML-Based Support System for Early Malaria Diagnosis

This is a Streamlit web application for malaria detection using microscopic blood smear images.

## Required model files

Place these files inside the `model/` folder:

- `best_malaria_mobilenetv2.keras`
- `class_mapping.json`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Class Mapping

The trained model used this mapping:

```python
{'Parasitized': 0, 'Uninfected': 1}
```

Therefore:

- Prediction probability < 0.5 means Parasitized
- Prediction probability >= 0.5 means Uninfected

## Disclaimer

This application is a diagnostic support system only. It is not a replacement for professional medical diagnosis.
