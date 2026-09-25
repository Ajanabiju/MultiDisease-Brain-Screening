# Multi-Disease Brain Screening Web App

Flask web-app prototype for a research project using a planned Hybrid ViT-GNN MRI pipeline.

## Important
This version is a web/prototype layer. It accepts MRI files and displays the intended processing pipeline.
It does NOT make a real medical diagnosis or generate fake prediction results.

## Run in VS Code

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open:
http://127.0.0.1:5000

## Planned ML pipeline

T1 MRI
-> preprocessing
-> MNI152 registration
-> Desikan-Killiany parcellation
-> ViT features + GNN features
-> feature fusion
-> classifier
-> explainability

The actual trained model must be integrated before any real research prediction is displayed.
