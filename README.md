# A/B Testing Using Real Kaggle E-Commerce Dataset

This project evaluates the performance of a new landing page design in increasing user conversions. It uses an actual dataset structure from Kaggle and applies statistical testing and uplift modeling using PySpark and EconML.

## Key Features
- Realistic dataset structure based on Kaggle
- A/B testing setup: Control vs. Treatment
- PySpark preprocessing + EconML for uplift modeling
- Hypothesis testing with p-values and confidence intervals

## Folder Structure
- `data/`: Realistic synthetic data from Kaggle schema
- `notebooks/`: Jupyter notebook for analysis
- `scripts/`: PySpark script for modeling
- `README.md` and `requirements.txt`

## Getting Started
Install required packages:
```bash
pip install -r requirements.txt
```
Run the notebook:
```bash
jupyter notebook notebooks/kaggle_ab_analysis.ipynb
```
