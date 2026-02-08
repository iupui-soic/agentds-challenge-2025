# Human-AI Collaborative Modeling for Clinical Prediction Tasks in the AgentDS Healthcare Benchmark

This repository contains our solutions for the [AgentDS Healthcare Data Challenge](https://agentds.org/domains/healthcare), submitted as part of the AgentDS workshop at [IEEE ICHI 2026](https://ieeexplore.ieee.org/xpl/conhome/1803988/all-proceedings). Our team (PLHI-IUI) ranked **5th overall** in the healthcare domain across all three challenges.

## About the AgentDS Benchmark

The [AgentDS Benchmark](https://agentds.org/about) is a standardized framework for evaluating human-AI collaboration across diverse data science tasks. The healthcare domain features three clinical prediction challenges using multimodal data -- structured admission records, unstructured clinical notes, scanned PDF receipts, and time-series vital signs -- with standardized metrics ensuring reproducible comparison. Datasets are available on [HuggingFace](https://huggingface.co/datasets/lainmn/AgentDS-Healthcare).

## Challenges

### Challenge 1: 30-Day Hospital Readmission Prediction
**Notebook:** [`challenge1.ipynb`](challenge1.ipynb)

- **Task:** Binary classification -- predict whether an inpatient admission will be followed by a readmission within 30 days
- **Data:** 5,000 training admissions with 8 structured clinical features + 10,000 unstructured discharge note summaries
- **Metric:** Macro-F1
- **Our Score:** **0.8986** (5th place)

**Approach:**
- 887 total features: 37 domain-engineered structured features + 850 TF-IDF features (trigrams from discharge notes)
- Stacking ensemble of 5 base learners (3 XGBoost variants with different depth/learning rate profiles + L1/L2 logistic regression) combined via 8-fold stratified cross-validation
- Meta-learner: Logistic regression trained on out-of-fold probability predictions
- Clinical feature engineering: interaction terms (age x Charlson, age x ED visits), composite risk scores, non-linear age/comorbidity effects

### Challenge 2: ED Cost Forecasting
**Notebook:** [`challenge2.ipynb`](challenge2.ipynb)

- **Task:** Regression -- predict total emergency department costs over the next 3 years
- **Data:** 2,000 patients with 4 structured features + 1 PDF billing receipt per patient
- **Metric:** MAE (USD)
- **Our Score:** **$465.13** (5th place)

**Approach:**
- 20+ features: 8 structured cost-pattern features + 12 PDF-extracted features via PyPDF2 regex parsing
- PDF feature extraction: total billed amount, line item counts, CPT/HCPCS code analysis, high-cost procedure flags, procedure diversity scores
- Ensemble averaging of Random Forest, Gradient Boosting, and Extra Trees regressors
- Domain-informed features: cost-per-visit ratio, log-transformed costs, chronic condition interactions

### Challenge 3: Discharge Readiness Assessment
**Notebook:** [`challenge3.ipynb`](challenge3.ipynb)

- **Task:** Binary classification -- predict whether a patient will be ready for discharge by day 11 of their hospital stay
- **Data:** 1,000 hospital stays with 3 structured features + 10 days of vital signs (HR, SBP, DBP, temperature, respiratory rate) + daily progress notes
- **Metric:** Macro-F1
- **Our Score:** **0.7939** (5th place)

**Approach:**
- 72 total features: 7 structured + 50 vital sign features + 15 clinical NLP features
- Vital sign processing: temporal statistics (mean, std, min, max), trend indicators (linear regression slope, day-to-day variance), recent stability (days 7-10 window), threshold crossings for clinical instability
- Clinical keyword extraction from progress notes: positive indicators (ambulatory, stable, improved) and negative indicators (confused, fever, unstable), mobility scores
- Two-stage stacking ensemble: Random Forest + Gradient Boosting base models with logistic regression meta-learner

## Overall Results

| Challenge | Metric | Score | Rank |
|-----------|--------|-------|------|
| 1: Readmission Prediction | Macro-F1 | 0.8986 | 5th |
| 2: ED Cost Forecasting | MAE (USD) | $465.13 | 5th |
| 3: Discharge Readiness | Macro-F1 | 0.7939 | 5th |
| **Domain Score** | **Aggregate** | **0.8430** | **5th** |

## Key Findings

- **Multimodal feature extraction** was the highest-value human contribution (+0.041 F1 average gain)
- **Domain-guided feature engineering** consistently outperformed automated feature selection by 1-3%
- **Ensemble diversity** via manually-configured model variants outperformed random hyperparameter search
- **Clinical keyword extraction** was more effective than generic TF-IDF for short daily notes (Challenge 3)
- **Statistical aggregation** of time-series features outperformed raw values for small sample sizes (N=1,000)

## Requirements

- Python 3.11
- scikit-learn 1.3.0
- XGBoost 2.0.3
- PyPDF2 3.0.1
- pandas, numpy, scipy
- huggingface_hub (for downloading data from HuggingFace)

## Team

**PLHI-IUI** -- Purkayastha Lab for Health Innovation, Dept. of Biomedical Engineering and Informatics, Indiana University Indianapolis

- Lalitha Pranathi Pulavarthy
- Raajitha Muthyala
- Aravind V. Kuruvikkattil
- Zhenan Yin
- Rashmita Kudamala
- Saptarshi Purkayastha

## Citation

If you use this work, please cite our paper:

```bibtex
@inproceedings{pulavarthy2026humanai,
  title={Human-AI Collaborative Modeling for Clinical Prediction Tasks in the AgentDS Healthcare Benchmark},
  author={Pulavarthy, Lalitha Pranathi and Muthyala, Raajitha and Kuruvikkattil, Aravind V. and Yin, Zhenan and Kudamala, Rashmita and Purkayastha, Saptarshi},
  booktitle={Proceedings of the AgentDS Workshop at IEEE International Conference on Healthcare Informatics (ICHI)},
  year={2026}
}
```

## License

This project is provided for research and educational purposes as part of the AgentDS Benchmark challenge.