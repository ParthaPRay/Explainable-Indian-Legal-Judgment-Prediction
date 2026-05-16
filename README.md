# Explainable Indian Legal Judgment Prediction

**Repository:** [Explainable-Indian-Legal-Judgment-Prediction](https://github.com/ParthaPRay/Explainable-Indian-Legal-Judgment-Prediction)  
**Notebook:** `AI_Law.ipynb`  
**Author:** Partha Pratim Ray  
**Affiliation:** Department of Computer Applications, Sikkim University, India  

---

## Overview

This repository presents a reproducible Google Colab-based framework for **Explainable Indian Legal Judgment Prediction** using the **Jud-IPL: Indian Legal Judgement Prediction Dataset**. The work combines classical machine learning, domain-specific Transformer modeling, explainable artificial intelligence, subgroup analysis, counterfactual perturbation, and publication-grade visualization.

The primary objective is not only to predict legal judgment outcomes, but also to examine **why** a model makes a given prediction. Therefore, the notebook moves beyond ordinary classification and includes interpretability-oriented analyses such as TF-IDF feature inspection, SHAP-based explanation, legally meaningful counterfactual perturbation, proof-sentence comparison, and error analysis.

---

## Dataset

This work uses the **Jud-IPL: Indian Legal Judgement Prediction Dataset**, available on Kaggle:

[https://www.kaggle.com/datasets/bhavyabj/jud-ipl-indian-legal-judgement-prediction-dataset](https://www.kaggle.com/datasets/bhavyabj/jud-ipl-indian-legal-judgement-prediction-dataset)

Jud-IPL is a large-scale Indian legal judgment prediction dataset containing Supreme Court of India case documents. Each case document is annotated with a final judgment outcome and case category. The dataset supports legal NLP tasks such as:

- legal judgment prediction,
- legal document classification,
- legal text representation learning,
- explainable legal AI,
- domain-specific legal NLP benchmarking.

The dataset file used in this repository is:

```text
case_files_total.csv
````

The dataset contains the following major fields:

```text
name
case_category
case_type
case_info
judgement
tokens
sentences
label
proof_sentence
```

The target labels include:

```text
Accepted
Rejected
Other
```

For binary judgment prediction, the notebook keeps only `Accepted` and `Rejected` cases and maps them as:

```text
Rejected → 0
Accepted → 1
```

Cases with undetermined or unusable labels are removed during preprocessing.

---

## Dataset Citation

If you use the Jud-IPL dataset, please cite the original dataset source as requested by the dataset authors:

```bibtex
@inproceedings{jain2024clolex,
  author    = {Jain, S. and Harde, P. and Jain, B.},
  title     = {CloLex: Exploring Closed Lexicals for LegalTech},
  booktitle = {MIND 2024},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2024},
  doi       = {10.1007/978-3-032-14534-5_29}
}
```

---

## Use of InLegalBERT

This repository also uses **InLegalBERT** from Hugging Face as a domain-specific Transformer model for Indian legal text.

Model page:

[https://huggingface.co/law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)

InLegalBERT is a pre-trained legal-domain language model developed for Indian legal NLP. The model card states that it was trained on a large corpus of Indian legal documents collected from the Supreme Court and High Courts of India, covering multiple legal domains such as civil, criminal, and constitutional law. It follows the BERT-base configuration and is further trained using masked language modeling and next sentence prediction objectives. ([Hugging Face][1])

In this repository, InLegalBERT is used through Hugging Face Transformers for a sampled binary judgment prediction experiment.

---

## InLegalBERT Citation

If you use InLegalBERT, please cite:

```bibtex
@inproceedings{paul-2022-pretraining,
  url = {https://arxiv.org/abs/2209.06049},
  author = {Paul, Shounak and Mandal, Arpan and Goyal, Pawan and Ghosh, Saptarshi},
  title = {Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law},
  booktitle = {Proceedings of 19th International Conference on Artificial Intelligence and Law - ICAIL 2023},
  year = {2023}
}
```

---

## Repository Structure

```text
Explainable-Indian-Legal-Judgment-Prediction/
│
├── AI_Law.ipynb
├── README.md
└── outputs/
    ├── figures/
    └── tables/
```

The notebook generates publication-oriented outputs such as:

```text
figures/
tables/
judipl_results_tables.xlsx
judipl_publication_figures.zip
```

---

## Methodological Workflow

The complete workflow implemented in `AI_Law.ipynb` consists of the following stages.

---

## 1. Environment Setup

The notebook installs and imports required libraries for:

* data processing,
* machine learning,
* Transformer modeling,
* explainable AI,
* evaluation,
* visualization,
* result export.

Main Python libraries include:

```text
pandas
numpy
matplotlib
scikit-learn
transformers
datasets
evaluate
torch
shap
lime
openpyxl
```

The notebook is designed to run in **Google Colab**. It automatically checks whether CUDA/GPU is available and applies CPU-safe settings when no GPU is detected.

---

## 2. Kaggle Dataset Download

The notebook uses the Kaggle API to download the Jud-IPL dataset directly into the Colab environment. The user is prompted to provide a Kaggle API token securely.

Dataset download is performed using:

```bash
kaggle datasets download -d bhavyabj/jud-ipl-indian-legal-judgement-prediction-dataset
```

The dataset is extracted and loaded from:

```text
/content/judipl/case_files_total.csv
```

---

## 3. Dataset Audit

The notebook performs a basic dataset audit, including:

* column inspection,
* non-null count,
* null count,
* missing value percentage,
* number of unique values,
* data type inspection.

This audit helps identify missing values and confirms the usability of the key fields required for judgment prediction.

---

## 4. Data Cleaning and Label Preparation

The preprocessing stage performs the following operations:

* standardizes label values,
* standardizes case category values,
* removes unusable or undetermined labels,
* keeps only `Accepted` and `Rejected` cases,
* maps binary labels to numerical targets,
* removes rows with missing judgment text,
* fills missing textual fields where necessary.

The final binary prediction target is:

```text
Accepted / Rejected
```

---

## 5. Text Construction and Cleaning

A combined input text field is created using:

```text
case_info + judgement
```

The text is normalized by:

* converting to lowercase,
* removing unnecessary special characters,
* normalizing whitespace,
* preserving useful legal punctuation and numeric symbols.

A document length variable is also created using word counts, enabling later analysis of model behavior across short and long legal documents.

---

## 6. Exploratory Data Analysis

The notebook generates publication-style exploratory analyses, including:

* judgment outcome distribution,
* case category distribution,
* outcome distribution by case category,
* text length distribution by judgment outcome,
* descriptive statistics for tokens, sentences, and document length.

All plots are saved in high-resolution format for manuscript use.

---

## 7. Train-Test Split

The notebook follows the dataset description by applying an **80:20 train-test split without shuffling**. This preserves the original ordering logic of the dataset and provides a reproducible evaluation setting.

The resulting split is used for classical machine learning experiments and full-dataset evaluation.

---

## 8. Classical Machine Learning Baselines

The notebook trains and compares several TF-IDF-based classical machine learning models:

```text
TF-IDF + Logistic Regression
TF-IDF + Linear Support Vector Classifier
TF-IDF + Multinomial Naive Bayes
```

The TF-IDF representation uses unigram and bigram features with a controlled feature limit to make the pipeline computationally practical in Google Colab.

Each model is evaluated using:

* accuracy,
* macro precision,
* macro recall,
* macro F1-score,
* weighted precision,
* weighted recall,
* weighted F1-score,
* ROC-AUC,
* average precision where applicable,
* training time.

---

## 9. Classical Model Comparison

The notebook creates a comparative performance table and a publication-grade bar plot comparing the baseline models. The best-performing classical model is automatically identified using macro F1-score.

A confusion matrix is generated for the best classical model to inspect class-wise prediction behavior.

---

## 10. InLegalBERT Transformer Experiment

The notebook includes a sampled fine-tuning experiment using:

```text
law-ai/InLegalBERT
```

Since full Transformer fine-tuning on long legal documents can be computationally expensive, the notebook uses CPU/GPU-aware sampling:

* larger sample size when GPU is available,
* smaller sample size when running on CPU.

The InLegalBERT experiment uses:

* Hugging Face `AutoTokenizer`,
* `AutoModelForSequenceClassification`,
* Hugging Face `Dataset`,
* `Trainer`,
* dynamic padding,
* sequence truncation,
* binary classification head.

This component provides a domain-specific Transformer baseline for comparison against classical sparse lexical models.

---

## 11. Transformer Evaluation

The fine-tuned InLegalBERT model is evaluated on a sampled test set using the same performance framework as the classical models.

The notebook produces:

* Transformer result table,
* confusion matrix,
* ROC curve,
* precision-recall curve,
* combined model comparison table.

The Transformer experiment is interpreted as a resource-constrained legal Transformer feasibility study rather than a full-scale long-document Transformer benchmark.

---

## 12. Civil and Criminal Subgroup Analysis

The notebook performs subgroup evaluation across major legal categories:

```text
Civil
Criminal
```

This analysis examines whether the best model behaves consistently across different legal domains. It supports a more legally meaningful interpretation of the model beyond aggregate performance.

---

## 13. Case-Type-Wise Performance Analysis

The notebook further evaluates model performance across individual `case_type` groups, excluding very small groups where evaluation would be unstable.

This provides a more granular legal-domain analysis and helps identify whether some case types are easier or harder to classify.

---

## 14. Document-Length Effect Analysis

Legal judgments vary substantially in length. To study this effect, the notebook divides test documents into length-based groups and evaluates the best model separately on each group.

This analysis helps examine whether prediction quality changes across:

```text
Very Short
Short
Long
Very Long
```

documents.

---

## 15. TF-IDF Feature-Based Explanation

For the best linear model, the notebook extracts the most influential TF-IDF features associated with:

```text
Accepted
Rejected
```

This provides a transparent lexical explanation layer and allows inspection of legally meaningful terms, phrases, and decision-related expressions influencing the classifier.

---

## 16. SHAP-Based Explainable AI

The notebook uses SHAP to provide global feature attribution for the Logistic Regression baseline. SHAP is applied to a manageable sample of test documents to identify high-impact lexical features.

The SHAP workflow produces:

* global SHAP feature importance plot,
* ranked SHAP importance table,
* exportable explanation results.

This enables a model-agnostic explanation layer for legal judgment prediction.

---

## 17. Proof-Sentence Alignment Analysis

The dataset includes a `proof_sentence` field. The notebook uses this field as an explanation-oriented reference signal.

A proof-sentence overlap analysis is performed to examine whether globally important model features overlap with the textual evidence supplied in the dataset.

This strengthens the explainability component by connecting model explanations with dataset-provided legal evidence.

---

## 18. Counterfactual Perturbation Analysis

The notebook performs a counterfactual perturbation experiment by removing selected legally meaningful phrases and terms from test documents.

Examples of removed legal expressions include phrases related to:

```text
appeal allowed
appeal dismissed
petition allowed
petition dismissed
conviction
sentence
acquittal
section
evidence
petitioner
respondent
```

The purpose is to test whether model predictions change when legally salient cues are removed.

This analysis supports a robustness-oriented view of explainable legal AI.

---

## 19. Shortcut Sensitivity and Outcome-Phrase Ablation

To address possible lexical shortcut learning, the notebook performs an additional ablation experiment by removing explicit outcome-indicative phrases such as:

```text
appeal allowed
appeal dismissed
petition allowed
petition dismissed
the appeal succeeds
the appeal fails
we allow the appeal
we dismiss the appeal
```

The model is retrained and evaluated after removing these phrases.

This experiment helps distinguish between:

* genuine legal text modeling,
* direct outcome phrase detection,
* lexical shortcut reliance.

This is important for responsible legal AI evaluation.

---

## 20. Error Analysis

The notebook generates an error analysis table for misclassified cases. The table includes:

* case name,
* case category,
* case type,
* true label,
* predicted label,
* document length,
* proof sentence.

This helps inspect failure cases and supports qualitative discussion in a research paper.

---

## 21. Calibration Analysis

Because legal AI systems should not only predict outcomes but also express uncertainty responsibly, the notebook includes calibration analysis using a calibrated LinearSVC model.

This step generates:

* calibrated predictions,
* probability estimates,
* calibration curve.

The calibration component is useful for discussing confidence-aware legal decision support.

---

## 22. Exported Outputs

The notebook exports results in manuscript-friendly formats.

Generated table outputs include:

```text
baseline_results.csv
transformer_results.csv
combined_model_comparison.csv
subgroup_results.csv
case_type_performance.csv
length_bin_performance.csv
tfidf_feature_importance.csv
shap_importance_table.csv
counterfactual_summary.csv
counterfactual_changed_examples.csv
misclassified_examples.csv
error_summary.csv
judipl_results_tables.xlsx
```

Generated figure outputs include:

```text
label_distribution.png
case_category_distribution.png
outcome_by_case_category.png
text_length_distribution.png
baseline_model_comparison.png
best_baseline_confusion_matrix.png
combined_model_comparison.png
inlegalbert_confusion_matrix.png
roc_curve_comparison.png
precision_recall_curve_comparison.png
subgroup_macro_f1.png
tfidf_top_features.png
shap_global_feature_importance.png
counterfactual_change_rate.png
case_type_macro_f1.png
length_bin_macro_f1.png
calibration_curve.png
```

The figures are also compressed into:

```text
judipl_publication_figures.zip
```

---

## Key Contributions

This repository contributes a complete, reproducible framework for explainable legal judgment prediction in the Indian context. The main contributions are:

1. A full Google Colab pipeline for Jud-IPL-based legal judgment prediction.
2. Classical TF-IDF-based machine learning baselines for binary judgment outcome classification.
3. A sampled InLegalBERT-based Transformer experiment using Hugging Face.
4. Publication-grade exploratory data analysis and model comparison plots.
5. Civil/criminal subgroup evaluation.
6. Case-type-wise legal-domain evaluation.
7. Document-length effect analysis.
8. TF-IDF coefficient-based interpretability.
9. SHAP-based global feature attribution.
10. Proof-sentence alignment analysis.
11. Counterfactual legal text perturbation.
12. Shortcut phrase ablation for robustness testing.
13. Error analysis using misclassified examples.
14. Calibration analysis for confidence-aware legal AI.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ParthaPRay/Explainable-Indian-Legal-Judgment-Prediction.git
cd Explainable-Indian-Legal-Judgment-Prediction
```

### 2. Open the notebook

Open the following notebook in Google Colab:

```text
AI_Law.ipynb
```

### 3. Provide Kaggle API token

The notebook asks for a Kaggle API token. You can generate it from your Kaggle account and paste it when prompted.

The notebook stores the token securely inside the Colab runtime:

```text
/root/.kaggle/access_token
```

Do not hard-code or publicly commit your Kaggle token.

### 4. Run all cells

Run the notebook sequentially from top to bottom.

The notebook will:

* install dependencies,
* download the Kaggle dataset,
* preprocess legal judgments,
* train models,
* generate explanations,
* produce plots,
* export tables and figures.

---

## Main Notebook

```text
AI_Law.ipynb
```

The notebook contains all code needed for:

* dataset loading,
* preprocessing,
* exploratory analysis,
* model training,
* evaluation,
* explainability,
* counterfactual analysis,
* result export.

---

## Recommended Research Framing

A suitable research framing for this project is:

> This work investigates resource-efficient and explainable Indian legal judgment prediction using the Jud-IPL dataset. It compares classical sparse lexical models with a sampled domain-specific Transformer experiment and integrates interpretability methods such as TF-IDF coefficient analysis, SHAP attribution, proof-sentence alignment, and counterfactual perturbation.

A possible paper title is:

```text
Resource-Efficient Explainable Legal Judgment Prediction on Indian Supreme Court Cases Using TF-IDF, InLegalBERT, SHAP, and Counterfactual Perturbation
```

Another possible title is:

```text
From Prediction to Explanation: Resource-Efficient and Explainable Indian Legal Judgment Prediction Using Classical Machine Learning and InLegalBERT
```

---

## Ethical and Legal Disclaimer

This repository is intended strictly for academic and research purposes. The models developed here should not be used as standalone tools for legal decision-making.

Legal judgment prediction systems may reflect dataset bias, lexical shortcuts, historical patterns, annotation limitations, and jurisdiction-specific constraints. Any practical use of legal AI systems must involve domain experts, legal professionals, transparency safeguards, and human oversight.

The predictions generated by this notebook do not constitute legal advice.

---

## Limitations

The work has several limitations:

* The Transformer experiment may be constrained by Google Colab resources.
* Long legal documents are truncated for Transformer input.
* Classical models may learn outcome-indicative lexical patterns.
* The dataset may contain noisy or unevenly distributed legal categories.
* The analysis focuses on binary accepted/rejected prediction after removing other labels.
* Explainability methods provide approximations and should not be treated as definitive legal reasoning.
* The framework is designed for research and not for real-world judicial deployment.

---

## Future Work

Possible extensions include:

* long-document Transformer models,
* hierarchical legal document encoders,
* retrieval-augmented judgment prediction,
* legal rationale extraction,
* proof-sentence supervised explanation learning,
* fairness and bias analysis across case categories,
* temporal validation by judgment year,
* multi-class prediction including other/undetermined outcomes,
* comparison with additional Indian legal LLMs,
* human expert evaluation of model explanations.

---

## Acknowledgement

This work uses the Jud-IPL dataset from Kaggle and the InLegalBERT model from Hugging Face. The repository is intended to support reproducible research in explainable legal NLP for the Indian judicial context.

---

## References

### Jud-IPL Dataset

Jain, S., Harde, P., & Jain, B. (2024). *CloLex: Exploring Closed Lexicals for LegalTech*. MIND 2024. Communications in Computer and Information Science, Springer. DOI: `10.1007/978-3-032-14534-5_29`.

Dataset page:
[https://www.kaggle.com/datasets/bhavyabj/jud-ipl-indian-legal-judgement-prediction-dataset](https://www.kaggle.com/datasets/bhavyabj/jud-ipl-indian-legal-judgement-prediction-dataset)

### InLegalBERT

Paul, S., Mandal, A., Goyal, P., & Ghosh, S. (2023). *Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law*. Proceedings of the 19th International Conference on Artificial Intelligence and Law, ICAIL 2023.

Model page:
[https://huggingface.co/law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)

```bibtex
@inproceedings{paul-2022-pretraining,
  url = {https://arxiv.org/abs/2209.06049},
  author = {Paul, Shounak and Mandal, Arpan and Goyal, Pawan and Ghosh, Saptarshi},
  title = {Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law},
  booktitle = {Proceedings of 19th International Conference on Artificial Intelligence and Law - ICAIL 2023},
  year = {2023}
}
```

---

## License

This repository is intended for academic and research use. Please respect the license and citation requirements of the Jud-IPL dataset and the InLegalBERT model.




[1]: https://huggingface.co/law-ai/InLegalBERT "law-ai/InLegalBERT · Hugging Face"
