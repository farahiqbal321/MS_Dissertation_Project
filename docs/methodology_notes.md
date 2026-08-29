## Clinical Data Preprocessing

The original patient information spreadsheet was loaded using the second spreadsheet row as the header because the first row contained a merged neurological examination title.

Column names were standardised for programmatic use. Categorical variables were cleaned to remove spacing and spelling inconsistencies. One invalid Gender value coded as "N" was treated as missing.

Disease duration was calculated as the difference between current age and age of onset. One patient produced a negative derived disease duration because the recorded age of onset exceeded the current age. The original values were retained, while the derived Disease_Duration value was marked as missing.

EDSS scores were grouped into Low, Moderate and High disability categories. A composite Total_Abnormal_Neuro_Findings feature was created by summing the binary neurological examination indicators.

The cleaned dataset contained 60 patients and 32 variables and was exported as cleaned_clinical_data.csv.

## Multimodal Dataset Integration

The cleaned clinical dataset was merged with the patient-level FLAIR MRI biomarker dataset using Patient_ID as the unique identifier.

A one-to-one merge validation confirmed that all 60 clinical records had a corresponding MRI record and that no unmatched or duplicate patient identifiers were present.

Nine MRI-derived biomarkers were integrated with the clinical variables, producing a final multimodal dataset containing 60 patients and 41 variables.

Two missing values remained after integration: one missing Gender value and one missing Disease_Duration value. These values were retained for later imputation within the model-training pipeline to reduce the risk of data leakage.

Lesion_Voxels and Lesion_Volume_mm3 were found to be identical because the FLAIR voxel volume was 1 mm³. Both were retained in the integrated dataset for provenance, while Lesion_Voxels will be removed during feature preparation before modelling.

## Feature Engineering

A binary disability target was created by retaining Low EDSS disability as class 0 and combining Moderate and High categories as class 1. This produced 35 Low-disability and 25 Moderate/High-disability records.

Variables directly defining or revealing the target were excluded from predictive modelling, including EDSS and EDSS_Category. Patient_ID was retained only for record tracking. Lesion_Voxels was excluded because it duplicated Lesion_Volume_mm3, while the MRI–EDSS timing flag was excluded because it represented a data-collection condition rather than a patient disease characteristic.

Three predictor sets were defined: 28 clinical features, eight MRI-derived biomarkers and 36 combined multimodal features. Missing Gender and Disease_Duration values were retained so that imputation could be fitted within cross-validation and training pipelines, thereby reducing data leakage.

# Experiment 1 – Baseline Model Comparison

## Objective

The first experiment establishes baseline predictive performance using three different feature sets:

- Clinical features only
- MRI-derived lesion biomarkers only
- Combined multimodal features

The purpose is to determine whether multimodal learning provides measurable improvements over using clinical or MRI information independently.

---

## Dataset

The final model-ready dataset contained:

- 60 patients
- 35 Low disability
- 25 Moderate/High disability

The High disability class was merged with the Moderate class because only two High-disability patients were available, making three-class classification statistically unreliable.

---

## Feature Sets

Three predictor sets were evaluated.

### Clinical

27 features

Included demographic variables, disease history and neurological examination findings.

### MRI

8 lesion-derived biomarkers

- Lesion Volume
- Lesion Count
- Average Lesion Size
- Largest Lesion Size
- Smallest Lesion Size
- Small Lesions
- Medium Lesions
- Large Lesions

### Multimodal

35 predictors

Combination of clinical variables and MRI biomarkers.

---

## Constant Feature Removal

The variable "Fields" contained only one unique value across all patients.

Because constant predictors contain no discriminative information, the variable was automatically removed before model training.

---

## Train/Test Split

A single stratified train-test split was used.

Training set

- 48 patients

Testing set

- 12 patients

The held-out test set was reserved exclusively for final model evaluation and was not used during model selection or hyperparameter optimisation.

---

## Cross-Validation

Five-fold stratified cross-validation was performed on the training data.

Stratification preserved the class proportions within every fold.

Cross-validation was used to estimate generalisation performance while reducing dependence on a single train-validation split.

---

## Leakage Prevention

To prevent data leakage:

- Missing values were imputed inside each cross-validation fold.
- Standardisation was fitted only using training folds.
- One-hot encoding was fitted independently within each fold.
- The held-out test set remained completely untouched during baseline model comparison.

---

## Baseline Models

Six supervised learning algorithms were evaluated.

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine

All models were trained using identical cross-validation procedures to ensure fair comparison.

---

## Evaluation Metrics

Performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Mean values and standard deviations across all folds were reported.

An overfitting indicator was also calculated using:

Training Accuracy − Validation Accuracy

Large differences indicate poor generalisation.

## Experiment 2 – Model Refinement

Experiment 2 refines four candidates selected from the baseline cross-validation results: Clinical Logistic Regression, Clinical Random Forest, Clinical Support Vector Machine and Multimodal XGBoost.

Candidate selection was based on mean cross-validation ROC-AUC, F1-score, variability and evidence of overfitting rather than held-out test performance alone.

Repeated stratified five-fold cross-validation with ten repeats will be used to obtain more stable estimates for the small cohort. Hyperparameters will be optimised using only the training patients. The previously defined test set will remain excluded throughout tuning.

## Experiment 2 – Tuned Model Evaluation

The optimised candidate pipelines were independently re-evaluated using repeated stratified five-fold cross-validation with ten repeats, producing 50 validation results per candidate. This stage was conducted separately from the hyperparameter search to determine whether tuning improvements generalised across repeated data partitions.

The tuned Clinical Random Forest achieved the strongest balanced performance, with a mean ROC-AUC of 0.8303, mean accuracy of 0.7464, mean recall of 0.7850 and mean F1-score of 0.7193. Its train–validation accuracy gap was reduced to 0.0947, indicating substantially lower overfitting than the untuned baseline.

Tuned Clinical Logistic Regression achieved a mean ROC-AUC of 0.8246 and F1-score of 0.6937, with the smallest positive train–validation gap of 0.0448. It was retained as a simpler and more interpretable comparator.

Although the tuned Support Vector Machine and multimodal XGBoost models produced ROC-AUC values above 0.80, their threshold-based classification results were poor. They were therefore not selected for the principal explainability analysis.

## Experiment 3 – Explainability Analysis

Experiment 3 investigates the factors influencing predictions from the tuned Clinical Random Forest selected during Experiment 2. The analysis uses SHAP to provide global and patient-level explanations, permutation importance as a model-agnostic global comparison, and LIME to generate local surrogate explanations for selected patient records.

The tuned Clinical Logistic Regression model is retained as a transparent comparator. Explainability analysis is performed on the clinical feature set because the clinical-only models outperformed the multimodal candidate during repeated cross-validation.

The objective is not to interpret model outputs as clinical diagnoses. Explanations are treated as decision-support evidence showing how recorded features influenced model predictions within this dataset.

## Experiment 3 model selection

The tuned Clinical Random Forest was selected as the main explainability model because it achieved the strongest balanced repeated cross-validation performance.

Tuned Clinical Logistic Regression was retained as an interpretable comparator because its performance was close to Random Forest and its coefficients provide a transparent linear reference.

Multimodal XGBoost was not selected for the principal explainability analysis because its threshold-based recall and F1-score were substantially weaker, despite a reasonable ROC-AUC.

## Explainability methods

SHAP, permutation importance and LIME will be used together because they provide complementary perspectives:

- SHAP provides global and local contribution estimates.
- Permutation importance measures the reduction in predictive performance after disrupting individual features.
- LIME produces locally interpretable surrogate explanations for selected patient predictions.

Agreement and disagreement between methods will be reported rather than assuming any single explanation method represents clinical truth.

# Experiment 3 – Explainability Analysis

## Objective

The objective of Experiment 3 was to investigate both global and local explainability of the selected machine learning model using multiple explainability techniques. The held-out test set remained untouched throughout this experiment to preserve an unbiased final evaluation.

## Selected model

Clinical Random Forest

Selected following Experiment 2 hyperparameter optimisation.

## Data used

Training dataset only (48 patients)

Held-out test dataset (12 patients) reconstructed but not used.

## Explainability methods

- SHAP TreeExplainer
- Global SHAP importance
- SHAP beeswarm visualisation
- SHAP waterfall plots
- SHAP decision plot
- LIME local explanations
- Permutation importance
- SHAP–LIME comparison
- SHAP–Permutation comparison

## Representative patients

Three representative patients were selected:

- Low Disability
- Moderate/High Disability
- Borderline prediction

The borderline case was selected as the patient with prediction probability closest to the classification threshold.

## Outputs generated

Figures:
- Global SHAP bar plot
- SHAP beeswarm plot
- Three SHAP waterfall plots
- SHAP decision plot
- Permutation importance plot
- Three LIME explanation plots

Tables:
- Global SHAP feature importance
- Patient SHAP values
- Selected patients
- LIME explanations
- SHAP-LIME agreement
- SHAP-Permutation comparison
- Experiment summary

## Key observations

Total_Abnormal_Neuro_Findings consistently appeared as the most influential predictor across SHAP, LIME and permutation importance.

The borderline patient demonstrated competing feature contributions, illustrating how explainability methods can support interpretation of uncertain predictions.

The held-out test set remained completely untouched.

