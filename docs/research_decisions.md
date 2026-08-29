## Duplicate lesion representation

Lesion_Voxels and Lesion_Volume_mm3 contained identical values because the FLAIR voxel dimensions were 1 × 1 × 1 mm. Lesion_Volume_mm3 was retained as the clinically interpretable measure, while Lesion_Voxels will be excluded from predictive modelling to avoid duplicate information.

## Binary disability target

The High EDSS group contained only two patients, making a three-class classification task statistically unstable. Moderate and High disability categories were therefore combined into one class, producing a binary target of Low versus Moderate/High disability.

## Constant feature

The Fields variable contained only one unique value across the dataset. It will be removed during model preparation because constant variables provide no discriminatory information.

## Composite neurological feature

Total_Abnormal_Neuro_Findings was retained for baseline modelling alongside its component neurological indicators. Its potential redundancy will be examined during feature refinement rather than removed before the baseline experiment.

# Research Decision Log

## Binary Classification

The original EDSS categories were Low, Moderate and High.

Because only two High-disability patients were available, Moderate and High were combined into a single class.

Final prediction task:

Low disability

versus

Moderate/High disability

Reason:

Improves statistical stability and avoids severe class imbalance.

---

## MRI Feature Selection

Only FLAIR lesion segmentation masks were used for predictive modelling.

Although T1 and T2 scans were available, previous literature suggests FLAIR imaging provides the most clinically relevant lesion information for disability assessment in Multiple Sclerosis.

The remaining modalities were retained only for dataset validation.

---

## Removal of Constant Variables

The variable "Fields" contained only one unique value across the dataset.

It was removed because constant predictors cannot contribute to classification performance.

---

## Removal of EDSS

EDSS was removed from the predictor variables.

Reason:

EDSS defines the target class.

Including EDSS would introduce target leakage.

---

## Lesion Volume

Lesion_Voxels and Lesion_Volume_mm3 were identical because voxel dimensions were 1 × 1 × 1 mm.

Lesion_Volume_mm3 was retained because it is clinically interpretable.

Lesion_Voxels was removed before modelling.

---

## Model Comparison Strategy

Baseline experiments compared:

Clinical-only models

MRI-only models

Multimodal models

using identical preprocessing and cross-validation procedures.

This allows direct comparison between modalities while controlling for algorithmic differences.

---

## Cross-Validation

Five-fold stratified cross-validation was selected because the dataset contains only 60 patients.

Using repeated train-validation splits provides a more reliable estimate of model performance than a single validation split.

---

## Hold-Out Test Set

A fixed 20% hold-out test set was created before model training.

The test set will not be accessed until all model development and hyperparameter optimisation have been completed.

This prevents optimistic performance estimates.

---

## Experiment 1 Findings

Initial baseline results indicate:

Clinical features consistently outperform MRI biomarkers.

MRI biomarkers alone show limited predictive capability.

Multimodal learning provides competitive performance but does not consistently exceed the strongest clinical models.

Several tree-based models demonstrate substantial train-validation performance gaps, indicating potential overfitting.

These observations motivate Experiment 2, which focuses on feature refinement, hyperparameter optimisation and improving model generalisation rather than simply increasing training accuracy.

## Experiment 2 candidate selection

MRI-only models were not advanced to extensive tuning because their mean cross-validation ROC-AUC values were close to chance.

Four candidates were retained:

- Clinical Logistic Regression
- Clinical Random Forest
- Clinical Support Vector Machine
- Multimodal XGBoost

These candidates represent different model families and preserve a direct clinical-versus-multimodal comparison.

The held-out test set will not be used during feature selection or hyperparameter optimisation.

## Final candidate selection after model refinement

The tuned Clinical Random Forest was selected as the primary predictive model because it achieved the strongest combination of repeated cross-validation ROC-AUC, accuracy, recall and F1-score.

The tuned Clinical Logistic Regression model was retained as an interpretable comparator because its ROC-AUC was close to that of Random Forest and it demonstrated the smallest train–validation performance gap.

Clinical Support Vector Machine and multimodal XGBoost were not advanced because their default-threshold precision, recall and F1-scores were substantially weaker despite reasonable ROC-AUC values.

## Multimodal modelling finding

The multimodal XGBoost model did not outperform the leading clinical-only models. This suggests that the available handcrafted FLAIR lesion biomarkers did not provide sufficient additional discriminatory information for disability classification in this 60-patient cohort.

This outcome will be reported as a research finding rather than interpreted as evidence that MRI is generally unhelpful. Potential explanations include the limited cohort size, the use of lesion-summary biomarkers rather than spatial or deep imaging representations, and the imperfect relationship between lesion burden and clinical disability.

# Experiment 3 Research Decisions

## Explainability methods

Decision:
Use SHAP, LIME and permutation importance rather than relying on a single explainability technique.

Reason:
Different explainability techniques provide complementary information. SHAP provides theoretically consistent feature attribution, LIME explains individual predictions locally, and permutation importance provides model-agnostic global importance.

---

## Training data only

Decision:
Perform all explainability analyses using the reconstructed training dataset.

Reason:
The held-out test dataset must remain untouched until final model evaluation to avoid information leakage.

---

## Representative patients

Decision:
Select three representative patients rather than explaining every patient.

Reason:
This provides examples of confident low-risk, confident high-risk and borderline predictions while keeping the analysis manageable and clinically interpretable.

---

## Local explainability

Decision:
Generate both SHAP waterfall plots and LIME explanations for each representative patient.

Reason:
Using two local explanation methods allows qualitative comparison of feature contributions and provides stronger evidence that explanations are consistent.

---

## Global explainability

Decision:
Compare SHAP with permutation importance.

Reason:
Agreement between two independent global importance methods increases confidence that important features are not artefacts of a single explainability technique.

---

## Agentic AI deferred

Decision:
Do not generate natural-language clinical explanations during Experiment 3.

Reason:
Experiment 3 focuses on explainability techniques only. Narrative evidence-based explanations are reserved for Experiment 4.

