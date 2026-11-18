
"""Paper-sourced content for the Hybrid SCD research prototype."""

TITLE = "Hybrid Model for Cost-Effective Sickle Cell Anemia Detection"

SUBTITLE = (
    "Image-based binary classification using EfficientNet-B0 "
    "and an OptiSCDNet-style residual CNN"
)

AFFILIATION = "Final-year research project"

DISCLAIMER = (
    "This application is an academic research prototype. Its output "
    "must not be used to diagnose, rule out, or make treatment decisions "
    "about sickle cell disease. Clinical confirmation requires appropriate "
    "medical testing."
)

BACKGROUND = (
    "This project investigates blood-smear image classification for "
    "sickle cell anemia using a dual-branch deep-learning architecture. "
    "The proposed hybrid combines an ImageNet-pretrained EfficientNet-B0 "
    "branch with a task-specific residual CNN branch. Their extracted "
    "features are concatenated and passed through a classification head "
    "that produces a sigmoid probability for the positive (sickle) class."
)

CONTRIBUTIONS = [
    "A dual-branch architecture combining EfficientNet-B0 and an "
    "OptiSCDNet-style residual CNN.",
    "Image preprocessing using resizing, median filtering and "
    "Y-channel histogram equalization.",
    "A duplicate-aware, group-stratified image split.",
    "Evaluation using accuracy, precision, sensitivity, specificity, "
    "F1-score, ROC-AUC, PR-AUC and a confusion matrix.",
    "A local single-image prediction demonstration."
]

LIMITATIONS = [
    "The data split is by image, not by patient or slide. "
    "Patient-level leakage may inflate results.",
    "Annotated positive images may introduce shortcut learning. "
    "Exact hashing cannot remove annotated near-duplicates.",
    "Each model was run once, without cross-validation.",
    "The normal class has only 26 images in the proposed hybrid test set.",
    "Test sets differ across experiments, so comparisons are indicative "
    "rather than strict rankings.",
    "No severity grading was achieved because severity labels were "
    "not available.",
    "CBC and morphological feature branches are not integrated.",
    "There is no external validation or clinical validation."
]

# All values below are reported for the proposed hybrid in the paper.
# The positive class is sickle; threshold = 0.5.
HYBRID_METRICS = {
    "Accuracy": "85.3%",
    "Sensitivity / Recall": "86.2%",
    "Specificity": "80.8%",
    "Precision": "95.7%",
    "F1-score": "90.7%",
    "ROC-AUC": "0.922",
    "PR-AUC": "0.979",
    "Test images": "156 (26 normal, 130 sickle)",
    "Accuracy 95% Wilson CI": "78.8–90.0%"
}

HYBRID_CM = {
    "TN": 21,
    "FP": 5,
    "FN": 18,
    "TP": 112
}

ALWAYS_POSITIVE_BASELINE = "83.3%"

NORMAL_CLASS = (
    "Precision 0.54, recall 0.81 and F1-score 0.65 "
    "(macro-F1 0.78)."
)

# Notebook Section 12: subgroup results.
SUBGROUP = [
    ["negative", 26, 21, "80.8%"],
    ["positive_unlabelled", 60, 47, "78.3%"],
    ["positive_labelled", 70, 65, "92.9%"]
]

COMPARISON_COLUMNS = [
    "Model",
    "Test images",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
    "Parameters"
]

COMPARISON = [
    ["VGG-16", "95", "86.3%", "95.0%", "85.1%", "89.8%", "0.943", "15.1M"],
    ["DenseNet-121", "95", "90.5%", "92.7%", "94.0%", "93.3%", "0.953", "8.75M"],
    ["InceptionV3", "91", "96.7%", "95.5%", "100%", "97.7%", "1.000", "21.8M"],
    ["EfficientNet-B0", "158", "96.2%", "96.3%", "99.2%", "97.7%", "0.973", "4.05M"],
    ["OptiSCD (MobileNetV2 variant)", "158", "93.0%", "97.6%", "93.9%", "95.7%", "0.960", "—"],
    ["DenseNet-121 + VGG-16 fusion", "106", "88.7%", "94.1%", "88.9%", "91.4%", "0.943", "22.15M"],
    ["OptiSCD + EfficientNet fusion", "149", "85.2%", "99.1%", "83.5%", "90.6%", "0.971", "4.96M"],
    ["Proposed hybrid", "156", "85.3%", "95.7%", "86.2%", "90.7%", "0.922", "4.96M"]
]

COMPARISON_NOTE = (
    "The models were evaluated on different test sets and, in some cases, "
    "different training subsets. These results are indicative, not a strict "
    "ranking. The proposed hybrid does not demonstrate a measurable gain "
    "from fusion in this experiment."
)

ARCHITECTURE_TEXT = (
    "The proposed hybrid has two parallel feature-extraction branches. "
    "Branch A uses ImageNet-pretrained EfficientNet-B0. Branch B is a "
    "custom residual CNN inspired by OptiSCDNet, with separable convolutions "
    "and squeeze-and-excitation channel attention. Their 1280-dimensional "
    "and 256-dimensional features are concatenated into a 1536-dimensional "
    "representation before the classification head. The final layer is a "
    "single sigmoid output representing the probability of the positive "
    "sickle class."
)

TRAINING_TEXT = [
    "Input image size: 224 × 224 × 3.",
    "Batch size: 16.",
    "Two training phases: frozen EfficientNet backbone followed by "
    "fine-tuning of its last 40 layers.",
    "Phase 1 learning rate: 1e-3; phase 2 learning rate: 2e-5.",
    "Validation AUC was used for checkpoint selection.",
    "The saved hybrid model contains 4,955,772 parameters.",
    "The notebook reports a 0.5 decision threshold; no threshold tuning "
    "was performed for the proposed hybrid."
]
