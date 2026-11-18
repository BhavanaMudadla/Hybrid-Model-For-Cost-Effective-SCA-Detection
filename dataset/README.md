
# Dataset

## Dataset Overview

The dataset used in this project consists of 1,052 peripheral blood smear images collected from multiple publicly available sources. Since individual online datasets contained relatively limited numbers of images, images from multiple sources were consolidated to construct a larger dataset for Sickle Cell Anemia classification.

## Original Data Sources

The images were collected from the following sources:

1. [Red Blood Cell RedTell Dataset – dse_data.zip](https://zenodo.org/records/7801430/files/dse_data.zip?download=1)
2. [Red Blood Cell RedTell Dataset – anemia_data.zip](https://zenodo.org/records/7801430/files/anemia_data.zip?download=1)
3. [erythrocytesIDB 2017 – Version 2](https://zenodo.org/records/18299474/files/Version%202,%20erythrocytesIDB%202017.zip?download=1)
4. [Sickle Cell Disease Dataset – Kaggle](https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset)

## Dataset Consolidation

The downloaded datasets contained images that were already organised into categories. The images were combined based on their existing class labels into three common folders.

The source identity of individual images was not retained during consolidation. Therefore, source-wise image counts are not available.

## Final Dataset Distribution

| Category | Number of Images |
|---|---:|
| Negative (Normal) | 184 |
| Positive (Labelled) | 422 |
| Positive (Unlabelled) | 446 |
| **Total** | **1,052** |

For binary classification, the two positive categories were treated as a single positive class.

| Binary Class | Number of Images |
|---|---:|
| Negative | 184 |
| Positive | 868 |
| **Total** | **1,052** |

## Folder Organisation

```text
dataset/
├── negative/
├── positive_labelled/
└── positive_unlabelled/
```

## Important Note

This repository documents the consolidated dataset and its preparation methodology. Original image redistribution is subject to the respective dataset licenses and permissions. Refer to the original sources before downloading or reusing the images.
