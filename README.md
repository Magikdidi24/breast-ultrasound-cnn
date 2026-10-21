# breast-ultrasound-cnn

Notebook where I train two models on breast ultrasound images with PyTorch: a ResNet-34 that classifies the image (normal / benign / malignant) and a U-Net that segments the lesion.

Training is on BUSI, after removing its duplicate images (155 of the 780 are copies, which inflates the scores if they end up in both train and test). Then I test on the BUSI test split and on BUS-BRA, a dataset from another hospital that the models never see during training. There's also Grad-CAM to check where the classifier is looking.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Magikdidi24/breast-ultrasound-cnn/blob/main/breast_cancer_detection_segmentation.ipynb)

## Running it

Easiest is Colab with a T4 GPU, the notebook downloads both datasets itself (BUSI from Kaggle, BUS-BRA from Zenodo). Locally:

```bash
pip install -r requirements.txt jupyter
jupyter notebook breast_cancer_detection_segmentation.ipynb
```

If you already have the datasets, set `BUSI_DIR` / `BUSBRA_DIR` in the config cell. `RUN_EXTERNAL_TEST = False` skips BUS-BRA.

## Datasets

- BUSI: Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief, 2020
- BUS-BRA: Gómez-Flores et al., *BUS-BRA: A breast ultrasound dataset for assessing computer-aided diagnosis systems*, Medical Physics, 2024 (CC BY 4.0)

They're not included in the repo.

Not a medical tool, just a research / learning project.
