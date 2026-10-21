# Breast ultrasound CNN

Classification (normal / benign / malignant) and lesion segmentation on breast ultrasound images, in PyTorch.

- ResNet-34 classifier and a U-Net with a ResNet-34 encoder
- trained on BUSI after removing its duplicate images (155 of the 780 are copies)
- tested on the BUSI test split and on BUS-BRA, a dataset from another hospital
- Grad-CAM to check what the classifier actually looks at

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Magikdidi24/breast-ultrasound-cnn/blob/main/breast_cancer_detection_segmentation.ipynb)

## Running it

Easiest is Colab with a T4 GPU. Both datasets are downloaded by the notebook (BUSI from Kaggle with `kagglehub`, BUS-BRA from Zenodo). Locally:

```bash
pip install -r requirements.txt jupyter
jupyter notebook breast_cancer_detection_segmentation.ipynb
```

If the datasets are already on disk, set `BUSI_DIR` and `BUSBRA_DIR` in the config cell. `RUN_EXTERNAL_TEST = False` skips BUS-BRA.

## Data

- BUSI: W. Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief, 2020.
- BUS-BRA: W. Gómez-Flores et al., *BUS-BRA: A breast ultrasound dataset for assessing computer-aided diagnosis systems*, Medical Physics, 2024 (CC BY 4.0).

The datasets are not included in this repo.

Research and teaching only, not a medical device.
