# breast-ultrasound-cnn

Notebook where I train two models on breast ultrasound images with PyTorch: a ResNet-34 that classifies the image (normal / benign / malignant) and a U-Net that segments the lesion.

Training is on BUSI. Of its 780 images, 155 are copies, which inflates the scores if they end up in both train and test. My check only catches 18 pairs and drops 20 images (it misses cropped or zoomed copies). Then I test on the BUSI test split and on BUS-BRA, a dataset from another hospital that the models never see during training. There's also Grad-CAM to check where the classifier is looking.

## Results

On the BUSI test split (114 images) the classifier gets 88.6% accuracy and 0.98 ROC-AUC, and the U-Net gets a mean Dice of 0.81 on the lesions. On BUS-BRA it's much worse:

| lesions only | BUSI test | BUS-BRA |
| --- | ---: | ---: |
| ROC-AUC malignant vs benign | 0.970 | 0.706 |
| malignant sensitivity | 0.806 | 0.473 |
| benign specificity | 0.938 | 0.811 |
| mean Dice | 0.806 | 0.699 |

The biggest drop is on finding the malignant lesions. The notebook has the confidence intervals and the per-class numbers.

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

Not a medical tool, just a learning project.
