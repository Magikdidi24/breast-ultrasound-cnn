# breast-ultrasound-cnn

Notebook where I train two models on breast ultrasound images with PyTorch: a ResNet-34 that classifies the image (normal / benign / malignant) and a U-Net that segments the lesion. There's also Grad-CAM to check where the classifier is looking.

This branch is v2. What changed since v1 and why:

- Data: v1 trained on the original BUSI. Of its 780 images, 155 are copies, and my own check only caught 18 pairs. v2 trains on the Curated BUSI from Aumente-Maestro et al. (450 images), where the authors removed the duplicates and the problematic cases with an automatic search plus a manual review. My duplicate check still runs, but only as a sanity check.
- Evaluation: 5-fold cross-validation instead of one 70/15/15 split. Every image gets a prediction from a model that never trained on it, so the in-domain numbers use all 450 images instead of 114.
- Generalization: the v1 models dropped hard on BUS-BRA, so v2 has stronger augmentation (random crop with some aspect ratio change, zoom out into a black frame, gamma, blur, speckle-like noise). At test time each image is also flipped, and on BUS-BRA the 5 fold models are averaged.
- Threshold: besides taking the most likely class, I pick a threshold on P(malignant) that finds 90% of the malignant images in the validation sets, and check what it gives on the test folds and on BUS-BRA.
- It runs on a local GPU now instead of Colab.

The v2 results will be here after I run it on my RTX 5070 Ti.

## v1 results

These are from v1: original BUSI with 20 images dropped by my duplicate check, and a single split with 114 test images. The classifier got 88.6% accuracy and 0.98 ROC-AUC, and the U-Net a mean Dice of 0.81 on the lesions. On BUS-BRA it was much worse:

| lesions only, v1 | BUSI test | BUS-BRA |
| --- | ---: | ---: |
| ROC-AUC malignant vs benign | 0.970 | 0.706 |
| malignant sensitivity | 0.806 | 0.473 |
| benign specificity | 0.938 | 0.811 |
| mean Dice | 0.806 | 0.699 |

The biggest drop was on finding the malignant lesions, and 17.5% of the BUS-BRA lesions were classified as normal. Because of the copies left in BUSI, the BUSI column is probably optimistic.

## Running it

You need a GPU with CUDA to train it in a reasonable time (it also runs on CPU, just slowly). An RTX 50 series card needs a torch build for CUDA 12.8 or newer. The notebook downloads both datasets from Zenodo itself.

```bash
pip install -r requirements.txt jupyter
jupyter notebook breast_cancer_detection_segmentation.ipynb
```

If you already have the datasets, set `BUSI_DIR` (the curated BUSI) / `BUSBRA_DIR` in the config cell. `RUN_EXTERNAL_TEST = False` skips BUS-BRA.

## Datasets

- Curated BUSI: Aumente-Maestro et al., *A multi-task framework for breast cancer segmentation and classification in ultrasound imaging*, Computer Methods and Programs in Biomedicine, 2025. The data is [on Zenodo](https://doi.org/10.5281/zenodo.19047974) (CC BY 4.0)
- BUSI, the original used in v1: Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief, 2020
- BUS-BRA: Gómez-Flores et al., *BUS-BRA: A breast ultrasound dataset for assessing computer-aided diagnosis systems*, Medical Physics, 2024 (CC BY 4.0)

They're not included in the repo.

Not a medical tool, just a learning project.
