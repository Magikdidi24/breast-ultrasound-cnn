# breast-ultrasound-cnn

Notebook where I train two models on breast ultrasound images with PyTorch: a ResNet-34 that classifies the image (normal / benign / malignant) and a U-Net that segments the lesion. There's also Grad-CAM to check where the classifier is looking.

This is v2. v1 trained on the original BUSI. Of its 780 images, 155 are copies, and my own check only caught 18 pairs, so its test scores were probably too optimistic. It also dropped hard on BUS-BRA, a dataset from another hospital. What v2 changes:

- Data: the Curated BUSI from Aumente-Maestro et al. (450 images), where the authors removed the duplicates and the problematic cases with an automatic search plus a manual review. My duplicate check still runs as a sanity check and finds nothing above its threshold.
- Evaluation: 5-fold cross-validation instead of one 70/15/15 split. Every image gets a prediction from a model that never trained on it, so the in-domain numbers use all 450 images instead of 114.
- Generalization: stronger augmentation (random crop with some aspect ratio change, zoom out into a black frame like a padded BUS-BRA image, gamma, blur, speckle-like noise). At test time each image is also flipped, and on BUS-BRA the 5 fold models are averaged.
- Threshold: besides taking the most likely class, I pick a threshold on P(malignant) that finds 90% of the malignant images in the validation sets, and check what it gives on the test folds and on BUS-BRA.

## Results

In-domain, out of fold on the 450 curated images: 82.9% accuracy (95% CI 79.3 to 86.0), ROC-AUC 0.939 and macro-F1 0.819 for the classifier, mean Dice 0.738 on the 386 lesions for the U-Net. The U-Net predicts a lesion on 21 of the 64 normal images (4 of 19 in v1). These are lower than v1 (88.6%, 0.98, Dice 0.81), but v1 probably had copies on both sides of its split and each v2 fold trains on about 300 images instead of 532, so the two aren't really comparable.

The external test is where v2 was supposed to help. Lesions only, since BUS-BRA has no normal images:

| | v1 BUSI test | v1 BUS-BRA | v2 curated BUSI | v2 BUS-BRA |
| --- | ---: | ---: | ---: | ---: |
| ROC-AUC malignant vs benign | 0.970 | 0.706 | 0.932 | 0.805 |
| malignant sensitivity | 0.806 | 0.473 | 0.823 | 0.590 |
| benign specificity | 0.938 | 0.811 | 0.892 | 0.845 |
| lesions predicted normal | 0.063 | 0.175 | 0.062 | 0.259 |
| mean Dice | 0.806 | 0.699 | 0.738 | 0.670 |

The classifier got clearly better on BUS-BRA: AUC 0.805 (95% CI 0.777 to 0.831, resampling patients) against 0.706 (0.674 to 0.735) for v1, so the two intervals don't overlap, and the gap with the in-domain AUC went from 0.26 to 0.13. Part of that comes from averaging the 5 models on BUS-BRA, which v1 didn't do, and from v1's BUSI score being inflated by the copies. With the threshold from the validation sets (0.213), it finds 81% of the malignant BUS-BRA lesions, but the benign specificity drops to 0.634. Two things got worse: 26% of the BUS-BRA lesions are now classified as normal (17.5% in v1), and the Dice on BUS-BRA went a bit down, 0.670 against 0.699.

## Running it

I ran v2 on Colab with a T4, about 23 min of training for the 10 models. A local CUDA GPU works too (an RTX 50 series card needs a torch build for CUDA 12.8 or newer), and it also runs on CPU, just slowly. The notebook downloads both datasets from Zenodo itself.

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
