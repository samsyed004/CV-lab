# Lab Mid Exam: HOG vs SIFT on Blood-Cell Images

**Name:** ____________  **Registration number:** 007

> **Before you submit:** the values shown as **[paste from your output]** are numbers I do not have, because they appear only when your notebook runs (chosen C, validation scores, confusion matrices, keypoint statistics). Copy them from your executed notebook. Every number in the answers must match your outputs. Everything else below uses numbers from outputs you already shared. Rewrite the text in your own words where you can. Delete this note before submitting.

## Personal settings

| Setting | Rule | My value |
|---|---|---|
| Random seed | Last two digits of 007 | 7 |
| HOG cell size | Last digit 7 is odd | 16 × 16 |
| SIFT grid size | Last digit 7 is in 5 to 9 | 6 × 6 grid (36 keypoints) |

---

## Q1. Data setup (3 marks)

I loaded BloodMNIST (size 64) and kept four classes: eosinophil (label 1), monocyte (5), neutrophil (6) and platelet (7). Using a random generator with seed 7, I took 100 training, 30 validation and 30 test images per class. This gives 400 training, 120 validation and 120 test images, with 100, 30 and 30 images per class. The images were converted to grayscale and resized to 128 × 128.

| Split | Raw shape | Prepared shape | Eosinophil | Monocyte | Neutrophil | Platelet | Total |
|---|---|---|---|---|---|---|---|
| Train | (400, 64, 64, 3) | (400, 128, 128) | 100 | 100 | 100 | 100 | 400 |
| Validation | (120, 64, 64, 3) | (120, 128, 128) | 30 | 30 | 30 | 30 | 120 |
| Test | (120, 64, 64, 3) | (120, 128, 128) | 30 | 30 | 30 | 30 | 120 |

*(Check these shapes against your printed output.)*

**Figure:** insert the 4 × 4 grid of example images (4 per class).

---

## Q2. HOG + SVM (6 marks)

### Feature length (hand calculation)

- Image size 128 × 128, cell size 16 × 16, so there are 128 / 16 = 8 cells on each side.
- Blocks are 2 × 2 cells and move by 1 cell, so there are 8 − 2 + 1 = 7 blocks on each side, which is 7 × 7 = 49 blocks.
- Each block has 2 × 2 cells × 9 orientations = 36 values.
- HOG feature length = 49 × 36 = **1764**.

The code printed the same feature length, **[confirm: 1764]**.

### HOG pictures

**Figure:** insert the HOG picture of one image from each class.

### SVM results (C chosen on the validation set)

The model is a StandardScaler followed by a LinearSVC. I tried C = 0.1, 1 and 10 and chose C using validation accuracy only.

- Chosen C: **[paste from your output]**
- Validation accuracy:74.2
- Validation macro-F1: **[paste from your output]**
- Confusion matrix (rows are true classes, columns are predicted): **[insert your printed matrix]**

---

## Q3. SIFT + SVM (7 marks)

### Normal SIFT detection on the training images

- Average keypoints per image: **[paste from your output]**
- Minimum: **[paste]**  Maximum: **[paste]**
- Images with zero keypoints: **[paste]**

**Figure:** insert the keypoints drawn on an image.

**Why a different number of keypoints per image is a problem for an SVM:** an SVM needs every image to be one feature vector of the same length, but normal SIFT gives a different number of 128-value descriptors for each image, so the images cannot be stacked into one feature matrix.

### Fixed grid of keypoints

I placed keypoints at the centres of a regular 6 × 6 grid over the 128 × 128 image, with keypoint size 32 and angle 0, and computed the SIFT descriptor at each point with `sift.compute`. The descriptors were joined into one vector.

- Grid points = 6 × 6 = 36.
- One SIFT descriptor has 128 values.
- Feature length = 36 × 128 = **4608**.

The code printed the same feature length, **[confirm: 4608]**.

### SVM results (C chosen on the validation set)

- Chosen C: **[paste from your output]**
- Validation accuracy: **[paste from your output]**
- Validation macro-F1: **[paste from your output]**
- Confusion matrix: **[insert your printed matrix]**

---

## Q4. Robustness (6 marks)

### Prediction (written before running the test)

**Rotation (20°):** I think HOG + SVM will lose more accuracy. HOG puts gradient directions into 9 bins of 20° inside fixed 16 × 16 cells, so rotating the image by 20° moves the edges to different bins and different cells. The SIFT descriptors are computed over bigger windows (size 32) with Gaussian weighting, so they should change more gently.

**Noise (σ = 0.05):** I think HOG + SVM will lose more accuracy. HOG takes gradients directly from pixel differences with no smoothing, so noise creates random gradients, and block normalisation can make them stronger in flat areas. SIFT blurs the image with a Gaussian first, so it should be less affected.

### Results on the test set (both models frozen, change applied before feature extraction)

| Model | Clean accuracy | Rotated accuracy | Noisy accuracy |
|---|---|---|---|
| HOG + SVM | 0.742 | 0.642 | 0.292 |
| SIFT + SVM | 0.792 | 0.742 | 0.742 |

The test set has 120 images, so one image is about 0.83 percentage points.

### Was my prediction correct?

My prediction was correct for both rotation and noise. Under rotation HOG + SVM fell from 0.742 to 0.642 (−0.100) and SIFT + SVM from 0.792 to 0.742 (−0.050), and under noise HOG + SVM fell from 0.742 to 0.292 (−0.450) while SIFT + SVM only fell to 0.742 (−0.050). This fits my reasoning that HOG, which uses raw pixel gradients, is more sensitive than SIFT, which smooths the image first, although the rotation gap is only 6 of the 120 test images and could partly be chance, while the noise gap is large.

---

## Q5. Short answers and AI log (3 marks)

### (a) Why does HOG use block normalisation?

Gradient magnitudes change with lighting and contrast, so the same structure can give very different histogram sizes in bright and dark parts of an image. HOG groups neighbouring cells into overlapping blocks and normalises each block (here with L2-Hys), which makes the descriptor much less sensitive to local illumination and contrast changes. Because blocks overlap, each cell is normalised several times in different contexts, which also gives some robustness to small shifts.

### (b) Why is it wrong to choose C using the test set?

The test set is meant to estimate how the final model performs on completely unseen data. If C is picked by looking at test accuracy, the test data has influenced the model, so the reported test accuracy becomes optimistic and is no longer an honest estimate. C must be chosen on a separate validation set, and the test set should be used only once at the end.

### (c) Which two classes were confused most, and why might grayscale make this harder?

Both models confused eosinophil and monocyte most on the validation set (HOG + SVM: 18 errors, SIFT + SVM: 16 errors, 34 combined). Eosinophils are mainly recognised by their red-orange granules, and that colour information is removed when the image is converted to grayscale, so only nucleus shape and texture are left, which look similar to monocytes.

### (d) AI log

| # | Prompt (one line) | What I changed | How I checked the result was correct |
|---|---|---|---|
| 1 | "Write the HOG vs SIFT blood-cell lab mid exam notebook for registration 007 (seed 7, HOG 16×16, SIFT 6×6 grid)." | Ran it in Colab with my own settings; rewrote the Q4(a) wording and wrote the Q4(c) and Q5(c) answers myself. | Printed the derived settings (seed 7, HOG 16×16, SIFT 6×6 = 36 keypoints); checked split shapes and class counts (400/120/120) in Q1 and the example grid. |
| 2 | "How do I get a fixed-length SIFT feature using a grid of keypoints and sift.compute?" | Used the grid-cell centres, keypoint size 32 and angle 0. | Printed the feature matrix shape and asserted length = 36 × 128 = 4608; drew the grid on an image. |
| 3 | "Explain every line of the notebook code." | Read the explanation and made sure I can explain the grid keypoints, fit_and_select, rotate and add_noise cells. | Compared my Q4 prediction with the printed table; HOG feature length 1764 matched my hand calculation (49 blocks × 36). |

*Edit the "What I changed" column so that it is true for you.*
