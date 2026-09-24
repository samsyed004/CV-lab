# Lab 03 — Edge Detection & Its Effect on Classification

This is the code for Lab 03, which builds on Lab 01 and Lab 02. It covers edge detection (Sobel, Prewitt, Laplacian, LoG, Canny), checks how noise messes with each of them, tunes Canny's thresholds, and then compares classification performance on raw vs. filtered vs. edge-only images.

Same dataset as before: HAM10000, same classes you picked in Lab 01/02.

## What's in here

- `lab03_edge_detection.ipynb` — the whole thing, all 6 tasks
- `task1_edge_comparison.png` — the required figure: Original → Sobel → Prewitt → Laplacian → LoG → Canny, for 3 classes
- `task1_sobel_gx_gy.png` — Sobel Gx vs Gy side by side
- `task2_noise_effect.png` — how each detector holds up under noise, before/after filtering
- `task2_edge_density_proxy.csv` — a rough number (edge pixel %) to help you judge noise sensitivity instead of just eyeballing it
- `table1_template.csv` — Table 1 with the columns set up, you just fill in the judgment calls (edge quality, noise sensitivity, observations)
- `task3_canny_params.png` + `table2_canny_params.csv` — the Canny threshold sweep, with edge counts for each config
- `task4_set_comparison.png` — raw vs filtered vs edge, side by side
- `table3_docx_format.csv` and `table3_full_long_format.csv` — the classification comparison, in the layout the doc wants and also a more detailed version
- `task6_confusion_matrices.png`, `task6_performance_bars.png` — visual comparison for whichever model ends up winning
- `lab03_results.xlsx` — everything above, one Excel file

## Running it

1. Open the notebook in Colab, GPU on.
2. Kaggle secrets should already be set up if you did Lab 01/02 in this account — if not, same steps as before (kaggle.com/settings → API → new token → add `KAGGLE_USERNAME`/`KAGGLE_KEY` as Colab secrets).
3. Check `SELECTED_CLASSES` near the top matches what you actually used in Lab 01/02.
4. Run through Tasks 1-3 first. This is where you'll need to actually look at the images and fill in Table 1 and Table 2 yourself — the notebook gives you numbers to back up your answers (edge density, edge counts) but "is this a good edge map" is still something you have to judge by eye.
5. Once you've decided on the best Canny setup and remembered what filter won in Lab 02, set `BEST_CANNY_CONFIG` and `BEST_FILTER` accordingly.
6. Run the rest — it trains 5 models (SVM, Random Forest, KNN, and 2 CNNs) on all 3 versions of the dataset and spits out Table 3, confusion matrices, and a bar chart at the end.
7. Excel file downloads automatically when it's done.

## A few things to know

- Only using 300 images per class by default, otherwise 15 separate training runs (5 models × 3 datasets) would take forever on Colab's free GPU. Bump it up if you've got the time/patience.
- CNN Model 2 (ResNet18) only trains its final layer, same trick as Lab 02, to keep things fast. CNN Model 1 trains from scratch though, so it's genuinely slower to converge — don't be surprised if it underperforms a bit with only 5 epochs.
- The original Table 3 template only has one column each for precision/recall/f1/times (not split by raw/filtered/edge), so I filled those specifically using the Edge results since that's what this lab is actually about. The full breakdown across all three is in the other CSV if you want it for your discussion.

## For the write-up

There's a section at the end of the notebook that basically points at which variable/output answers each of the 7 discussion questions, so you're not stuck digging back through everything when you write the report.
