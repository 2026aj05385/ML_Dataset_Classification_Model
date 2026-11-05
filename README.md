# ML_Dataset_Classification_Model
All the 5 ML models have to be implemented on the same dataset. 1. Logistic Regression 2. Decision Tree Classifier 3. K-Nearest Neighbor Classifier 4. Naive Bayes Classifier - Gaussian or Multinomial 5. Support Vector Machine
# 🌾 Paddy Productivity Classification (Low / Medium / High)

## a. Problem Statement

Farmers and agricultural planners would like to know, *before* harvest, whether a paddy plot is likely to deliver a **Low, Medium or High yield per hectare**, given how it was managed (seed rate, fertiliser and pesticide doses, nursery type, variety, soil) and the weather it experienced (rainfall, irrigation, temperature, humidity, wind).

The source data records a continuous yield in Kg, so the task is reformulated as a **3-class classification problem**:

```
yield per hectare = Paddy yield (Kg) / Hectares
Low    : yield per hectare <  5,855 Kg/ha      (bottom third)
Medium : 5,855 ≤ yield per hectare ≤ 6,181     (middle third)
High   : yield per hectare >  6,181 Kg/ha      (top third)
```

Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) are trained on the same data, evaluated with six metrics, saved, and served through a Streamlit web app.

*Why yield per hectare and not raw yield?* Raw yield is ~99.5 % correlated with the `Hectares` column alone, so binning it would make every model near-perfect and the comparison meaningless. Yield per hectare removes plot size and makes the problem genuinely about agronomy and weather.

## b. Dataset Description

| Item | Detail |
|---|---|
| Dataset | Paddy yield dataset (Kaggle) – plots across six agriblocks of Tamil Nadu (Cuddalore, Kurinjipadi, Panruti, Kallakurichi, Sankarapuram, Chinnasalem) |
| Source link | `<paste the Kaggle URL you downloaded it from>` |
| Raw size | 2,789 rows × 45 columns |
| After cleaning | **2,338 rows** (451 exact duplicate rows removed so identical rows cannot appear in both train and test) |
| Input features | **44** (36 numeric, 8 categorical) – meets the ≥ 12 features / ≥ 500 rows requirement |
| Target | `Yield_Class` – Low (795), Medium (801), High (742): well balanced |
| Missing values | None |
| Train / test split | 80 / 20, stratified, `random_state=11` → 1,870 train rows, **468 test rows** (`test_data.csv`) |

**Feature groups**

- *Plot and crop*: `Hectares`, `Agriblock`, `Variety` (CO_43, ponmani, delux ponni), `Soil Types` (clay, alluvial), `Nursery` (dry, wet)
- *Inputs applied*: seed rate, main-field and nursery land-preparation quantities, nursery area, DAP (day 20), weedicide (day 28), urea (day 40), potash (day 50), pesticide (day 60), micronutrients (day 70), trash bundles
- *Rainfall and irrigation* for four growth windows (days 1–30, 30–50, 51–70, 71–105)
- *Temperature* (min and max), *wind speed*, *wind direction* (categorical) and *relative humidity* for four 30-day periods

**Preprocessing** (stored inside each saved model as a scikit-learn `Pipeline`, so the app applies exactly the same transformations): numeric features → `StandardScaler`; categorical features → `OneHotEncoder(handle_unknown="ignore")`. Hyper-parameters of every model were tuned with 5-fold stratified `GridSearchCV` on the training split only (scoring = macro-F1, chosen so that a model is not rewarded for ignoring the Medium class).

| Model | Best hyper-parameters |
|---|---|
| Logistic Regression | C = 0.1 |
| Decision Tree | max_depth = 5, min_samples_leaf = 5 |
| KNN | n_neighbors = 11, weights = uniform |
| Naive Bayes (Gaussian) | var_smoothing = 1e-9 |
| SVM (RBF) | C = 10, gamma = "scale", probability = True |

## c. GitHub Repository Link

🔗 **Repository:** `https://github.com/2026aj05385/ML_Dataset_Classification_Model`
🚀 **Live app:** `https://mlclassificationmodels-ob3bdq8ixkjnfgqf5nfvhw.streamlit.app/`

```
project-folder/
├── streamlit_app.py        # Streamlit front-end (loads saved models, no retraining)
├── train_models.py         # data prep, tuning, evaluation, model saving
├── requirements.txt
├── README.md
├── test_data.csv           # 468 held-out rows (44 features + Yield_Class)
├── metrics_results.csv     # the comparison table below, as produced by train_models.py
├── .gitignore
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib   # each file = preprocessing + classifier pipeline
    └── metadata.joblib                 # feature list, class order, class cut-offs, versions
```

**Run locally**
```bash
pip install -r requirements.txt
python train_models.py --data paddydataset.csv   # re-creates model/, test_data.csv, metrics_results.csv
streamlit run streamlit_app.py
```

## d. Models Used

All five models use the same data, the same split and the same preprocessing. Metrics are computed on the 468-row test set. Precision, Recall and F1 are **weighted averages** over the three classes; AUC is **one-vs-rest, macro-averaged**; MCC is the multi-class Matthews correlation coefficient.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.4872 | 0.6610 | 0.4774 | 0.4872 | 0.4786 | 0.2347 |
| Decision Tree | **0.5598** | 0.6659 | 0.4244 | **0.5598** | 0.4517 | **0.4033** |
| KNN | 0.5085 | 0.6585 | 0.4692 | 0.5085 | 0.4704 | 0.2802 |
| Naive Bayes | 0.5214 | 0.6745 | 0.4766 | 0.5214 | **0.4800** | 0.3011 |
| Support Vector Machine | 0.5427 | **0.6858** | 0.4552 | 0.5427 | 0.4545 | 0.3606 |

*(Random guessing on three balanced classes gives about 0.33 accuracy, 0.50 AUC and 0.00 MCC, so every model learns something, but this is a hard problem.)*

Per-class F1 on the test set (Low / Medium / High): Logistic Regression 0.61 / 0.26 / 0.57; Decision Tree 0.68 / 0.01 / 0.68; KNN 0.64 / 0.16 / 0.59; Naive Bayes 0.66 / 0.17 / 0.62; SVM 0.68 / 0.06 / 0.64.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Lowest accuracy (0.487) and MCC (0.235), but the best weighted precision (0.477) and the only model that predicts all three classes in roughly balanced numbers (147 Low / 132 Medium / 189 High) and gives the Medium class a meaningful F1 (0.26). A linear boundary cannot capture the non-linear weather and variety interactions, so it is the weakest overall, but it is the most "even-handed" across classes. |
| Decision Tree | Best accuracy (0.560) and best MCC (0.403). A shallow tree (depth 5) cleanly separates Low from High (recall for High = 0.96), but it almost never predicts Medium (8 of 468 predictions; Medium F1 ≈ 0.01). That is why its weighted precision (0.424) is low even though accuracy is high. Strong on the extremes, blind to the middle band. |
| KNN | Weakest ranking quality (AUC 0.655) and below-average accuracy (0.502). With 44 standardised and one-hot features, many of which are redundant copies of the same signal (see below), distances become less informative, so neighbours are not very meaningful. It predicts Medium only a little (Medium F1 0.16). |
| Naive Bayes | Middle of the pack on accuracy (0.521) but the highest F1 (0.480) and the second-best AUC (0.675). The independence assumption is clearly violated (inputs and weather columns are strongly correlated), yet its probability estimates still rank classes reasonably well. It is fast and does attempt the Medium class (74 predictions, F1 0.17). |
| Support Vector Machine | Best AUC (0.686) and second-best accuracy (0.543) and MCC (0.361), so the best-ranking and most robust non-tree model. The RBF kernel handles the non-linear structure better than Logistic Regression, but like the tree it largely collapses the Medium class (21 Medium predictions, F1 0.06). |
| **Overall winner for this dataset?** | **Decision Tree** – highest Accuracy (0.560), Recall (0.560) and MCC (0.403), the most reliable single summary for a 3-class problem. **SVM is a close runner-up** (best AUC, second-best Accuracy and MCC). Caveat: with 468 test rows the 95 % margin of error on accuracy is roughly ±4.5 points, so the gaps between the top three models are within noise; the clear conclusions are that Logistic Regression and KNN trail, and that *all* models struggle with the Medium class. |

### Why is the ceiling so low? (dataset insights)

- Eleven columns (`Hectares`, seed rate, main-field and nursery land-prep quantities, nursery area, DAP, weedicide, urea, potash, micronutrients, pesticide) are exact multiples of the plot size, so they carry **one** piece of information, not eleven.
- Rainfall, irrigation, temperature, wind and humidity take only a handful of distinct values (they are essentially per-agriblock / per-season constants), so they act as proxies for location rather than plot-level signal.
- The Medium class is a narrow middle band (≈ 5,855–6,181 Kg/ha), so small noise pushes plots into neighbouring classes. Models trade Medium accuracy for cleaner Low/High separation.
- Possible improvements: ordinal/regression modelling of yield per hectare, feature selection to remove redundant columns, ensemble methods, and adding plot-level agronomic variables.

## Streamlit App Features

- **CSV upload** of test data (or one click to use the bundled `test_data.csv`), with clear error messages for missing feature columns, a missing target column or invalid labels
- **Model selection dropdown** for all five classifiers
- **Evaluation metrics** (Accuracy, AUC, Precision, Recall, F1, MCC) for the selected model
- **Confusion matrix and classification report** for the selected model, plus a warning if the model never predicts a class
- **Compare-all-models tab** showing the same table and a bar chart for any metric, computed on the uploaded data
- **Row-level predictions** with class probabilities and a CSV download
- Models are loaded from `model/*.joblib`; **nothing is retrained** in the app
