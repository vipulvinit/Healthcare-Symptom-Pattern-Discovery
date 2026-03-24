# Healthcare Symptom Pattern Discovery
## Unsupervised Machine Learning Pipeline

---

## Overview

This repository contains five Jupyter notebooks that implement an end-to-end unsupervised machine learning pipeline for **discovering latent symptom patterns** in healthcare records, using the [Disease Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) from Kaggle.

The pipeline identifies clinically meaningful symptom clusters without using disease labels — enabling early risk stratification in settings where confirmed diagnoses are unavailable.

---

## Repository Structure

```
├── dataset.csv                                    ← Raw input dataset (Kaggle)
│
├── Notebook_1_Data_Understanding_Preprocessing.ipynb
├── Notebook_2_Feature_Engineering_Modelling.ipynb
├── Notebook_3_Visualization_Analysis.ipynb
├── Notebook_4_Medical_Interpretation.ipynb
├── Notebook_5_Ethics_Limitations.ipynb
│
├── Healthcare_Symptom_Pattern_Discovery.pptx     ← Presentation slides
└── README.md
```

---

## Execution Order

> ⚠️ **The notebooks must be run in order (1 → 2 → 3 → 4 → 5)**. Each notebook produces a CSV that the next notebook reads as input.

| Step | Notebook | Input | Output CSV |
|------|----------|-------|------------|
| 1 | Data Understanding & Preprocessing | `dataset.csv` | `task1_encoded.csv` |
| 2 | Feature Engineering & Modelling | `task1_encoded.csv` | `task2_clustered.csv` |
| 3 | Visualization & Analysis | `task2_clustered.csv` | `task3_viz_data.csv` |
| 4 | Medical Interpretation | `task3_viz_data.csv` | `task4_interpreted.csv` + `task4_clinical_summary.csv` |
| 5 | Ethics, Limitations & Final Report | `task4_interpreted.csv` | `task5_final_report.csv` |

---

## Notebook Descriptions

### Notebook 1 — Data Understanding and Preprocessing
**Task:** Encode symptom presence effectively

- Loads the raw dataset (4,920 patient records, 17 symptom columns, 41 diseases)
- Performs exploratory data analysis: disease distribution, missing values, symptom frequency
- Melts wide format → long format and standardises symptom strings
- Builds a **binary patient × symptom matrix** (4,920 × 130)
- Generates symptom co-occurrence heatmap
- **Outputs:** `task1_encoded.csv`

### Notebook 2 — Feature Engineering and Unsupervised Modelling
**Task:** Cluster patient symptom profiles

- Loads binary matrix from Notebook 1
- Engineers 6 clinical aggregate features: `symptom_count`, `fever_flag`, `gi_score`, `resp_score`, `skin_score`, `pain_score`
- Applies `StandardScaler` + `PCA` (retaining ≥ 80% variance)
- Computes Ward linkage matrix and plots truncated dendrogram
- Evaluates Silhouette Score and Davies-Bouldin Index for k=2..12
- Fits **Hierarchical Clustering (Ward)** and **Gaussian Mixture Model (GMM)** at optimal k
- Reports Adjusted Rand Index against disease labels (post-hoc only)
- **Outputs:** `task2_clustered.csv`

### Notebook 3 — Visualization and Analysis
**Task:** Visualise symptom clusters

- Loads clustered data from Notebook 2
- PCA 2D scatter plots (HC vs GMM side-by-side)
- t-SNE non-linear projection for both clustering methods
- Cluster-symptom heatmap (top 35 discriminative symptoms)
- Per-sample Silhouette plot with cluster colouring
- Disease composition stacked bar chart (post-hoc validation)
- GMM assignment confidence distribution violin/histogram
- **Outputs:** `task3_viz_data.csv`

### Notebook 4 — Medical Interpretation
**Task:** Interpret potential health risk groupings

- Loads visualisation data from Notebook 3
- Computes per-cluster mean symptom presence rates
- Extracts top-10 defining symptoms per cluster
- Identifies dominant disease and cluster purity
- Applies **rule-based heuristic** to assign clinical risk categories:
  - Hepatic / Jaundice Risk
  - Respiratory Risk
  - Gastrointestinal Risk
  - Dermatological Risk
  - Neurological Risk
  - Metabolic / Endocrine Risk
  - Musculoskeletal Risk
  - Febrile / Infectious Risk
- Produces **radar/spider charts** across 8 clinical dimensions
- Builds a full **Clinical Summary Table**
- **Outputs:** `task4_interpreted.csv`, `task4_clinical_summary.csv`

### Notebook 5 — Ethics, Limitations and Final Report
**Task:** Discuss limitations of unsupervised health analysis

- Loads interpreted data from Notebook 4
- Full pipeline summary dashboard (5-panel figure)
- Structured documentation of **7 ethical considerations**
- Structured documentation of **9 technical & clinical limitations**
- **5 societal benefits** identified
- 10 recommended future work directions
- Consolidated final report CSV
- **Outputs:** `task5_final_report.csv`

---

## Software Environment

### Python Version
```
Python 3.10.x or higher recommended
```

### Required Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥ 1.5 | Data manipulation |
| `numpy` | ≥ 1.23 | Numerical operations |
| `matplotlib` | ≥ 3.6 | Plotting |
| `seaborn` | ≥ 0.12 | Statistical visualisation |
| `scikit-learn` | ≥ 1.2 | PCA, clustering, metrics |
| `scipy` | ≥ 1.9 | Hierarchical clustering, linkage |

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### Running the Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab
```

Then open and run each notebook in order (Kernel → Restart & Run All).

---

## Dataset

**Source:** [Disease Symptom Description Dataset — Kaggle](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)

**Citation:**
> itachi9604. "Disease Symptom Description Dataset." Kaggle, 2020.

**Key statistics:**
- 4,920 patient records
- 41 disease classes (120 records each)
- 17 symptom slot columns (Symptom_1 to Symptom_17)
- 130 unique symptoms after preprocessing

The dataset is placed as `dataset.csv` in the root of the repository.

---

## Models Used

| Model | Library | Configuration |
|-------|---------|---------------|
| Hierarchical Clustering | `scipy.cluster.hierarchy` | Ward linkage, Euclidean distance |
| Gaussian Mixture Model | `sklearn.mixture.GaussianMixture` | Full covariance, 5 initialisations |
| PCA | `sklearn.decomposition.PCA` | 80% variance threshold |
| t-SNE | `sklearn.manifold.TSNE` | perplexity=40, max_iter=1000 |

---

## Figures Generated

Each notebook saves figures automatically to the working directory:

| Notebook | Figures |
|----------|---------|
| 1 | `task1_disease_distribution.png`, `task1_symptom_frequency.png`, `task1_cooccurrence_heatmap.png` |
| 2 | `task2_pca_variance.png`, `task2_dendrogram.png`, `task2_cluster_metrics.png` |
| 3 | `task3_pca_scatter.png`, `task3_tsne_scatter.png`, `task3_cluster_symptom_heatmap.png`, `task3_silhouette_plot.png`, `task3_cluster_composition.png`, `task3_gmm_confidence.png` |
| 4 | `task4_risk_bubble_chart.png`, `task4_radar_charts.png` |
| 5 | `task5_pipeline_summary.png`, `task5_ethics_limitations.png` |

---

## Assignment Mapping

| Task | Notebook | Marks |
|------|----------|-------|
| Encode symptom presence effectively | Notebook 1 | Data Understanding (5) |
| Cluster patient symptom profiles | Notebook 2 | Unsupervised Analysis (5) |
| Visualise symptom clusters | Notebook 3 | Unsupervised Analysis (5) |
| Interpret potential health risk groupings | Notebook 4 | Medical Interpretation (5) |
| Discuss limitations of unsupervised health analysis | Notebook 5 | Reporting (15) |

---

## Notes

- All notebooks are **fully commented** — every statement includes an explanation of its purpose and contribution.
- The `Disease` column is **never used as input to any model**; it is retained only for post-hoc evaluation using Adjusted Rand Index.
- Random seeds (`np.random.seed(42)`) are set throughout for reproducibility.
- All figures are saved as `.png` files with `bbox_inches='tight'` for clean export.
