Of course! That's a great addition to make the `README` even more informative for anyone looking at your project. I'll add a "Tech Stack" section.

Here is the updated `README.md` with the new section integrated.

---

# Predicting Next Purchase Day: A Machine Learning Approach

This repository contains the full codebase for the term paper titled *"Development and Research of Models for Predicting the Next Purchase Based on Transaction History"*.

The project aims to solve a common household problem: running out of everyday essentials (Fast-Moving Consumer Goods or FMCG). By analyzing a user's purchase history, the model predicts the number of days until the next purchase of a specific product category, forming the core of a "smart shopping list" application.

## Key Features & Methodology

The project follows a comprehensive data science pipeline, from raw data processing to model evaluation:

*   **Semantic Category Clustering**: Raw product sub-categories were clustered into unified macro-categories using LLMs to handle product interchangeability (e.g., grouping all types of milk).
*   **Physical Unit Normalization**: A sophisticated parser was developed to convert unstructured text descriptions of product sizes (e.g., "16 OZ", "1/2 GAL", "6 PK") into three standardized physical vectors: **Weight (grams)**, **Volume (milliliters)**, and **Count (units)**.
*   **Sessionization for Churn Detection**: User histories were algorithmically split into distinct "sessions" of continuous consumption, allowing the model to ignore anomalous gaps in data caused by long breaks or user churn.
*   **Advanced Feature Engineering**: A rich feature set of **52 predictors** was engineered for each transaction, including lag features, normalized lags, rolling statistics (mean, std, EMA), and heuristic-based projections.
*   **Modeling**: Implementation and comparison of multiple baseline models against a powerful Gradient Boosting model (CatBoost).
*   **Stratified Evaluation**: Model performance is rigorously evaluated using MAE (Mean Absolute Error) and MASE (Mean Absolute Scaled Error) metrics, calculated separately for **"Cold Start"** and **"Warm State"** scenarios.

## Tech Stack

*   **Language**: Python 3.13.7
*   **Core Data Science Libraries**: Pandas, NumPy, Scikit-learn
*   **Gradient Boosting**: CatBoost
*   **Development Environment**: Jupyter Notebooks, VS Code

## Repository Structure

The project is organized into a modular structure to separate data processing, feature engineering, and modeling logic.

```
.
├── data/
│   └── Dunnhumby/          # Raw, intermediate, and final datasets
├── files/                  # Helper files, e.g., JSON for category mapping
├── models/
│   ├── classes/            # Python classes for baselines and model wrappers
│   └── fitted/             # Saved (trained) model artifacts (.joblib files)
├── notebooks/
│   ├── data-pipeline.ipynb   # 1. Initial data cleaning, normalization, and sessionization
│   ├── dataset-pipeline.ipynb# 2. Feature engineering and train/val/test split
│   └── ml-pipeline.ipynb     # 3. Model training, evaluation, and experiments
├── src/                    # Source code with modular functions for the pipeline
│   ├── dataset_constructor.py
│   ├── download.py
│   ├── features_extraction.py
│   ├── metrics_calculation.py
│   └── ...
├── .gitignore
└── requirements.txt
```

## How to Reproduce the Results

The project is designed to be fully reproducible. The dataset is downloaded automatically.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/your-repo-name.git
    cd your-repo-name
    ```

2.  **Set up the environment:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Run the data preparation pipelines:**
    Execute the notebooks sequentially. They will handle everything from downloading the raw data to generating the final feature-engineered datasets.
    *   **Run `notebooks/data-pipeline.ipynb`:** This notebook will download the Dunnhumby dataset, perform initial cleaning, normalize physical units, and create consumption sessions, saving the result as `outflow_data.csv`.
    *   **Run `notebooks/dataset-pipeline.ipynb`:** This notebook will take the processed data, apply the feature engineering logic from `src/`, and create the final `dataset_train.csv`, `dataset_val.csv`, and `dataset_test.csv`.

4.  **Train models and experiment:**
    *   Open **`notebooks/ml-pipeline.ipynb`** to train the baseline and CatBoost models, evaluate their performance, and conduct new experiments.
