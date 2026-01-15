# Medical Disease Prediction using K-Nearest Neighbors (KNN)

A machine learning project that implements a diagnostic engine using the K-Nearest Neighbors (KNN) algorithm. The system analyzes patterns of symptoms to classify a patient's condition into one of many specific disease categories.

## Overview

The K-Nearest Neighbors (KNN) algorithm is a non-parametric, instance-based learning method used for classification. In this medical context, the model predicts a disease label for a new patient by identifying the "K" most similar cases in the historical dataset based on the presence or absence of specific symptoms.

## Dataset

- **Source:** Multi-symptom disease dataset (`30_symptoms_dataset.csv`).
- **Structure:** The dataset contains 18 columns, including a `Disease` label and up to 17 symptoms per entry.
- **Disease Categories:** Includes conditions like Fungal infection, Allergy, GERD, Chronic cholestasis, Malaria, and many others.
- **Symptoms:** Features ranging from `itching` and `skin_rash` to `vomiting`, `fatigue`, and `joint_pain`.

## Objectives

- Understand the logic of **Instance-based Learning** and the **KNN algorithm**.
- Perform significant **Data Restructuring**: Transforming wide-format symptom data into a sparse matrix for modeling.
- Implement **Distance Metrics**: Using Euclidean distance to determine "closeness" between patient profiles.
- Select an optimal **K-value** to balance model sensitivity and bias.
- Predict disease labels for new symptom profiles and validate against ground truth data.

## Methods and Analysis

The project follows a comprehensive medical classification workflow:

- **The KNN Logic**
  1. Pick a value for K (number of neighbors).
  2. For a new "query point" (patient symptoms), calculate the distance to all existing points.
  3. Find the K closest neighbors.
  4. Assign the class label that appears most frequently among those K neighbors.



- **Data Preprocessing & Encoding**
  - Handling the high dimensionality of symptoms.
  - Converting categorical symptom strings into numerical values using **Label Encoding** or **One-Hot Encoding** logic to ensure mathematical distance can be calculated.

- **Exploratory Data Analysis (EDA)**
  - Identifying the most common symptoms associated with specific disease clusters.
  - Analyzing the distribution of diseases in the dataset to ensure balanced learning.

- **Prediction & Evaluation**
  - Splitting the data into training and test sets.
  - Using the model to predict outcomes for "New Records" and mapping numerical labels back to human-readable disease names (e.g., Label [8] -> Hypothyroidism).



## Tech Stack

- **Language:** Python 3
- **Libraries:**
  - `pandas` and `numpy`: For handling the multi-column symptom matrix.
  - `matplotlib` and `seaborn`: For visualizing symptom correlations.
  - `scikit-learn`: For KNN implementation (`KNeighborsClassifier`) and data scaling.
- **Environment:** Jupyter / Google Colab

## How to Run

1. **Clone this repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/medical-knn-prediction.git
   cd medical-knn-prediction

2. *Create and activate a virtual environment (optional but recommended):*
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. *Install dependencies:*
   pip install pandas numpy seaborn matplotlib scikit-learn

4. Ensure the dataset is present: Place 30_symptoms_dataset.csv in the root folder.

5. *Open the notebook:*
   jupyter notebook 30_KNN_medical_disease_prediction_good.ipynb
