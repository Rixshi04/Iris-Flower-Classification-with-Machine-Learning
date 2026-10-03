# Iris Flower Classification with Machine Learning

## Overview
This project classifies Iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — using a K-Nearest Neighbors (KNN) classifier with standardized features.

The project uses the built-in scikit-learn Iris dataset, so no external dataset download is required.

## Dataset
The Iris dataset contains 150 samples with four input features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

The target contains three species:
- Setosa
- Versicolor
- Virginica

## Project Workflow
1. Load the Iris dataset from scikit-learn.
2. Split the data into training and testing sets.
3. Fit a `StandardScaler` on the training data.
4. Transform the training and test features using the fitted scaler.
5. Train a KNN classifier with `n_neighbors=3`.
6. Evaluate the classifier using accuracy, classification report, and confusion matrix.
7. Save the fitted scaler and KNN model together in one reusable artifact.

## Installation
Install the required dependencies:

```bash
python -m pip install numpy pandas scikit-learn joblib
```

## How to Run
From the repository root:

```bash
python iris.py
```

The script prints the evaluation metrics and creates:

```text
iris_knn_pipeline.pkl
```

## Reusable Model Artifact
The saved `iris_knn_pipeline.pkl` contains:
- `scaler` — the fitted `StandardScaler`
- `model` — the fitted `KNeighborsClassifier`
- `target_names` — Iris class names
- `feature_names` — feature names used by the model

Saving the scaler together with the classifier is important because new input data must be transformed using the same scaling parameters learned from the training data.

Example:

```python
import joblib

artifact = joblib.load("iris_knn_pipeline.pkl")

scaler = artifact["scaler"]
model = artifact["model"]
target_names = artifact["target_names"]

new_flower = [[5.1, 3.5, 1.4, 0.2]]
new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)[0]
print("Predicted species:", target_names[prediction])
```

## Evaluation Metrics
The training script reports:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Model
The project uses **K-Nearest Neighbors (KNN)** with standardized input features.

## Future Improvements
- Hyperparameter tuning for KNN.
- Cross-validation.
- A small prediction API using Flask or FastAPI.
- A web interface for entering flower measurements.

## Contributors
- Rishi (@Rixshi04)

## License
This project is licensed under the MIT License.