from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


MODEL_PATH = Path(__file__).resolve().parent / "iris_knn_pipeline.pkl"


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save preprocessing and model together so the artifact can be reused
    # with raw Iris measurements.
    artifact = {
        "scaler": scaler,
        "model": knn,
        "target_names": iris.target_names,
        "feature_names": iris.feature_names,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved reusable model artifact to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
