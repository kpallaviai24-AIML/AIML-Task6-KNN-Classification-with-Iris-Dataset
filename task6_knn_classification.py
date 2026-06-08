# ==========================================================
# TASK 6 : K-NEAREST NEIGHBORS (KNN) CLASSIFICATION
# ==========================================================

import os
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# CREATE FOLDERS
# ==========================================================

os.makedirs("Output_Visualizations", exist_ok=True)
os.makedirs("Model_Output", exist_ok=True)
os.makedirs("Predictions", exist_ok=True)
os.makedirs("Saved_Model", exist_ok=True)

print("=" * 60)
print("TASK 6 : KNN CLASSIFICATION")
print("=" * 60)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("dataset/Iris.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst Five Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================================
# REMOVE ID COLUMN
# ==========================================================

if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)

# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

plt.figure(figsize=(6,4))

sns.countplot(
    x="Species",
    data=df
)

plt.title("Target Distribution")

plt.savefig(
    "Output_Visualizations/target_distribution.png"
)

plt.close()

# ==========================================================
# FEATURE DISTRIBUTION
# ==========================================================

df.hist(
    figsize=(10,8)
)

plt.tight_layout()

plt.savefig(
    "Output_Visualizations/feature_distribution.png"
)

plt.close()

# ==========================================================
# PAIRPLOT
# ==========================================================

sns.pairplot(
    df,
    hue="Species"
)

plt.savefig(
    "Output_Visualizations/pairplot.png"
)

plt.close()

# ==========================================================
# ENCODE TARGET
# ==========================================================

encoder = LabelEncoder()

df["Species"] = encoder.fit_transform(
    df["Species"]
)

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.savefig(
    "Output_Visualizations/correlation_heatmap.png"
)

plt.close()

# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df.drop(
    "Species",
    axis=1
)

y = df["Species"]

# ==========================================================
# NORMALIZATION
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

# ==========================================================
# FIND BEST K
# ==========================================================

k_values = range(1,21)

accuracy_scores = []

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    accuracy_scores.append(
        accuracy_score(
            y_test,
            pred
        )
    )

best_k = k_values[
    accuracy_scores.index(
        max(accuracy_scores)
    )
]

print(
    f"\nBest K Value: {best_k}"
)

# ==========================================================
# K ACCURACY CURVE
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    k_values,
    accuracy_scores,
    marker="o"
)

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K Accuracy Curve")

plt.savefig(
    "Output_Visualizations/k_accuracy_curve.png"
)

plt.close()

# ==========================================================
# FINAL KNN MODEL
# ==========================================================

knn = KNeighborsClassifier(
    n_neighbors=best_k
)

knn.fit(
    X_train,
    y_train
)

print(
    "\nKNN Model Training Completed"
)
# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred = knn.predict(X_test)

# ==========================================================
# EVALUATION METRICS
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nAccuracy Score : {accuracy:.4f}"
)

with open(
    "Model_Output/evaluation_metrics.txt",
    "w"
) as f:

    f.write(
        "KNN CLASSIFICATION METRICS\n\n"
    )

    f.write(
        f"Best K Value : {best_k}\n"
    )

    f.write(
        f"Accuracy Score : {accuracy:.4f}\n"
    )

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.savefig(
    "Output_Visualizations/confusion_matrix.png"
)

plt.close()

# ==========================================================
# SAVE CONFUSION MATRIX VALUES
# ==========================================================

with open(
    "Model_Output/confusion_matrix_values.txt",
    "w"
) as f:

    f.write(
        str(cm)
    )

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
)

with open(
    "Model_Output/classification_report.txt",
    "w"
) as f:

    f.write(report)

print("\nClassification Report")
print(report)

# ==========================================================
# SAVE BEST K VALUE
# ==========================================================

with open(
    "Model_Output/best_k_value.txt",
    "w"
) as f:

    f.write(
        f"Best K Value = {best_k}\n"
    )

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

prediction_df = pd.DataFrame(
    {
        "Actual": encoder.inverse_transform(
            y_test
        ),
        "Predicted": encoder.inverse_transform(
            y_pred
        )
    }
)

prediction_df.to_csv(
    "Predictions/knn_predictions.csv",
    index=False
)

# ==========================================================
# CROSS VALIDATION
# ==========================================================

cv_scores = cross_val_score(
    knn,
    X_scaled,
    y,
    cv=5
)

with open(
    "Model_Output/cross_validation_results.txt",
    "w"
) as f:

    f.write(
        "Cross Validation Scores\n\n"
    )

    f.write(
        str(cv_scores)
    )

    f.write(
        f"\n\nMean Accuracy : {cv_scores.mean():.4f}"
    )

print(
    f"\nCross Validation Mean Accuracy : {cv_scores.mean():.4f}"
)

# ==========================================================
# DISTANCE METRIC COMPARISON
# ==========================================================

metrics = [
    "euclidean",
    "manhattan",
    "minkowski"
]

metric_scores = []

for metric in metrics:

    temp_model = KNeighborsClassifier(
        n_neighbors=best_k,
        metric=metric
    )

    temp_model.fit(
        X_train,
        y_train
    )

    temp_pred = temp_model.predict(
        X_test
    )

    metric_scores.append(
        accuracy_score(
            y_test,
            temp_pred
        )
    )

plt.figure(figsize=(8,5))

sns.barplot(
    x=metrics,
    y=metric_scores
)

plt.title(
    "Distance Metric Comparison"
)

plt.ylabel(
    "Accuracy"
)

plt.savefig(
    "Output_Visualizations/distance_metric_comparison.png"
)

plt.close()

with open(
    "Model_Output/distance_metric_results.txt",
    "w"
) as f:

    for metric, score in zip(
        metrics,
        metric_scores
    ):

        f.write(
            f"{metric} : {score:.4f}\n"
        )

# ==========================================================
# SAVE MODEL
# ==========================================================

pickle.dump(
    knn,
    open(
        "Saved_Model/knn_model.pkl",
        "wb"
    )
)

print(
    "\nModel Saved Successfully"
)
# ==========================================================
# LEARNING CURVE
# ==========================================================

train_sizes, train_scores, test_scores = learning_curve(
    knn,
    X_scaled,
    y,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(
    train_scores,
    axis=1
)

test_mean = np.mean(
    test_scores,
    axis=1
)

plt.figure(figsize=(8,5))

plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    train_sizes,
    test_mean,
    marker="o",
    label="Validation Accuracy"
)

plt.title("Learning Curve")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig(
    "Output_Visualizations/learning_curve.png"
)

plt.close()

# ==========================================================
# NORMALIZED VS ORIGINAL COMPARISON
# ==========================================================

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.boxplot(
    data=X
)

plt.title(
    "Original Features"
)

plt.xticks(rotation=90)

plt.subplot(1,2,2)

sns.boxplot(
    data=pd.DataFrame(
        X_scaled,
        columns=X.columns
    )
)

plt.title(
    "Normalized Features"
)

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    "Output_Visualizations/normalized_vs_original.png"
)

plt.close()

# ==========================================================
# CLASS PREDICTION SCATTER
# ==========================================================

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X_test[:,0],
    X_test[:,1],
    c=y_pred,
    cmap="viridis"
)

plt.title(
    "Class Prediction Scatter"
)

plt.xlabel(
    "Feature 1"
)

plt.ylabel(
    "Feature 2"
)

plt.colorbar(scatter)

plt.savefig(
    "Output_Visualizations/class_prediction_scatter.png"
)

plt.close()

# ==========================================================
# DECISION BOUNDARY
# ==========================================================

X_boundary = X_scaled[:, :2]

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_boundary,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

boundary_model = KNeighborsClassifier(
    n_neighbors=best_k
)

boundary_model.fit(
    X_train_b,
    y_train_b
)

x_min, x_max = X_boundary[:,0].min() - 1, X_boundary[:,0].max() + 1
y_min, y_max = X_boundary[:,1].min() - 1, X_boundary[:,1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = boundary_model.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.4
)

plt.scatter(
    X_boundary[:,0],
    X_boundary[:,1],
    c=y,
    edgecolors="k"
)

plt.title(
    "Decision Boundary"
)

plt.savefig(
    "Output_Visualizations/decision_boundary.png"
)

plt.close()

# ==========================================================
# MODEL SUMMARY
# ==========================================================

with open(
    "Model_Output/model_summary.txt",
    "w"
) as f:

    f.write(
        "KNN CLASSIFICATION MODEL SUMMARY\n\n"
    )

    f.write(
        f"Dataset Size : {df.shape}\n"
    )

    f.write(
        f"Number of Features : {X.shape[1]}\n"
    )

    f.write(
        f"Training Samples : {len(X_train)}\n"
    )

    f.write(
        f"Testing Samples : {len(X_test)}\n"
    )

    f.write(
        f"Best K Value : {best_k}\n"
    )

    f.write(
        f"Accuracy : {accuracy:.4f}\n"
    )

    f.write(
        f"Cross Validation Accuracy : {cv_scores.mean():.4f}\n"
    )

# ==========================================================
# FINAL CONSOLE OUTPUT
# ==========================================================

print("\n" + "=" * 60)
print("TASK 6 COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Best K Value : {best_k}")
print(f"Accuracy : {accuracy:.4f}")
print(f"Cross Validation Accuracy : {cv_scores.mean():.4f}")

print("\nGenerated Folders:")
print("Output_Visualizations/")
print("Model_Output/")
print("Predictions/")
print("Saved_Model/")

print("\n✅ All Reports, Graphs, Predictions and Model Files Saved Successfully.")
print("=" * 60)