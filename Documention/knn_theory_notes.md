# KNN Theory Notes

## What is KNN?

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression tasks. It is an instance-based learning method that classifies a data point based on the majority class of its nearest neighbors.

## Working Principle

1. Select the value of K.
2. Calculate the distance between the test point and all training points.
3. Identify the K nearest neighbors.
4. Count the frequency of each class among neighbors.
5. Assign the class with the highest frequency.

## Distance Metrics

### Euclidean Distance

Measures the straight-line distance between two points.

### Manhattan Distance

Measures distance along horizontal and vertical paths.

### Minkowski Distance

Generalized distance metric that includes Euclidean and Manhattan distances.

## Importance of Normalization

KNN relies on distance calculations. Features with larger scales can dominate distance computations. Therefore, normalization is essential.

## Advantages

* Simple to understand and implement.
* No training phase required.
* Effective for small datasets.
* Supports multi-class classification.

## Disadvantages

* Computationally expensive for large datasets.
* Sensitive to noise and outliers.
* Performance depends on the value of K.
* Requires feature scaling.

## Applications

* Medical Diagnosis
* Recommendation Systems
* Pattern Recognition
* Image Classification
* Financial Analysis
