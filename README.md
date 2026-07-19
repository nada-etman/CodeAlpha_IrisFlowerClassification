# 🌸 Iris Flower Classification

A machine learning project that classifies Iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — based on sepal and petal measurements. 
---

##  Overview

The Iris dataset is one of the most well-known datasets in machine learning, containing 150 samples of iris flowers with four numeric features per sample. This project walks through a complete classification pipeline: exploratory data analysis, preprocessing, training multiple models, comparing their performance, and visualizing the results.

##  Dataset

| Feature | Description |
|---|---|
| `SepalLengthCm` | Sepal length in centimeters |
| `SepalWidthCm` | Sepal width in centimeters |
| `PetalLengthCm` | Petal length in centimeters |
| `PetalWidthCm` | Petal width in centimeters |
| `Species` | Target class (Setosa / Versicolor / Virginica) |

- 150 total samples, 50 per class (balanced dataset)

##  Tech Stack

- **Python 3**
- **Pandas** – data loading & manipulation
- **Matplotlib / Seaborn** – data visualization
- **Scikit-learn** – preprocessing, model training & evaluation

##  Project Workflow

1. **Data Loading & Cleaning** — Loaded the dataset and dropped the non-informative `Id` column.
2. **Exploratory Data Analysis (EDA)** — Generated a pairplot and a correlation heatmap to understand feature relationships and class separability.
3. **Preprocessing** — Encoded species labels with `LabelEncoder` and split the data into training (80%) and testing (20%) sets using stratified sampling.
4. **Model Training** — Trained and compared three classification algorithms:
   - Logistic Regression
   - Support Vector Machine (SVM, linear kernel)
   - Random Forest Classifier
5. **Evaluation** — Compared model accuracy and generated a classification report (precision, recall, F1-score) for each model.
6. **Best Model Selection** — Automatically selected the best-performing model and visualized its results with a confusion matrix.
7. **Feature Importance** — Used the Random Forest model to rank feature importance.

##  Results

### Model Comparison

| Model | Accuracy |
|---|---|
| Logistic Regression | ~96–100% |
| **SVM (Linear Kernel)** | **100%** |
| Random Forest | ~96–100% |

> The SVM model achieved perfect classification on the test set, correctly identifying all 30 test samples (10 per species) with zero misclassifications.

### Confusion Matrix (Best Model — SVM)

The confusion matrix below shows the SVM model made **zero misclassifications**, correctly identifying all Setosa, Versicolor, and Virginica samples in the test set.

![Confusion Matrix](iris_confusion_matrix.png)

### Feature Correlation Heatmap

Petal length and petal width are strongly correlated (0.96), and both correlate strongly with sepal length, making petal measurements the most informative features for classification.

![Correlation Heatmap](iris_correlation_heatmap.png)

### Feature Importance (Random Forest)

`PetalWidthCm` and `PetalLengthCm` are by far the most important features for distinguishing between species, while sepal measurements contribute far less.

![Feature Importance](iris_feature_importance.png)

### Pairplot — Feature Distribution by Species

The pairplot clearly shows that Setosa is linearly separable from the other two species, while Versicolor and Virginica show some overlap in sepal measurements but separate well on petal measurements.

![Pairplot](iris_pairplot.png)

##  How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Iris_Flower_Classification.git
   cd Iris_Flower_Classification
   ```

2. Install the required dependencies:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```

3. Run the script:
   ```bash
   python iris_classification.py
   ```

The script will print dataset info, model accuracy scores, and classification reports to the console, and save all visualizations (pairplot, heatmap, confusion matrix, feature importance) as `.png` files in the project directory.

##  Project Structure

```
CodeAlpha_Iris_Flower_Classification/
│
├── Iris.csv                        # Dataset
├── iris_classification.py          # Main script
├── iris_pairplot.png               # EDA visualization
├── iris_correlation_heatmap.png    # Feature correlation heatmap
├── iris_confusion_matrix.png       # Confusion matrix of the best model
├── iris_feature_importance.png     # Random Forest feature importance
└── README.md                       # Project documentation
```

##  Key Takeaways

- Petal measurements (length & width) are significantly more predictive of species than sepal measurements.
- Setosa is easily separable from the other two species using a single feature.
- A simple linear SVM is sufficient to achieve perfect classification on this dataset.


