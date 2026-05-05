# Diabetes Prediction using KNN

This project uses the K-Nearest Neighbors (KNN) algorithm to predict whether a patient has diabetes based on medical diagnostic measurements such as Glucose levels, BMI, and Insulin.

## Project Structure
* **KNN_diabetes.ipynb**: Jupyter notebook containing the full data analysis, visualization, and training process.
* **KNN_diabetes.py**: Python script version of the model workflow.
* **diabetes.csv**: The Pima Indians Diabetes dataset.
* **diabetes_knn_model.pkl**: The trained K-Nearest Neighbors model.
* **diabetes_scaler.pkl**: The StandardScaler object used to normalize the data.
* **requirements.txt**: List of Python libraries required to run the project.

## Dataset Features
The model uses 8 diagnostic predictors:
1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age
* **Target:** Outcome (0 for non-diabetic, 1 for diabetic).

## Workflow

### 1. Data Processing
* Loaded the dataset and performed exploratory data analysis (EDA).
* Split the data into training (80%) and testing (20%) sets.
* Scaled the features using `StandardScaler` to ensure the KNN distance calculation is accurate across different units.

### 2. Model Training
* Implemented the `KNeighborsClassifier` with `n_neighbors=5`.
* Trained the model on the scaled training dataset.

### 3. Evaluation
* **Accuracy:** The model achieved an accuracy of approximately 69.5% on the test set.
* **Visualization:** A confusion matrix heatmap was generated using Seaborn to visualize true positives vs. false positives.

## Usage
1. Install requirements: `pip install -r requirements.txt`.
2. To make predictions, load the `diabetes_scaler.pkl` to transform your input data before passing it to the `diabetes_knn_model.pkl`.
