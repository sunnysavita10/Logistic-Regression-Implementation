# Diabetes Data Analysis and Prediction

This repository contains a Python script for analyzing and predicting diabetes using the Pima Indians Diabetes Dataset. The script performs various data analysis and visualization tasks and applies a logistic regression model to predict diabetes outcomes.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis and Visualization](#analysis-and-visualization)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The dataset used in this project is the Pima Indians Diabetes Dataset, which is available in the `diabetes2.csv` file. The dataset includes the following features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (target variable, 1 indicates diabetes, 0 indicates no diabetes)

## Installation
To run this project, you need to have Python and the following libraries installed:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using the following command:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn

## Usage
1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/diabetes-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd diabetes-prediction
    ```
3. Run the script:
    ```bash
    python diabetes_analysis.py
    ```

## Analysis and Visualization
The script performs the following data analysis and visualization tasks:
- Displays dataset information and summary statistics.
- Checks for missing values.
- Generates box plots for each feature grouped by the outcome.
- Plots histograms for each feature grouped by the outcome.
- Calculates and visualizes the correlation matrix using a heatmap.

Example visualizations include:
- Pairplots
- Box plots for features such as Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.
- Histograms for features based on diabetes outcome.
- Violin plots to visualize the distribution of features.

## Model Training and Evaluation
The script trains a logistic regression model using the following steps:
1. Splits the data into training and testing sets.
2. Fits a logistic regression model to the training data.
3. Predicts outcomes on the test data.
4. Evaluates the model using confusion matrix and classification report.

## Results
The model's performance is evaluated using various metrics:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Mean Squared Error

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

