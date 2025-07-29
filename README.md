# Employee Salary Prediction Using Ensemble Learning

## Project Overview

This project focuses on predicting employee salaries using multiple features such as gender, years of experience, job position, and location. By applying ensemble learning methods, the model leverages the strengths of multiple algorithms to provide reliable and accurate salary predictions.

Salary prediction is important for organizations to benchmark compensation and support HR decision-making. The project uses a dataset containing 400 employee records and applies machine learning techniques to build regression models that estimate salaries based on input features.

## Dataset Description

The dataset includes the following columns:

- **ID**: Unique identifier for each employee (removed before modeling).
- **Gender**: Employee gender (categorical).
- **Experience (Years)**: Total years of professional experience (numerical).
- **Position**: Job title or role (categorical).
- **Location**: City of employment (categorical).
- **Salary**: Annual salary in USD (target variable).

## Methodology

1. **Data Preprocessing**
   - Dropped the `ID` column as it does not contribute to salary prediction.
   - Applied one-hot encoding to categorical features (`Gender`, `Position`, `Location`) to transform them into numerical format.
   - Separated features and target variable (`Salary`).

2. **Data Splitting**
   - Divided the dataset into training (80%) and test (20%) subsets with a fixed random state for reproducibility.

3. **Model Training**
   - Trained three ensemble regression models:
     - **Random Forest Regressor:** Uses bagging of decision trees to reduce overfitting.
     - **Gradient Boosting Regressor:** Builds trees sequentially to minimize errors.
     - **Voting Regressor:** An ensemble that averages predictions from Random Forest and Gradient Boosting to improve overall performance.

4. **Model Evaluation**
   - Assessed models on test data using:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - R-squared (R²) score

5. **Prediction Example**
   - Performed a sample salary prediction for a hypothetical employee profile demonstrating model usability.

6. **Model Saving**
   - Persisted the final Voting Regressor and one-hot encoder using `joblib` for future deployment or inference.

## Results Summary

| Model               | MAE         | MSE           | R² Score   |
|---------------------|-------------|---------------|------------|
| Random Forest       | Moderate    | Moderate      | Good       |
| Gradient Boosting   | Slightly better than RF | Lower than RF | Best among models |
| Voting Regressor    | Comparable to Gradient Boosting | Competitive | Very Good  |

*Note:* Exact numeric values will vary depending on data splits and runs, but Gradient Boosting and Voting Regressor generally perform best.

## Dependencies / Tools

- Python 3.x
- pandas
- scikit-learn
- joblib

You can install required packages using:


## How to Run

1. Ensure `employee_data_with_locations.csv` is in your working directory.
2. Run the main Python script:


3. The script will:
   - Preprocess data
   - Train the ensemble models
   - Evaluate performances on test data
   - Produce a sample salary prediction output
   - Save the trained model and encoder files (`salary_predictor.pkl`, `feature_encoder.pkl`)

## Usage Example (Sample Prediction)

The model estimate can be queried by providing employee details like:


This returns a predicted salary score reflecting the learned relationships in the dataset.

## Future Work & Improvements

- Include additional relevant features such as education level or certifications.
- Explore hyperparameter tuning to optimize model accuracy.
- Implement advanced ensembling methods like stacking or blending.
- Build a user-friendly front-end to interact with the model predictions.

## Author

[Rachit Srivastava]

---

*This project demonstrates practical application of ensemble learning algorithms to a real-world regression problem, aligning with standard data science practices.*

