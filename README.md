# Titanic - Machine Learning from Disaster

This project is part of the CSE 572 Data Mining Course Homework 1, where we apply various machine learning models to predict the survival of passengers on the Titanic based on their characteristics.

This project uses the famous [Titanic dataset](https://www.kaggle.com/c/titanic/data) to predict the survival of passengers. The dataset includes features such as age, gender, class, and more. We preprocess the data and apply multiple machine learning models to predict the survival rate of passengers.

## Models and Accuracy Scores

After applying the preprocessing steps and training multiple machine learning models, the following accuracy scores were achieved:

| Model                      | Score  |
|-----------------------------|--------|
| Random Forest               | 86.76% |
| Decision Tree               | 86.76% |
| K-Nearest Neighbors (KNN)   | 83.84% |
| Logistic Regression         | 80.36% |
| Linear SVC                  | 78.90% |
| Perceptron                  | 78.34% |
| Support Vector Machines     | 78.23% |
| Stochastic Gradient Descent | 75.76% |
| Naive Bayes                 | 72.28% |

## Data Preprocessing

The following data preprocessing steps were applied:

1. **Data Cleaning**: 
   - Missing values in `Age` were filled with median values based on `Sex` and `Pclass`.
   - Missing values in `Embarked` and `Fare` were filled with the most frequent and median values, respectively.
   
2. **Feature Engineering**:
   - Created a `Title` feature from the `Name` column, which represents social status.
   - Engineered `FamilySize` and `IsAlone` features from the `SibSp` and `Parch` columns.
   - Combined `Age` and `Pclass` to create the `Age*Class` feature.

3. **Discretization**:
   - Discretized `Age` and `Fare` into bins for better model performance.

4. **Encoding**:
   - Categorical features like `Sex`, `Embarked`, and `Title` were converted into numerical values for machine learning models.

5. **Dropping Irrelevant Features**:
   - Columns like `Ticket`, `Cabin`, `PassengerId`, and `Name` were dropped as they did not contribute significantly to the model's performance.

## Machine Learning Models

The following machine learning models were trained on the processed dataset:

- **Logistic Regression**
- **Support Vector Machines (SVC and LinearSVC)**
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**
- **Decision Tree Classifier**
- **Naive Bayes**
- **Perceptron**
- **Stochastic Gradient Descent (SGD)**

## Results

The Random Forest and Decision Tree classifiers achieved the highest accuracy of 86.76%. K-Nearest Neighbors also performed well, with an accuracy of 83.84%. 

## References

- [Kaggle Titanic Data](https://www.kaggle.com/c/titanic/data)
- [Kaggle Titanic Data Science Solutions](https://www.kaggle.com/code/preejababu/titanic-data-science-solutions)
- [Kaggle Titanic Journey](https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic)
- [Random Forest Starter](https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests)
- [Best Working Classifier](https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier)