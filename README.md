# CSE 572 Data Mining Homework 2

## Dataset Overview
The Titanic dataset provides details of passengers aboard the RMS Titanic, including attributes such as `Age`, `Sex`, `Fare`, `Pclass` (passenger class), and `Embarked` (port of embarkation). The goal is to predict whether a passenger survived or not. The code is in the `Homework-2-Titanic-DM.ipynb` file.

## Preprocessing Steps

### 1. Handling Missing Values
Several columns in the Titanic dataset contain missing values. To address this:
   - **Age**: Filled missing values with the median value.
   - **Cabin**: Dropped due to a large number of missing entries.
   - **Embarked**: Filled missing values with the mode (most frequent value).

```python
# Fill missing values in 'Age' with the median
data['Age'] = data['Age'].fillna(data['Age'].median())

# Drop the 'Cabin' column due to too many missing values
data = data.drop(columns=['Cabin'])

# Fill missing values in 'Embarked' with the mode
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
```

### 2. Encoding Categorical Variables
To make categorical variables usable for machine learning models:
   - **Sex**: Encoded as binary values (0 for male, 1 for female).
   - **Embarked**: One-hot encoded to create separate columns for each port (`Embarked_Q`, `Embarked_S`), excluding the base category (`Embarked_C`).

```python
# Encode 'Sex' as binary values
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked' column
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
```

### 3. Feature Scaling
To ensure numerical consistency across features, we apply standard scaling:
   - Scaled numerical features using `StandardScaler` for attributes like `Age` and `Fare`.

```python
from sklearn.preprocessing import StandardScaler

# Define the features to be scaled
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = data[features]
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

### 4. Splitting Data
We split the data into training and validation sets for model evaluation.

```python
from sklearn.model_selection import train_test_split

# Define target variable
y = data['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Dependencies
This code requires the following libraries:
- `pandas`
- `scikit-learn`

To install these dependencies, use:
```bash
pip install pandas scikit-learn
```