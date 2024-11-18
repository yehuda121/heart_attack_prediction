import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle

# Load the dataset
data = pd.read_csv('C:\\Users\\yehud\\Desktop\\final_exam\\heart_attack_dataset.csv')

# Convert categorical data to numerical using Label Encoding
label_encoders = {}
for column in ['Sex', 'Country', 'Continent', 'Hemisphere', 'Diet']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the Blood Pressure column into Systolic and Diastolic
data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True)
data['Systolic'] = pd.to_numeric(data['Systolic'])
data['Diastolic'] = pd.to_numeric(data['Diastolic'])

# Drop the original Blood Pressure and Patient ID columns
data = data.drop(columns=['Blood Pressure', 'Patient ID'])

# Check if the 'Diet' column is entirely NaN
if data['Diet'].isna().all():
    data = data.drop(columns=['Diet'])
else:
    # Handle the 'Diet' column if it has missing values but is not entirely NaN
    data['Diet'].fillna(data['Diet'].mode()[0], inplace=True)

# Impute missing values (NaN) with the median of each column
numeric_columns = data.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(data[numeric_columns]), columns=numeric_columns)

# Shuffle the data
data_imputed = shuffle(data_imputed, random_state=42)

# Preprocessing: Split the dataset into features and target variable
X = data_imputed.drop(columns=['Heart Attack Risk'])
y = data_imputed['Heart Attack Risk']

# Preprocessing: Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate and print the mean and variance of the normalized data. the mean should be zero and the var should be one.
mean = np.mean(X_scaled, axis=0)
variance = np.var(X_scaled, axis=0)
print("Mean of the normalized data: ", mean)
print("\nVariance of the normalized data: ", variance)

# Base model: Decision Tree
base_model = DecisionTreeClassifier(random_state=42)

# 1) Cross-Validation with 80% of data
all_cv_scores = []
print("\n1) Cross-Validation with 80% of data:")
for i in range(5):
    # Shuffle the data
    X_shuffled, y_shuffled = shuffle(X_scaled, y, random_state=None)
    kf = KFold(n_splits=5, shuffle=True, random_state=42) # n_splits=5 => each split is 20% => 20%-80%
    cv_scores = cross_val_score(base_model, X_shuffled, y_shuffled, cv=kf)
    # print(f"Model {i+1} - Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    all_cv_scores.append(np.mean(cv_scores))
    
average_accuracy = np.mean(all_cv_scores)
print(f"Average Cross-Validation Accuracy across all models: {average_accuracy:.4f}")
    
# 2) Random 50% sampling
print("\n2) Random 50% sampling:")
random_sample_scores = []
for i in range(5):
    # Randomly split the data: 50% train, 50% test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=None)
    # Train the model on the training set
    model = DecisionTreeClassifier(random_state=i*42)
    model.fit(X_train, y_train)
    # Evaluate the model on the test set
    score = accuracy_score(y_test, model.predict(X_test))
    random_sample_scores.append(score)
    # print(f"Model {i+1} - Random Sampling Accuracy: {score:.4f}")

# Calculate and print the average accuracy across all models
average_random_sample_accuracy = np.mean(random_sample_scores)
print(f"Average Random Sampling Accuracy across all models: {average_random_sample_accuracy:.4f}")

# 3) AdaBoost with Decision Tree as base model
print("\n3) AdaBoost with Decision Tree as base model:")
all_ada_scores = []
for i in range(5):
    # Create AdaBoost model with Decision Tree as the base model
    ada_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1 , random_state=i*42), n_estimators=50, random_state=i*42)

    # Perform cross-validation
    ada_scores = cross_val_score(ada_model, X_scaled, y, cv=kf)
    # Print the result for each iteration
    # print(f"Model {i+1} - AdaBoost Accuracy: {np.mean(ada_scores):.4f} ± {np.std(ada_scores):.4f}")
    all_ada_scores.append(np.mean(ada_scores))
average_ada_accuracy = np.mean(all_ada_scores)
print(f"Average AdaBoost Accuracy across all models: {average_ada_accuracy:.4f}")

