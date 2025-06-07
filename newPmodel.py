import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Step 1: Load Data
df = pd.read_csv("Churn_Modelling.csv")
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

# Step 2: Feature Engineering
df['Tenure_to_Age'] = df['Tenure'] / (df['Age'])
df['Balance_to_Salary'] = df['Balance'] / (df['EstimatedSalary'] + 1)


# Step 3: Encode Categorical Variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

df['agewithnum']=df['Age']*df['NumOfProducts']
df['balancewithprod']=df['Balance']*df['NumOfProducts']
df['IsActiveMember_NuProd']=df['IsActiveMember'] * df['NumOfProducts']
df['HasCrCard_NuProd']=df['HasCrCard'] * df['NumOfProducts']
df['AgewithBalance']=df['Age']*df['Balance']


# Step 4: Split features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]
breakpoint()
# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 6: Apply SMOTE (slight oversampling)
smote = SMOTE(sampling_strategy=0.6, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Step 7: Try multiple scalers
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'MaxAbsScaler': MaxAbsScaler(),
    #'Normalizer': Normalizer()
}

# Step 8: Define Parameter Grids for GridSearch
param_grids = {
    # "Logistic Regression": {
    #     'C': [0.01, 0.1, 1, 10, 0.001, 0.0001],
    #     'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    #     'solver': ['lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag']
    # }
    # ,
    # "Random Forest": {
    #     'n_estimators': [200, 700, 800, 900, 1000, 1100],
    #     'max_depth': [20, 30, 40, 50, 60],
    #     'min_samples_split': [7, 10, 12, 15, 18]
    # },
    "XGBoost": {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 6, 10, 12, 15],
        'learning_rate': [0.01, 0.1, 0.001, 0.0001]
    }
}

# Step 9: Define Models
base_models = {
    # "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=3000),
    # "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
    # "XGBoost": XGBClassifier(random_state=42)
}

# Step 10: Evaluate models with each scaler
for scaler_name, scaler in scalers.items():
    print(f"\n================ Using {scaler_name} ================")
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)

    for model_name, model in base_models.items():
        print(f"\n--- GridSearchCV: {model_name} ---")
        grid = GridSearchCV(model, param_grids[model_name], cv=5, scoring='f1', n_jobs=-1, verbose=1)
        grid.fit(X_train_scaled, y_train_smote)
        best_model = grid.best_estimator_
        y_probs = best_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_probs >= 0.5).astype(int)

        print(f"Best Parameters: {grid.best_params_}")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}")
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists("scalers"):
            os.makedirs("scalers")
        # Save model and scaler
        model_filename = f"models/{model_name.replace(' ', '_')}_{scaler_name}.pkl" 
        scaler_filename = f"scalers/scaler_{scaler_name}.pkl"
        joblib.dump(best_model, model_filename)
        joblib.dump(scaler, scaler_filename)
        print(f"Saved model as {model_filename} and scaler as {scaler_filename}")



# FOr flask testing
'''# Step 11: Load model and scaler to predict on new data
print("\n✅ Example: Load a model and predict on new data")
example_data = X_test.iloc[[0]]  # simulate new input sample

# Choose the right scaler and model (change as needed)
scaler = joblib.load("scaler_StandardScaler.pkl")
model = joblib.load("Logistic_Regression_StandardScaler.pkl")

# Preprocess and predict
example_scaled = scaler.transform(example_data)
prediction = model.predict(example_scaled)
print("New sample prediction:", "Churn" if prediction[0] == 1 else "Stay")'''