import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df):
    # Assuming 'Churn' is the target variable
    X = df.drop('churn', axis=1)
    y = df['churn']

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Create preprocessing steps for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit the preprocessor and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after preprocessing
    onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = onehot_encoder.get_feature_names(categorical_features).tolist()
    feature_names = numeric_features.tolist() + cat_feature_names

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor, feature_names

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")


def xgboost_model(X_train, X_test, y_train, y_test, feature_names):
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    print("\nXGBoost Model:")
    evaluate_model(model, X_test, y_test)
    # plot_feature_importance(model, feature_names)

    
    return model

def random_forest_model(X_train, X_test, y_train, y_test, feature_names):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("\nRandom Forest Model:")
    evaluate_model(model, X_test, y_test)

    
    return model

def logistic_regression_model(X_train, X_test, y_train, y_test, feature_names):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    print("\nLogistic Regression Model:")
    evaluate_model(model, X_test, y_test)
    
    return model

def svm_model(X_train, X_test, y_train, y_test, feature_names):
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    print("\nSupport Vector Machine Model:")
    evaluate_model(model, X_test, y_test)

    return model

# if __name__ == "__main__":
#     # Load your dataset
#     df = pd.read_csv("Customertravel.csv")
    
#     # Preprocess the data
#     X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
    
#     # Run models
#     xgb_model = xgboost_model(X_train, X_test, y_train, y_test, feature_names)
#     rf_model = random_forest_model(X_train, X_test, y_train, y_test, feature_names)
#     lr_model = logistic_regression_model(X_train, X_test, y_train, y_test, feature_names)
#     svm_model = svm_model(X_train, X_test, y_train, y_test, feature_names)