from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from prediction import (
    preprocess_data,
    xgboost_model,
    random_forest_model,
    logistic_regression_model,
    svm_model
)

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded CSV file
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    # Preprocess the data
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)

    # Initialize a dictionary to store results
    results = {}

    # Run all models and store their results
    models = [
        ("XGBoost", xgboost_model),
        ("Random Forest", random_forest_model),
        ("Logistic Regression", logistic_regression_model),
        ("SVM", svm_model)
    ]

    for model_name, model_func in models:
        model = model_func(X_train, X_test, y_train, y_test, feature_names)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        results[model_name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "auc_roc": float(roc_auc_score(y_test, y_pred_proba))
        }

    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)