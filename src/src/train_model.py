import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

DATA_PATH = "data/car_data.csv"   # place your CSV here
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def prepare(df):
    # expects columns: year, mileage, power, price
    df = df.dropna(subset=['price']).copy()
    current_year = 2025
    df['age'] = current_year - df['year']
    X = df[['age','mileage','power']]
    y = df['price']
    return X, y

def train():
    df = load_data()
    X, y = prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if _name_ == "_main_":
    train()
