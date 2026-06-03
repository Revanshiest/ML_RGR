import os
import pickle
import pandas as pd
import numpy as np
import optuna

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor, BaggingRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def tune_and_train(name, model_class, param_space, X_train, y_train, X_test, y_test, n_trials=20, fit_kwargs=None):
    print(f"\n--- Tuning {name} ---")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    def objective(trial):
        params = param_space(trial)
        model = model_class(**params)
        if fit_kwargs:
            model.fit(X_tr, y_tr, **fit_kwargs)
        else:
            model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best params for {name}: {study.best_params}")
    
    best_model = model_class(**study.best_params)
    if fit_kwargs:
        best_model.fit(X_train, y_train, **fit_kwargs)
    else:
        best_model.fit(X_train, y_train)
        
    test_preds = best_model.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    r2 = r2_score(y_test, test_preds)
    
    print(f"{name} Test MSE: {mse:.4f}")
    print(f"{name} Test R2: {r2:.4f}")
    
    return best_model

def main():
    print("Loading data...")
    df_train = pd.read_csv("../data/moldova_train.csv", sep=",", encoding="utf-8")
    df_test = pd.read_csv("../data/moldova_test.csv", sep=",", encoding="utf-8")

    cols_to_drop = ['Transmission_Manual', 'Fuel_type_Diesel', 'Style_Pickup', 'Make', 'Model']
    df_train = df_train.drop(columns=cols_to_drop, errors='ignore')
    df_test = df_test.drop(columns=cols_to_drop, errors='ignore')

    X_train_reg = df_train.drop(columns=['Price(euro)']).values
    y_train_reg = df_train['Price(euro)'].values
    X_test_reg = df_test.drop(columns=['Price(euro)']).values
    y_test_reg = df_test['Price(euro)'].values

    os.makedirs("models", exist_ok=True)
    print("Training models with Optuna (Metric: MSE)...")

    # ML1: XGBoost (Replaced Linear Regression)
    def xgb_space(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'random_state': 42,
            'n_jobs': -1
        }
    ml1 = tune_and_train("ML1 (XGBoost)", XGBRegressor, xgb_space, X_train_reg, y_train_reg, X_test_reg, y_test_reg, n_trials=15)
    with open("models/ml1_xgboost.pkl", "wb") as f:
        pickle.dump(ml1, f)

    # ML2: GradientBoosting
    def gb_space(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'random_state': 42
        }
    ml2 = tune_and_train("ML2 (GradientBoosting)", GradientBoostingRegressor, gb_space, X_train_reg, y_train_reg, X_test_reg, y_test_reg, n_trials=15)
    with open("models/ml2_gradient_boosting.pkl", "wb") as f:
        pickle.dump(ml2, f)

    # ML3: CatBoost
    def cb_space(trial):
        return {
            'iterations': trial.suggest_int('iterations', 100, 300),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'random_seed': 42,
            'verbose': False
        }
    ml3 = tune_and_train("ML3 (CatBoost)", CatBoostRegressor, cb_space, X_train_reg, y_train_reg, X_test_reg, y_test_reg, n_trials=15)
    ml3.save_model("models/ml3_catboost.cbm")

    # ML4: Random Forest
    def rf_space(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'random_state': 42,
            'n_jobs': -1
        }
    ml4 = tune_and_train("ML4 (Random Forest)", RandomForestRegressor, rf_space, X_train_reg, y_train_reg, X_test_reg, y_test_reg, n_trials=15)
    with open("models/ml4_random_forest.pkl", "wb") as f:
        pickle.dump(ml4, f)

    # ML5: Stacking
    print("\n--- Tuning ML5 (Stacking) ---")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_reg, y_train_reg, test_size=0.2, random_state=42)
    def objective_stack(trial):
        rf_n = trial.suggest_int('rf_n_estimators', 20, 80)
        gb_n = trial.suggest_int('gb_n_estimators', 20, 80)
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=rf_n, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=gb_n, random_state=42))
        ]
        model = StackingRegressor(estimators=estimators, final_estimator=XGBRegressor(random_state=42, n_jobs=-1))
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    study_stack = optuna.create_study(direction='minimize')
    study_stack.optimize(objective_stack, n_trials=10)
    print(f"Best params for Stacking: {study_stack.best_params}")
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=study_stack.best_params['rf_n_estimators'], random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=study_stack.best_params['gb_n_estimators'], random_state=42))
    ]
    ml5 = StackingRegressor(estimators=estimators, final_estimator=XGBRegressor(random_state=42, n_jobs=-1))
    ml5.fit(X_train_reg, y_train_reg)
    ml5_preds = ml5.predict(X_test_reg)
    print(f"ML5 (Stacking) Test MSE: {mean_squared_error(y_test_reg, ml5_preds):.4f}")
    print(f"ML5 (Stacking) Test R2: {r2_score(y_test_reg, ml5_preds):.4f}")
    with open("models/ml5_stacking.pkl", "wb") as f:
        pickle.dump(ml5, f)

    # ML6: Bagging (Replaced MLPRegressor)
    def bag_space(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 10, 100),
            'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
            'max_features': trial.suggest_float('max_features', 0.5, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
    ml6 = tune_and_train("ML6 (Bagging)", BaggingRegressor, bag_space, X_train_reg, y_train_reg, X_test_reg, y_test_reg, n_trials=15)
    with open("models/ml6_bagging.pkl", "wb") as f:
        pickle.dump(ml6, f)

    print("\nModels successfully tuned, trained and saved to 'models/' directory.")

if __name__ == "__main__":
    main()
