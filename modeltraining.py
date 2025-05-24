import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 打印 XGBoost 版本
print(f"XGBoost version: {xgb.__version__}")

# 输入数据文件和模型输出文件
INPUT_DATA_FILE = 'final_data.csv'
MODEL_OUTPUT_FILE = 'jak2_pci50_model_tuned_top350_xgb3_v5.joblib' # 再次更新文件名
PLOT_OUTPUT_FILE = 'jak2_prediction_vs_actual_tuned_top350_xgb3_v5.png' # 再次更新文件名
FEATURE_IMPORTANCE_FILE = 'feature_importance.csv'
N_TOP_FEATURES = 350

def load_data(filename):
    """Loads the processed data from the CSV file."""
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    print(f"Data loaded. Shape: {df.shape}")
    return df

def prepare_features_target(df):
    """Loads feature importance, selects top N features, separates features (X)
    and target (y) from the dataframe, and scales selected features."""
    print("Preparing features and target...")
    fallback_triggered = False

    try:
        print(f"Loading feature importance from {FEATURE_IMPORTANCE_FILE}...")
        importance_df = pd.read_csv(FEATURE_IMPORTANCE_FILE)
        top_features = importance_df.sort_values(by='importance', ascending=False)['feature'].head(N_TOP_FEATURES).tolist()
        print(f"Selected top {N_TOP_FEATURES} features based on importance.")
    except FileNotFoundError:
        print(f"Error: Feature importance file '{FEATURE_IMPORTANCE_FILE}' not found.")
        print("Proceeding with all original features that start with 'morgan_', 'maccs_' or are in the defined descriptor list.")
        fallback_triggered = True
    except KeyError as e:
        print(f"Error: Column '{e}' not found in feature importance file.")
        print("Please ensure the file has 'feature' and 'importance' columns.")
        print("Proceeding with all original features that start with 'morgan_', 'maccs_' or are in the defined descriptor list.")
        fallback_triggered = True

    if fallback_triggered:
        morgan_cols = [col for col in df.columns if col.startswith('morgan_')]
        maccs_cols = [col for col in df.columns if col.startswith('maccs_')]
        desc_cols = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
                     'FractionCSP3', 'NumAromaticRings', 'NumAliphaticRings',
                     'NumRotatableBonds', 'ExactMolWt', 'qed', 'BalabanJ', 'BertzCT']
        top_features = [col for col in morgan_cols + maccs_cols + desc_cols if col in df.columns]
        if not top_features:
            raise ValueError("Fallback feature selection failed: No recognized features found in the data file.")
        print(f"Using {len(top_features)} fallback features.")

    available_features_in_df = df.columns.tolist()
    selected_features = [f for f in top_features if f in available_features_in_df]

    if not selected_features:
        raise ValueError("No features selected. Check feature importance file and data file consistency.")

    if not fallback_triggered and len(selected_features) < N_TOP_FEATURES:
        print(f"Warning: Some features from the top {N_TOP_FEATURES} list were not found in the data file. Using {len(selected_features)} features.")
        missing_features = [f for f in top_features if f not in available_features_in_df]
        print(f"Missing features example (max 5): {missing_features[:5]}")

    print(f"Using final selected features (first 5): {selected_features[:5]}...")
    X_raw = df[selected_features].copy()

    for col in X_raw.columns:
        X_raw.loc[:, col] = pd.to_numeric(X_raw[col], errors='coerce')
    X_raw.fillna(X_raw.median(), inplace=True)

    X = X_raw.values
    y = df['pCI50'].values
    print(f"Original Features shape (Selected & Available): ({X.shape[0]}, {len(selected_features)})")
    print(f"Target shape: {y.shape}")

    print("Scaling selected features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Scaled Features shape: {X_scaled.shape}")

    return X_scaled, y, scaler, selected_features

def train_evaluate_model(X_train, y_train, X_test, y_test, grid_search_best_params):
    """Trains the final model with best parameters found by GridSearch and early stopping."""
    print("Training final model with best parameters and early stopping...")

    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        n_estimators=1500,
        early_stopping_rounds=50,
        eval_metric='rmse',
        **grid_search_best_params
    )

    eval_set = [(X_test, y_test)]
    final_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )

    if hasattr(final_model, 'best_iteration') and final_model.best_iteration is not None:
        actual_trees = final_model.best_iteration
        print(f"Actual number of trees used (from model.best_iteration): {actual_trees}")
    else:
        try:
            actual_trees = final_model.get_booster().num_boosted_rounds()
            print(f"Actual number of trees used (from booster.num_boosted_rounds()): {actual_trees}")
        except Exception:
            actual_trees = final_model.get_params()['n_estimators']
            print(f"Could not determine exact early stopping point. Using n_estimators: {actual_trees}")

    print("Evaluating final tuned model...")
    y_pred_train = final_model.predict(X_train)
    y_pred_test = final_model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)

    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print("--- Tuned Training Set Performance ---")
    print(f"R-squared (R2): {r2_train:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_train:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_train:.4f}")
    print("-------------------------------------")
    print("--- Tuned Test Set Performance ---")
    print(f"R-squared (R2): {r2_test:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_test:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_test:.4f}")
    print("--------------------------------")

    return final_model, y_test, y_pred_test

def run_grid_search(X_train, y_train):
    """Performs GridSearchCV to find the best hyperparameters."""
    # Best parameters found in your previous run:
    # {'colsample_bytree': 0.75, 'gamma': 0.25, 'learning_rate': 0.025,
    # 'max_depth': 12, 'reg_alpha': 0.3, 'reg_lambda': 0.6, 'subsample': 0.8}
    # We will construct a param_grid around these values, aiming for a similar number of candidates (972)
    # 972 = 3*3*3*3*2*2*1.5 (not possible with int options) or use different combinations
    # Let's aim for a grid that includes these points.
    # Example: 3 (lr) * 3 (depth) * 2 (subsample) * 2 (colsample) * 3 (gamma) * 3 (alpha) * 2 (lambda) = 324 * 2 = 648.
    # To get 972: 3^5 * 2^2 = 243 * 4 = 972. (5 params with 3 options, 2 params with 2 options)

    param_grid_targeted = {
        'learning_rate': [0.02, 0.025, 0.03],       # 3 options (includes 0.025)
        'max_depth': [11, 12, 13],                  # 3 options (includes 12)
        'subsample': [0.75, 0.8, 0.85],             # 3 options (includes 0.8)
        'colsample_bytree': [0.7, 0.75, 0.8],       # 3 options (includes 0.75)
        'gamma': [0.2, 0.25, 0.3],                  # 3 options (includes 0.25)
        'reg_alpha': [0.25, 0.3],                   # 2 options (includes 0.3)
        'reg_lambda': [0.55, 0.6]                   # 2 options (includes 0.6)
    }
    # Total combinations: 3*3*3*3*3*2*2 = 972

    xgb_reg_for_grid = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        n_estimators=100
    )

    print("Performing GridSearchCV for hyperparameter tuning...")
    num_candidates = np.prod([len(v) for v in param_grid_targeted.values()])
    print(f"GridSearchCV will fit {num_candidates} candidates, each with 5 folds, totalling {num_candidates * 5} fits.")

    search = GridSearchCV(
        estimator=xgb_reg_for_grid,
        param_grid=param_grid_targeted,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=2
    )
    search.fit(X_train, y_train)
    print(f"Best parameters found by GridSearch: {search.best_params_}")
    print(f"Best R2 score during GridSearch: {search.best_score_:.4f}")
    return search.best_params_

def save_model_artifacts(model_to_save, scaler_to_save, feature_cols_to_save, filename):
    """Saves the model, scaler, and feature columns."""
    print(f"Saving model artifacts to {filename}...")
    artifacts = {
        'model': model_to_save,
        'scaler': scaler_to_save,
        'feature_cols': feature_cols_to_save
    }
    joblib.dump(artifacts, filename)
    print("Model artifacts saved.")

def plot_results(y_true, y_pred, filename):
    print(f"Plotting results to {filename}...")
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor=None)
    min_val = min(y_true.min(), y_pred.min()) - 0.5
    max_val = max(y_true.max(), y_pred.max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
    plt.xlabel("Actual pCI50")
    plt.ylabel("Predicted pCI50")
    plt.title("Actual vs. Predicted pCI50 (Tuned Test Set)")
    plt.grid(True)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(filename)
    print("Plot saved.")

def main():
    df = load_data(INPUT_DATA_FILE)
    try:
        X_scaled, y, scaler, feature_cols = prepare_features_target(df)
    except ValueError as e:
        print(f"Error during feature preparation: {e}")
        return

    if X_scaled.shape[1] == 0 :
        print("Error: No features available after preparation step. Aborting.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features used for training: {X_train.shape[1]}")

    if X_train.shape[1] == 0:
        print("Error: Training data has no features. Aborting model training.")
        return

    best_params = run_grid_search(X_train, y_train)
    model, y_true_test, y_pred_test = train_evaluate_model(X_train, y_train, X_test, y_test, best_params)

    # 将 model, scaler, feature_cols 作为独立参数传递
    save_model_artifacts(model, scaler, feature_cols, MODEL_OUTPUT_FILE)

    plot_results(y_true_test, y_pred_test, PLOT_OUTPUT_FILE)

if __name__ == '__main__':
    main()