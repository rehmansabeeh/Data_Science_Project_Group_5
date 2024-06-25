from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import xgboost as xgb


def model_results(X_train, X_test, y_train, y_test):
    model_xgb = xgb.XGBRegressor(max_depth=7, n_estimators=20)
    scaler_xgb = StandardScaler()
    X_train_scaled_xgb = scaler_xgb.fit_transform(X_train)
    X_test_scaled_xgb = scaler_xgb.transform(X_test)
    model_xgb.fit(X_train_scaled_xgb, y_train)
    y_pred_xgb = model_xgb.predict(X_test_scaled_xgb)

    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)

    print("mae_xgb : ", mae_xgb, " r2_xgb : ", r2_xgb, " mse_xgb : ", mse_xgb)
    # Define the parameter grid to search over
    param_grid = {
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "activation": ["relu", "tanh"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "adaptive"],
        "max_iter": [500, 1000],
        "random_state": [42],
    }

    mlp = MLPRegressor()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        verbose=2,
    )
    scaler_mlp = StandardScaler()
    X_train_scaled_mlp = scaler_mlp.fit_transform(X_train)
    X_test_scaled_mlp = scaler_mlp.transform(X_test)

    # Perform Grid Search to find the best parameters
    grid_search.fit(X_train_scaled_mlp, y_train)

    # Evaluate best model on test set
    best_model = grid_search.best_estimator_

    best_model.fit(X_train_scaled_mlp, y_train)
    y_pred_mlp = best_model.predict(X_test_scaled_mlp)

    mse_mlp = mean_squared_error(y_test, y_pred_mlp)
    mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
    r2_mlp = r2_score(y_test, y_pred_mlp)
    print("mse_mlp : ", mse_mlp, " mae_mlp : ", mae_mlp, " r2_mlp : ", r2_mlp)

    # Printing the metrics
    models = ["XGBRegressor", "MLPRegressor"]
    metrics = ["MAE", "MSE", "R²"]
    mae_scores = [mae_xgb, mae_mlp]
    mse_scores = [mse_xgb, mse_mlp]
    r2_scores = [r2_xgb, r2_mlp]
    plt.figure(figsize=(15, 5))

    # Plot MAE
    plt.subplot(1, 3, 1)
    plt.bar(models, mae_scores, color=["green", "blue"])
    plt.ylabel("Score")
    plt.title("Mean Absolute Error (MAE)")

    # Plot MSE
    plt.subplot(1, 3, 2)
    plt.bar(models, mse_scores, color=["orange", "purple"])
    plt.ylabel("Score")
    plt.title("Mean Squared Error (MSE)")

    # Plot R²
    plt.subplot(1, 3, 3)
    plt.bar(models, r2_scores, color=["red", "pink"])
    plt.ylabel("Score")
    plt.title("R² Score")

    plt.tight_layout()
    plt.show()
    return best_model
