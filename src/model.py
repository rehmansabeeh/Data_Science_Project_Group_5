import joblib
import requests
import io
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def download_from_sharepoint_as_pickle():
    url = "https://drive.google.com/uc?export=download&id=1XFSZLsTqZxYFkWauip9gvYTYxN9Xu_7k"

    # Step 1: Download the file
    response = requests.get(url)
    bytes_file_obj = io.BytesIO()
    bytes_file_obj.write(response.content)
    bytes_file_obj.seek(0)
    pickle_obj = joblib.load(bytes_file_obj)
    return pickle_obj


def create_datetime(row):
    # Assuming the current year for the datetime
    year = datetime.now().year
    row["month_of_arrival"] = row["month_of_arrival"].astype(int)
    row["day_of_arrival"] = row["day_of_arrival"].astype(int)
    row["hour_of_arrival"] = row["hour_of_arrival"].astype(int)
    return datetime(
        year, row["month_of_arrival"], row["day_of_arrival"], row["hour_of_arrival"]
    )


# Apply the function to each row


def model_results(df_merged, mean_Data):
    # Split the data into train and test sets
    if "TOTAL_TIME_TAKEN" in df_merged.columns:
        X = df_merged.drop(columns=["TOTAL_TIME_TAKEN"])

        y = df_merged["TOTAL_TIME_TAKEN"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # List of models to evaluate

        models = {
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "SVR": SVR(),
            "MLP": MLPRegressor(),
        }

        # Dictionary to store results and predictions

        results = {}

        predictions = {}

        scaler_mlp = StandardScaler()

        X_train_scaled = scaler_mlp.fit_transform(X_train)

        X_test_scaled = scaler_mlp.transform(X_test)

        # Train and evaluate each model

        for name, model in models.items():

            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            mse = mean_squared_error(y_test, y_pred)

            mae = mean_absolute_error(y_test, y_pred)

            r2 = r2_score(y_test, y_pred)

            results[name] = {"MSE": mse, "MAE": mae, "R²": r2}

            predictions[name] = y_pred

        # Print results
        for name, metrics in results.items():

            print(f"Model: {name}")

            for metric, value in metrics.items():

                print(f"{metric}: {value}")

            print("\n")

        # Plot parity plots

        plt.figure(figsize=(20, 10))

        for i, (name, y_pred) in enumerate(predictions.items()):

            plt.subplot(3, 3, i + 1)

            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot(
                [y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                "r--",
                linewidth=2,
            )

            plt.title(f"{name} Parity Plot")

            plt.xlabel("Actual Values")

            plt.ylabel("Predicted Values")

            plt.grid(True)

        plt.tight_layout()

        plt.show()

        print("--------- best model from these is Decision Tree --------")

        model = DecisionTreeRegressor()
        name = "Decision Tree"

        results = {}
        predictions = {}

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        X_test = pd.DataFrame(X_test, columns=X.columns)
        X_test["predictions"] = y_pred
        X_test["actual"] = y_test

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MSE": mse, "MAE": mae, "R²": r2}
        predictions[name] = y_pred

        # Print results
        for name, metrics in results.items():
            print(f"Model: {name}")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
            print("\n")

        # Plot parity plots
        plt.figure(figsize=(20, 10))

        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            linewidth=2,
        )
        plt.title(f"{name} Parity Plot")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        joblib.dump(model, "model.joblib")
    else:
        X_test = df_merged
        model = download_from_sharepoint_as_pickle()
        y_pred = model.predict(X_test)

        X = pd.DataFrame(X_test, columns=X_test.columns)
        X_test["predictions"] = y_pred

    final_results = X_test.merge(mean_Data, on="PACKAGE_TYPE", how="left")
    final_results = final_results.drop(
        columns=[
            "Unnamed: 0",
            "MEAN_READY_FOR_SHIPPMENT_TO_TRANSPORT_ORDER",
            "MEAN_DELIVERY_NOTE_TO_READY_FOR_SHIPPMENT",
        ]
    )
    numeric_columns = final_results.select_dtypes(include="number").columns
    final_results[numeric_columns] = final_results[numeric_columns].round(2)

    for column in final_results.select_dtypes(include=["float64", "int64"]).columns:
        final_results[column] = final_results[column].apply(
            lambda x: str(x).replace(".", ",")
        )
    final_results.to_excel("../data/pred.xlsx")
    return final_results
