import joblib
import requests
import io

import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder
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


column_mapping = {
    "ID_x": "ID_x",
    "AUFTRAGSNUMMER": "ORDER_NUMBER",
    "EINGANGSDATUM_UHRZEIT": "RECEIPT_DATE_TIME",
    "LIEFERSCHEINNUMMER": "DELIVERY_NOTE_NUMBER",
    "STATUS": "STATUS",
    "VERPACKT_DATUM_UHRZEIT": "PACKED_DATE_TIME",
    "SONDERFAHRT": "SPECIAL_TRIP",
    "LIEFERSCHEIN_DATUM_UHRZEIT": "DELIVERY_NOTE_DATE_TIME",
    "BEREITGESTELLT_DATUM_UHRZEIT": "PROVIDED_DATE_TIME",
    "TA_DATUM_UHRZEIT": "TA_DATE_TIME",
    "NOTIZ": "NOTE",
    "AUFTRAGANNAHME_DATUM_UHRZEIT": "ORDER_ACCEPTANCE_DATE_TIME",
    "DISPLAYNAME": "DISPLAY_NAME",
    "LAND": "COUNTRY",
    "GELOESCHT": "DELETED",
    "BEMERKUNG_GELOESCHT": "DELETED_COMMENT",
    "ARCHIV": "ARCHIVE",
    "GELOESCHT_DATUM_UHRZEIT": "DELETED_DATE_TIME",
    "TRACKING_NUMMER": "TRACKING_NUMBER",
    "DIENSTLEISTER": "SERVICE_PROVIDER",
    "KLAERUNG_LS": "CLARIFICATION_LS",
    "KLAERUNG_TA": "CLARIFICATION_TA",
    "GEFAHRGUT": "HAZARDOUS_MATERIAL",
    "PO_NUMMER": "PO_NUMBER",
    "ANGEKUENDIGT": "ANNOUNCED",
    "PRIO": "PRIORITY",
    "VERSENDER": "SENDER",
    "ID_y": "ID_y",
    "PACKSTUECK_ID": "PACKAGE_ID",
    "LAENGE_IN_CM": "LENGTH_IN_CM",
    "BREITE_IN_CM": "WIDTH_IN_CM",
    "HOEHE_IN_CM": "HEIGHT_IN_CM",
    "GEWICHT_IN_KG": "WEIGHT_IN_KG",
    "PACKSTUECKART": "PACKAGE_TYPE",
    "TBL_MVA_ID": "TBL_MVA_ID",
}

numerical_columns = ["PACKAGE_DENSITY", "TOTAL_TIME_TAKEN"]


columns_to_drop = [
    "DELIVERY_NOTE_NUMBER",
    "ID_x",
    "ORDER_NUMBER",
    "STATUS",
    "NOTE",
    "DISPLAY_NAME",
    "DELETED",
    "DELETED_COMMENT",
    "ARCHIVE",
    "DELETED_DATE_TIME",
    "TRACKING_NUMBER",
    "SERVICE_PROVIDER",
    "CLARIFICATION_LS",
    "CLARIFICATION_TA",
    "HAZARDOUS_MATERIAL",
    "PO_NUMBER",
    "ANNOUNCED",
    "SENDER",
    "ID_y",
    "PACKAGE_ID",
    "TBL_MVA_ID",
]


# Helper Functions
def categorize_hour(hour):
    if 0 <= hour < 6:
        return 1
    elif 6 <= hour < 12:
        return 2
    elif 12 <= hour < 18:
        return 3
    elif 18 <= hour < 24:
        return 4


def remove_outliers(df, columns):
    # Function to remove outliers from numerical columns
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def generate_synthetic_data(df, times=10):
    # Function to generate synthetic data based on existing dataframe
    synthetic_data = pd.DataFrame()
    for i in range(times):
        temp_df = df.copy()

        # Apply small random variations to numerical columns
        for col in ["LENGTH_IN_CM", "WIDTH_IN_CM", "HEIGHT_IN_CM", "WEIGHT_IN_KG"]:
            temp_df[col] = temp_df[col] * (1 + 0.01 * np.random.randn(len(temp_df)))

        # Add some variation to datetime columns
        for col in [
            "RECEIPT_DATE_TIME",
            "PACKED_DATE_TIME",
            "PROVIDED_DATE_TIME",
            "TA_DATE_TIME",
            "ORDER_ACCEPTANCE_DATE_TIME",
        ]:
            temp_df[col] = temp_df[col] + pd.to_timedelta(
                np.random.randint(-60, 60, size=len(temp_df)), unit="m"
            )

        synthetic_data = pd.concat([synthetic_data, temp_df], ignore_index=True)

    return synthetic_data


def create_label_encodings(df_merged):
    mode_value = df_merged["PACKAGE_TYPE"].mode()[0]
    df_merged["PACKAGE_TYPE"] = df_merged["PACKAGE_TYPE"].replace(" ", mode_value)

    # Label encoding for PACKAGE_TYPE
    label_encoder = LabelEncoder()
    df_merged["PACKAGE_TYPE"] = label_encoder.fit_transform(df_merged["PACKAGE_TYPE"])

    # Handle missing values and convert PRIORITY to integer
    df_merged["PRIORITY"] = df_merged["PRIORITY"].fillna(1)
    df_merged["PRIORITY"] = df_merged["PRIORITY"].replace(" ", 1)
    df_merged["PRIORITY"] = df_merged["PRIORITY"].astype(int)

    # Standardize COUNTRY names and encode them
    df_merged["COUNTRY"] = df_merged["COUNTRY"].str.strip()
    df_merged["COUNTRY"] = df_merged["COUNTRY"].replace(
        [
            "",
            "de",
            "De",
            "DE - FCA Brucker",
            "DE - DAP | Schenker",
            "^DE",
            "DE - Schenker/UPS",
            "dE",
            "de- Koller",
            "DER",
            "TNT | 070275454 | FCA",
            "FCA",
            "D",
            "DR",
        ],
        "DE",
    )

    df_merged["COUNTRY"] = df_merged["COUNTRY"].replace(
        ["Österreich", "at", "AUT"], "AT"
    )
    df_merged["COUNTRY"] = df_merged["COUNTRY"].replace(["IN - Bidadi"], "IN")
    df_merged["COUNTRY"] = df_merged["COUNTRY"].replace(["Dänemark"], "DK")
    df_merged["COUNTRY"] = df_merged["COUNTRY"].replace(["Tr"], "TR")

    label_encoder = LabelEncoder()
    df_merged["COUNTRY"] = label_encoder.fit_transform(df_merged["COUNTRY"])

    return df_merged


def create_density(df_merged):
    df_merged["PACKAGE_VOLUME"] = (
        df_merged["LENGTH_IN_CM"] * df_merged["WIDTH_IN_CM"] * df_merged["HEIGHT_IN_CM"]
    ) / (100 * 100 * 100)

    df_merged["PACKAGE_DENSITY"] = (
        df_merged["WEIGHT_IN_KG"] / df_merged["PACKAGE_VOLUME"]
    )
    mean_package_density = df_merged["PACKAGE_DENSITY"].mean()
    df_merged["PACKAGE_DENSITY"].fillna(mean_package_density, inplace=True)

    # Filter out rows with PACKAGE_DENSITY <= 0
    df_merged = df_merged[df_merged["PACKAGE_DENSITY"] > 0]
    return df_merged


def calculate_time(df_merged):
    if "RECEIPT_DATE_TIME" in df_merged.columns:
        date = pd.to_datetime(df_merged["RECEIPT_DATE_TIME"])
        df_merged["month_of_arrival"] = date.dt.month
        df_merged["day_of_arrival"] = date.dt.day
        df_merged["hour_of_arrival"] = date.dt.hour

    if "DELIVERY_NOTE_DATE_TIME" in df_merged.columns:
        if "RECEIPT_DATE_TIME" in df_merged.columns:
            df_merged["TIME_TAKEN_TO_PACK"] = (
                pd.to_datetime(df_merged["DELIVERY_NOTE_DATE_TIME"])
                - pd.to_datetime(df_merged["RECEIPT_DATE_TIME"])
            ).dt.total_seconds() / 3600

            df_merged = df_merged[df_merged["TIME_TAKEN_TO_PACK"] > 0]

        if "PROVIDED_DATE_TIME" in df_merged.columns:
            df_merged["TIME_DIFF_DELIVERY_NOTE_TO_READY_FOR_SHIPPMENT"] = (
                pd.to_datetime(df_merged["PROVIDED_DATE_TIME"])
                - pd.to_datetime(df_merged["DELIVERY_NOTE_DATE_TIME"])
            ).dt.total_seconds() / 3600  # in hours

            df_merged = df_merged[
                df_merged["TIME_DIFF_DELIVERY_NOTE_TO_READY_FOR_SHIPPMENT"] > 0
            ]
    if (
        "TA_DATE_TIME" in df_merged.columns
        and "PROVIDED_DATE_TIME" in df_merged.columns
    ):

        df_merged["TIME_DIFF_READY_FOR_SHIPPMENT_TO_TRANSPORT_ORDER"] = (
            pd.to_datetime(df_merged["TA_DATE_TIME"])
            - pd.to_datetime(df_merged["PROVIDED_DATE_TIME"])
        ).dt.total_seconds() / 3600  # in hours
    if (
        "TIME_DIFF_READY_FOR_SHIPPMENT_TO_TRANSPORT_ORDER" in df_merged.columns
        and "TIME_TAKEN_TO_PACK" in df_merged.columns
        and "TIME_DIFF_DELIVERY_NOTE_TO_READY_FOR_SHIPPMENT" in df_merged.columns
    ):
        df_merged["TOTAL_TIME_TAKEN"] = (
            df_merged["TIME_DIFF_READY_FOR_SHIPPMENT_TO_TRANSPORT_ORDER"]
            + df_merged["TIME_TAKEN_TO_PACK"]
            + df_merged["TIME_DIFF_DELIVERY_NOTE_TO_READY_FOR_SHIPPMENT"]
        )
    # Calculate MEAN_DELIVERY_NOTE_TO_READY_FOR_SHIPPMENT and MEAN_READY_FOR_SHIPPMENT_TO_TRANSPORT_ORDER
    if "TIME_DIFF_DELIVERY_NOTE_TO_READY_FOR_SHIPPMENT" in df_merged.columns:
        df_merged["MEAN_DELIVERY_NOTE_TO_READY_FOR_SHIPPMENT"] = df_merged.groupby(
            "PACKAGE_TYPE"
        )["TIME_DIFF_DELIVERY_NOTE_TO_READY_FOR_SHIPPMENT"].transform("mean")
    if "TIME_DIFF_READY_FOR_SHIPPMENT_TO_TRANSPORT_ORDER" in df_merged.columns:
        df_merged["MEAN_READY_FOR_SHIPPMENT_TO_TRANSPORT_ORDER"] = df_merged.groupby(
            "PACKAGE_TYPE"
        )["TIME_DIFF_READY_FOR_SHIPPMENT_TO_TRANSPORT_ORDER"].transform("mean")
    if (
        "MEAN_READY_FOR_SHIPPMENT_TO_TRANSPORT_ORDER" in df_merged.columns
        and "MEAN_DELIVERY_NOTE_TO_READY_FOR_SHIPPMENT" in df_merged.columns
    ):
        temp_df = df_merged[
            [
                "PACKAGE_TYPE",
                "MEAN_READY_FOR_SHIPPMENT_TO_TRANSPORT_ORDER",
                "MEAN_DELIVERY_NOTE_TO_READY_FOR_SHIPPMENT",
            ]
        ]
        temp_df = temp_df.drop_duplicates()
        # temp_df.to_excel("../data/mean_processing_time.xlsx")
    return df_merged, temp_df


# Define your main processing code
def pre_processing():
    main_file_path, package_file_path = (
        "../data/Export_TBL_MVA.xlsx",
        "../data/Export_Packstuecke_MVA.xlsx",
    )
    data = pd.read_excel(main_file_path)
    data = data[(data["GELOESCHT"] != 1) & (data["STATUS"] == "TA erstellt")]

    package_data = pd.read_excel(package_file_path)

    df_merged = pd.merge(
        data, package_data, left_on="ID", right_on="TBL_MVA_ID", how="inner"
    )

    # Assuming column_mapping is defined somewhere
    df_merged.rename(columns=column_mapping, inplace=True)

    # remove unwanted columns
    df_merged = df_merged.drop(columns=columns_to_drop)

    # Calculate PACKAGE_VOLUME and PACKAGE_DENSITY
    df_merged = create_density(df_merged)

    # create label encodings for categorical data
    df_merged = create_label_encodings(df_merged)

    if (
        "DELIVERY_NOTE_DATE_TIME" in df_merged.columns
        and "RECEIPT_DATE_TIME" in df_merged.columns
    ):

        # Calculate time_taken_to_pack and remove rows with time_taken_to_pack <= 0
        df_merged, temp_df = calculate_time(df_merged)

        df_merged = generate_synthetic_data(df_merged, times=5)

        # Generate synthetic data

        df_merged = df_merged[df_merged["TIME_TAKEN_TO_PACK"] > 0]

        # Drop unnecessary columns
    df_merged = df_merged.drop(
        columns=[
            "RECEIPT_DATE_TIME",
            "PACKED_DATE_TIME",
            "DELIVERY_NOTE_DATE_TIME",
            "PROVIDED_DATE_TIME",
            "TA_DATE_TIME",
            "ORDER_ACCEPTANCE_DATE_TIME",
            "LENGTH_IN_CM",
            "WIDTH_IN_CM",
            "HEIGHT_IN_CM",
            "WEIGHT_IN_KG",
            "PACKAGE_VOLUME",
            "TIME_DIFF_DELIVERY_NOTE_TO_READY_FOR_SHIPPMENT",
            "TIME_DIFF_READY_FOR_SHIPPMENT_TO_TRANSPORT_ORDER",
            "PROCESSING_TIME_DURATION",
            "MEAN_DELIVERY_NOTE_TO_READY_FOR_SHIPPMENT",
            "MEAN_READY_FOR_SHIPPMENT_TO_TRANSPORT_ORDER",
            "TIME_TAKEN_TO_PACK",
        ],
        errors="ignore",
    )

    # Remove outliers if needed
    df_merged = remove_outliers(df_merged, numerical_columns)

    # Export the pre-processed data if needed
    # df_merged.to_excel("pre_processed.xlsx")

    # Print or further process df_merged as required
    # print(df_merged)
    return df_merged, temp_df


df_merged, temp_df = pre_processing()
df_merged = df_merged.dropna()

# mean_Data = pd.read_excel("mean_processing_time.xlsx")

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
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2
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

final_results = X_test.merge(temp_df, on="PACKAGE_TYPE", how="left")
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
final_results.to_excel("pred.xlsx")
