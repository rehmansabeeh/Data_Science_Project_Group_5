import pandas as pd
from sklearn.preprocessing import LabelEncoder
from constants import column_mapping

from sklearn.ensemble import IsolationForest


def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def remove_outliers_isolation_forest(df, columns, contamination=0.05):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    for col in columns:
        df["outlier"] = iso_forest.fit_predict(df[[col]])
        df = df[df["outlier"] == 1]
        df.drop(columns="outlier", inplace=True)
    return df


numerical_columns = [
    # "PACKAGE_VOLUME",
    "PACKAGE_DENSITY",
    "TIME_DIFF_VERPACKT_TO_LIEFERSCHEIN",
    "TIME_DIFF_LIEFERSCHEIN_TO_BEREITGESTELLT",
    "TIME_DIFF_BEREITGESTELLT_TO_TA",
    "HOUR_OF_DAY_EINGANG",
    "HOUR_OF_DAY_VERPACKT",
    "HOUR_OF_DAY_LIEFERSCHEIN",
    "HOUR_OF_DAY_BEREITGESTELLT",
    "HOUR_OF_DAY_TA",
    "PROCESSING_TIME_DURATION",
]


def preprocessing(main_file_path, package_file_path):
    # package details
    data = pd.read_excel(main_file_path)
    data = data[(data["GELOESCHT"] != 1) & (data["STATUS"] == "TA erstellt")]
    package_data = pd.read_excel(package_file_path)
    df_merged = pd.merge(
        data, package_data, left_on="ID", right_on="TBL_MVA_ID", how="inner"
    )

    df_merged.rename(columns=column_mapping, inplace=True)

    df_merged = df_merged[df_merged["WEIGHT_IN_KG"] > 0]

    mode_value = df_merged["PACKAGE_TYPE"].mode()[0]
    df_merged["PACKAGE_TYPE"] = df_merged["PACKAGE_TYPE"].replace(" ", mode_value)
    label_encoder = LabelEncoder()
    df_merged["PACKAGE_TYPE"] = label_encoder.fit_transform(df_merged["PACKAGE_TYPE"])

    df_merged["PACKAGE_VOLUME"] = (
        df_merged["LENGTH_IN_CM"] * df_merged["WIDTH_IN_CM"] * df_merged["HEIGHT_IN_CM"]
    ) / (100 * 100 * 100)

    df_merged["PACKAGE_DENSITY"] = (
        df_merged["WEIGHT_IN_KG"] / df_merged["PACKAGE_VOLUME"]
    )
    mean_package_density = df_merged["PACKAGE_DENSITY"].mean()
    df_merged["PACKAGE_DENSITY"].fillna(mean_package_density, inplace=True)

    # time difference
    df_merged["PROCESSING_TIME_DURATION"] = (
        pd.to_datetime(df_merged["TA_DATE_TIME"])
        - pd.to_datetime(df_merged["RECEIPT_DATE_TIME"])
    ).dt.total_seconds() / 3600  # in hours

    df_merged["TIME_DIFF_VERPACKT_TO_LIEFERSCHEIN"] = (
        pd.to_datetime(df_merged["DELIVERY_NOTE_DATE_TIME"])
        - pd.to_datetime(df_merged["PACKED_DATE_TIME"])
    ).dt.total_seconds() / 3600  # in hours
    df_merged = df_merged[df_merged["TIME_DIFF_VERPACKT_TO_LIEFERSCHEIN"] > 0]

    df_merged["TIME_DIFF_LIEFERSCHEIN_TO_BEREITGESTELLT"] = (
        pd.to_datetime(df_merged["PROVIDED_DATE_TIME"])
        - pd.to_datetime(df_merged["DELIVERY_NOTE_DATE_TIME"])
    ).dt.total_seconds() / 3600  # in hours
    df_merged = df_merged[df_merged["TIME_DIFF_LIEFERSCHEIN_TO_BEREITGESTELLT"] > 0]

    df_merged["TIME_DIFF_BEREITGESTELLT_TO_TA"] = (
        pd.to_datetime(df_merged["TA_DATE_TIME"])
        - pd.to_datetime(df_merged["PROVIDED_DATE_TIME"])
    ).dt.total_seconds() / 3600  # in hours

    # specific time

    df_merged["HOUR_OF_DAY_EINGANG"] = pd.to_datetime(
        df_merged["RECEIPT_DATE_TIME"]
    ).dt.hour

    df_merged["HOUR_OF_DAY_VERPACKT"] = pd.to_datetime(
        df_merged["PACKED_DATE_TIME"]
    ).dt.hour

    df_merged["HOUR_OF_DAY_LIEFERSCHEIN"] = pd.to_datetime(
        df_merged["DELIVERY_NOTE_DATE_TIME"]
    ).dt.hour

    df_merged["HOUR_OF_DAY_BEREITGESTELLT"] = pd.to_datetime(
        df_merged["PROVIDED_DATE_TIME"]
    ).dt.hour

    df_merged["HOUR_OF_DAY_TA"] = pd.to_datetime(df_merged["TA_DATE_TIME"]).dt.hour

    most_common_provider = df_merged["SERVICE_PROVIDER"].mode()[0]
    df_merged["SERVICE_PROVIDER"] = df_merged["SERVICE_PROVIDER"].replace(
        " ", most_common_provider
    )
    df_merged["SERVICE_PROVIDER"] = df_merged["SERVICE_PROVIDER"].replace(
        "0", most_common_provider
    )

    df_merged["SERVICE_PROVIDER"] = df_merged["SERVICE_PROVIDER"].fillna(
        most_common_provider
    )

    label_encoder = LabelEncoder()
    df_merged["SERVICE_PROVIDER"] = label_encoder.fit_transform(
        df_merged["SERVICE_PROVIDER"]
    )

    df_merged["PRIORITY"] = df_merged["PRIORITY"].fillna(1)
    df_merged["PRIORITY"] = df_merged["PRIORITY"].replace(" ", 1)
    df_merged["PRIORITY"] = df_merged["PRIORITY"].astype(int)

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

    df_merged["HISTORICAL_PROCESSING_TIME_MEAN"] = df_merged.groupby("PACKAGE_TYPE")[
        "PROCESSING_TIME_DURATION"
    ].transform("mean")
    df_merged["HISTORICAL_PROCESSING_TIME_MEDIAN"] = df_merged.groupby("PACKAGE_TYPE")[
        "PROCESSING_TIME_DURATION"
    ].transform("median")
    df_merged["HISTORICAL_PROCESSING_TIME_STD"] = df_merged.groupby("PACKAGE_TYPE")[
        "PROCESSING_TIME_DURATION"
    ].transform("std")

    df_merged = df_merged.drop(
        columns=[
            "ID_x",
            "ORDER_NUMBER",
            "RECEIPT_DATE_TIME",
            "DELIVERY_NOTE_NUMBER",
            "STATUS",
            "PACKED_DATE_TIME",
            "SPECIAL_TRIP",
            "DELIVERY_NOTE_DATE_TIME",
            "PROVIDED_DATE_TIME",
            "TA_DATE_TIME",
            "NOTE",
            "ORDER_ACCEPTANCE_DATE_TIME",
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
            "LENGTH_IN_CM",
            "WIDTH_IN_CM",
            "HEIGHT_IN_CM",
            "WEIGHT_IN_KG",
            "TBL_MVA_ID",
            "PACKAGE_VOLUME",
            # "PACKAGE_TYPE",
            # "COUNTRY",
            # "PRIORITY",
            # "PACKAGE_DENSITY",
            # "PROCESSING_TIME_DURATION",
            # "TIME_DIFF_VERPACKT_TO_LIEFERSCHEIN",
            # "TIME_DIFF_LIEFERSCHEIN_TO_BEREITGESTELLT",
            # "TIME_DIFF_BEREITGESTELLT_TO_TA",
            # "HOUR_OF_DAY_EINGANG",
            # "HOUR_OF_DAY_VERPACKT",
            # "HOUR_OF_DAY_LIEFERSCHEIN",
            # "HOUR_OF_DAY_BEREITGESTELLT",
            "HISTORICAL_PROCESSING_TIME_MEAN",
            "HISTORICAL_PROCESSING_TIME_MEDIAN",
            "HISTORICAL_PROCESSING_TIME_STD",
            # "HOUR_OF_DAY_TA",
        ]
    )
    df_merged = remove_outliers(df_merged, numerical_columns)

    return df_merged
