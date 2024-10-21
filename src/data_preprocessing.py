import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import make_pipeline

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean the data from a CSV file.

    This function loads the data from the provided filepath, removes any duplicate rows,
    and returns the cleaned DataFrame.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned data.
    """
    df = pd.read_csv(filepath)
    df.drop_duplicates(inplace=True)
    return df

def outlier_capping(df: pd.DataFrame, outliers: list) -> pd.DataFrame:
    """
    Cap outliers in the specified columns of the DataFrame.

    This function applies the IQR (Interquartile Range) method to cap values that are
    beyond 1.5 times the IQR in the specified columns. Values greater than the upper
    bound are capped at the upper bound, and values below the lower bound are capped
    at the lower bound.

    Args:
        df (pd.DataFrame): The input DataFrame.
        outliers (list): A list of column names in which to apply outlier capping.

    Returns:
        pd.DataFrame: The DataFrame with outliers capped in the specified columns.
    """
    for i in outliers:
        q1 = df[i].quantile(0.25)
        q3 = df[i].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q3 - 1.5 * iqr
        df.loc[df[i] > upper_limit, i] = upper_limit
        df.loc[df[i] < lower_limit, i] = lower_limit
    return df

def split_data(df: pd.DataFrame, target: str):
    """
    Split the data into training and testing sets.

    This function splits the DataFrame into features (X) and target (y), and then divides
    them into training and test sets using a 70-30 split.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target (str): The column name of the target variable.

    Returns:
        tuple: A tuple containing four DataFrames: X_train, X_test, y_train, y_test.
    """
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=0.3, random_state=42)

def create_preprocessor(scaler_type: str):
    """
    Create a preprocessing pipeline with imputation and scaling.

    This function returns a pipeline that first imputes missing values using K-Nearest
    Neighbors (KNN) imputation and then applies a scaling method based on the provided
    scaler type.

    Args:
        scaler_type (str): The type of scaler to use. Acceptable values are 'standard',
                           'minmax', or 'robust'. If none of these values are provided,
                           'robust' scaling is used by default.

    Returns:
        sklearn.pipeline.Pipeline: A pipeline that applies KNN imputation and scaling.
    """
    imputer = KNNImputer(n_neighbors=3)
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()

    return make_pipeline(imputer, scaler)
