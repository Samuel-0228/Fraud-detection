import pandas as pd
import numpy as np
from ipaddress import ip_address
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def ip_to_int(ip_str):
    """Convert IP address string to integer."""
    try:
        return int(ip_address(ip_str))
    except:
        return np.nan


def clean_fraud_data(df):
    """Clean Fraud_Data.csv."""
    df = df.copy()  # Avoid modifying original
    # Convert timestamps
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # Handle missing age (impute median)
    if 'age' in df.columns:
        median_age = df['age'].median()
        # Non-inplace to avoid FutureWarning
        df['age'] = df['age'].fillna(median_age)

    # Cap purchase_value outliers (99th percentile)
    if 'purchase_value' in df.columns:
        p99 = df['purchase_value'].quantile(0.99)
        df['purchase_value'] = np.clip(df['purchase_value'], 0, p99)

    # Remove duplicates
    df = df.drop_duplicates()

    return df


def merge_geolocation(df, ip_df):
    """Merge IP to country using range-based lookup."""
    # Convert IP to int
    df['ip_int'] = df['ip_address'].apply(ip_to_int)
    ip_df['lower_int'] = ip_df['lower_bound_ip_address'].astype('uint64')
    ip_df['upper_int'] = ip_df['upper_bound_ip_address'].astype('uint64')

    # Sort and merge asof
    df_sorted = df.sort_values('ip_int').reset_index(drop=True)
    ip_sorted = ip_df.sort_values('lower_int').reset_index(drop=True)

    merged = pd.merge_asof(df_sorted, ip_sorted, left_on='ip_int', right_on='lower_int',
                           direction='backward')
    merged = merged.drop(
        ['ip_int', 'lower_int', 'upper_bound_ip_address'], axis=1, errors='ignore')
    merged = merged.rename(columns={'country': 'country_code'})

    # Handle unmapped
    merged['country_code'] = merged['country_code'].fillna('Unknown')

    return merged.sort_index()


def prepare_features(df, numerical_cols, categorical_cols):
    """Scale numerics and encode categoricals."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ])
    return preprocessor.fit_transform(df[numerical_cols + categorical_cols])
