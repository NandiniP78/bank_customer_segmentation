import pandas as pd
import numpy as np


def create_customer_features(df):
    """
    Convert transaction-level data into customer-level features
    for customer segmentation using K-Means and DBSCAN.
    """

    data = df.copy()

    # 1. Convert columns to appropriate data types
    data['CustomerDOB'] = pd.to_datetime(
        data['CustomerDOB'], errors='coerce'
    )

    data['TransactionDate'] = pd.to_datetime(
        data['TransactionDate'], errors='coerce'
    )

    data['TransactionAmount (INR)'] = pd.to_numeric(
        data['TransactionAmount (INR)'], errors='coerce'
    )

    data['CustAccountBalance'] = pd.to_numeric(
        data['CustAccountBalance'], errors='coerce'
    )

    # 2. Remove rows without essential customer/date information
    data = data.dropna(
        subset=['CustomerID', 'TransactionDate']
    )

    if data.empty:
        raise ValueError("No valid transaction records found.")

    # 3. Calculate customer age using a common reference date
    reference_date = data['TransactionDate'].max()

    data['CustomerAge'] = (
        reference_date.year - data['CustomerDOB'].dt.year
        - (
            (reference_date.month < data['CustomerDOB'].dt.month)
            |
            (
                (reference_date.month == data['CustomerDOB'].dt.month)
                &
                (reference_date.day < data['CustomerDOB'].dt.day)
            )
        )
    )

    # Remove implausible ages
    data.loc[
        (data['CustomerAge'] < 18) |
        (data['CustomerAge'] > 100),
        'CustomerAge'
    ] = np.nan

    # 4. Aggregate transaction data at customer level
    customer_features = data.groupby('CustomerID').agg(
        CustomerAge=('CustomerAge', 'median'),

        AvgTransactionAmount=(
            'TransactionAmount (INR)', 'mean'
        ),

        MedianTransactionAmount=(
            'TransactionAmount (INR)', 'median'
        ),

        TransactionFrequency=(
            'TransactionID', 'count'
        ),

        TransactionStd=(
            'TransactionAmount (INR)', 'std'
        ),

        TotalTransactionAmount=(
            'TransactionAmount (INR)', 'sum'
        ),

        AccountBalance=(
            'CustAccountBalance', 'median'
        ),

        ActiveDays=(
            'TransactionDate', 'nunique'
        )
    ).reset_index()

    # 5. Handle customers with only one transaction
    customer_features['TransactionStd'] = (
        customer_features['TransactionStd'].fillna(0)
    )

    return customer_features