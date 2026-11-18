
import pandas as pd
import numpy as np


# ==================================================
# 1. DATETIME FEATURE ENGINEERING
# ==================================================

def create_datetime_features(df):

    df = df.copy()

    # Convert transaction date
    df['TransactionDate'] = pd.to_datetime(
        df['TransactionDate'],
        errors='coerce'
    )

    # Extract day of week
    df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek

    # Weekend indicator
    df['IsWeekend'] = (
        df['DayOfWeek'].isin([5, 6]).astype(int)
    )

    # Extract hour from HHMMSS transaction time
    time_str = (
        df['TransactionTime']
        .astype('string')
        .str.replace(r'\.0$', '', regex=True)
        .str.zfill(6)
    )

    df['TransactionHour'] = pd.to_numeric(
        time_str.str[:2],
        errors='coerce'
    )

    # Validate hour
    df.loc[
        ~df['TransactionHour'].between(0, 23),
        'TransactionHour'
    ] = np.nan

    return df


# ==================================================
# 2. TRANSACTION-LEVEL FEATURE ENGINEERING
# ==================================================

def create_transaction_features(df):

    df = df.copy()

    # Transaction amount relative to account balance
    df['AmountBalanceRatio'] = (
        df['TransactionAmount (INR)'] /
        df['CustAccountBalance'].replace(0, np.nan)
    )

    return df


# ==================================================
# 3. CUSTOMER-RELATIVE FEATURE ENGINEERING
# ==================================================

def create_customer_features(df):

    df = df.copy()

    # Ensure chronological ordering
    df = df.sort_values(
        ['CustomerID', 'TransactionDate', 'TransactionTime']
    ).copy()

    grouped = df.groupby('CustomerID')[
        'TransactionAmount (INR)'
    ]

    # Historical average transaction amount
    df['HistoricalAvgAmount'] = (
        grouped.transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    # Historical standard deviation
    df['HistoricalStdAmount'] = (
        grouped.transform(
            lambda x: x.shift(1).expanding().std()
        )
    )

    # Number of previous transactions
    df['PreviousTransactionCount'] = (
        df.groupby('CustomerID').cumcount()
    )

    # Deviation from historical average
    df['HistoricalAmountDeviation'] = (
        df['TransactionAmount (INR)'] -
        df['HistoricalAvgAmount']
    )

    # Historical z-score
    df['HistoricalAmountZScore'] = (
        df['HistoricalAmountDeviation'] /
        df['HistoricalStdAmount'].replace(0, np.nan)
    )

    return df


# ==================================================
# 4. POPULATION-LEVEL FEATURE SET
# ==================================================

def create_feature_set_A(df):

    features = [
        'TransactionAmount (INR)',
        'TransactionHour',
        'DayOfWeek',
        'IsWeekend',
        'CustAccountBalance',
        'AmountBalanceRatio'
    ]

    return df[features].copy()


# ==================================================
# 5. EXTENDED FEATURE SET
# ==================================================

def create_feature_set_B(df):

    features = [
        'TransactionAmount (INR)',
        'TransactionHour',
        'DayOfWeek',
        'IsWeekend',
        'CustAccountBalance',
        'AmountBalanceRatio',
        'HistoricalAvgAmount',
        'HistoricalStdAmount',
        'PreviousTransactionCount',
        'HistoricalAmountDeviation',
        'HistoricalAmountZScore'
    ]

    return df[features].copy()

def prepare_feature_set_A(df):
    df = create_datetime_features(df)
    df = create_transaction_features(df)

    X_A = create_feature_set_A(df)

    return df, X_A