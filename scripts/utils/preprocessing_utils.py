import pandas as pd

def forward_fill(df):
    return df.fillna(method='ffill')

def median_fill(df):
    return df.fillna(df.median(numeric_only=True))