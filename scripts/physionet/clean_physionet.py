import pandas as pd

RAW_PATH     = r"C:\Users\Rohil\physionet.org\files\challenge-2019\1.0.0\training\training_setA\physionet_raw.csv"
CLEANED_PATH = r"C:\Users\Rohil\physionet.org\files\challenge-2019\1.0.0\training\training_setA\physionet_cleaned.csv"
LABELS_PATH  = r"C:\Users\Rohil\physionet.org\files\challenge-2019\1.0.0\training\training_setA\physionet_labels.csv"

def clean_physionet(df):
    vital_cols = ['heart_rate', 'resp_rate', 'temperature',
                  'sbp', 'dbp', 'spo2', 'wbc', 'lactate']
    
    # Forward fill within each patient
    df = df.sort_values(['patient_id'])
    df[vital_cols] = (
        df.groupby('patient_id')[vital_cols]
        .transform(lambda x: x.ffill())
    )
    
    # Fill remaining missing with column median
    for col in vital_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df

def make_label_table(df):
    labels = (
        df.groupby('patient_id')['sepsis_label']
        .max()
        .reset_index()
        .rename(columns={'sepsis_label': 'label'})
    )
    return labels

if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)
    
    print("Cleaning...")
    df_clean = clean_physionet(df)
    
    print("Creating label table...")
    labels = make_label_table(df_clean)
    
    df_clean.to_csv(CLEANED_PATH, index=False)
    labels.to_csv(LABELS_PATH, index=False)
    
    print(f"\n--- Summary ---")
    print(f"Total patients : {labels['patient_id'].nunique()}")
    print(f"Sepsis cases   : {labels['label'].sum()}")
    print(f"Sepsis rate    : {labels['label'].mean():.1%}")
    print(f"\nMissing values after cleaning:")
    print(df_clean.isnull().sum())