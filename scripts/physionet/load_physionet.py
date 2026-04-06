import pandas as pd
import os

DATA_DIR = r"C:\Users\Rohil\physionet.org\files\challenge-2019\1.0.0\training\training_setA"

def load_physionet(data_dir):
    dfs = []
    
    for fname in os.listdir(data_dir):
        if not fname.endswith('.psv'):
            continue
        
        filepath = os.path.join(data_dir, fname)
        df = pd.read_csv(filepath, sep='|')
        df['patient_id'] = fname.replace('.psv', '')
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined

def rename_columns(df):
    rename_map = {
        'HR'         : 'heart_rate',
        'Resp'       : 'resp_rate',
        'Temp'       : 'temperature',
        'SBP'        : 'sbp',
        'DBP'        : 'dbp',
        'O2Sat'      : 'spo2',
        'WBC'        : 'wbc',
        'Lactate'    : 'lactate',
        'SepsisLabel': 'sepsis_label',
    }
    return df.rename(columns=rename_map)

def keep_relevant_columns(df):
    cols = ['patient_id', 'heart_rate', 'resp_rate', 'temperature',
            'sbp', 'dbp', 'spo2', 'wbc', 'lactate', 'sepsis_label']
    return df[cols]

if __name__ == "__main__":
    print("Loading files...")
    df = load_physionet(DATA_DIR)
    print(f"Loaded {df['patient_id'].nunique()} patients")
    
    df = rename_columns(df)
    df = keep_relevant_columns(df)
    
    df.to_csv(r"C:\Users\Rohil\physionet.org\files\challenge-2019\1.0.0\training\training_setA\physionet_raw.csv", index=False)
    print("Saved to physionet_raw.csv")
    print(df.head())