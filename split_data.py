import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # Read file with all data:
    df = pd.read_csv(os.path.join(os.getcwd(), 'data/data.csv'))

    # Sort parties alphabetically:
    parties = sorted(df['Party'].unique().tolist())
    
    # Give Schoof the final index
    parties.remove('Schoof')
    parties.append('Schoof')

    # Give parties an index:
    df['Label'] = df['Party'].apply(lambda x: parties.index(x))

    # Take all Schoof data and save:
    df_schoof = df[df['Label'] == len(parties)-1]
    df_schoof.to_csv(path_or_buf=os.path.join(os.getcwd(), "data/schoof.csv"), index=False)

    # Split all data in training, validation and test set:
    df = df[df['Label'] != len(parties)-1]

    # 80% train, 10% validation, 10% test
    train, val_test = train_test_split(df, train_size=0.8, random_state=42, shuffle=True, stratify=df['Label'])
    val, test = train_test_split(val_test, train_size=0.5, random_state=42, shuffle=True, stratify=val_test['Label'])

    # Save to .csv:
    train.to_csv(path_or_buf=os.path.join(os.getcwd(), "data/train.csv"), index=False)
    val.to_csv(path_or_buf=os.path.join(os.getcwd(), "data/val.csv"), index=False)
    test.to_csv(path_or_buf=os.path.join(os.getcwd(), "data/test.csv"), index=False)