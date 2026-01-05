import pandas as pd 
from preprocessing.text_cleaner import clean_text 

text_col = 3
label_col = 2

path = 'data/dataset.csv'

def load_dataset(path):
    df = pd.read_csv(path, header=None)

    # Select only setiment + text 
    df = df[[label_col, text_col]].dropna()

    # Rename for consistency 
    df.columns = ['sentiment', 'text']

    # Normalize text 
    df['text'] = df['text'].astype(str).apply(clean_text)

    # Normalize sentiment labels
    df['sentiment'] = df['sentiment'].astype(str).str.lower()

    label_map = {
        'positive':'positive',
        'negative':'negative',
        'netural':'neutral'
    }

    df['sentiment'] = df['sentiment'].map(label_map)
    df = df.dropna()

    return df

if __name__ == "__main__":
    df = load_dataset(path)
    print(df.head())
    print(df["sentiment"].value_counts())