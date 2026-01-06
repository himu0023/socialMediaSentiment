import pandas as pd 
from preprocessing.text_cleaner import clean_text

path = 'data/twitter_training.csv'

def load_dataset(path):
    df = pd.read_csv(path,
                     header=None,
                     names = ['id','entity','sentiment','text'],
                     quotechar='"',
                     encoding = "utf-8")


    # Keep only sentiment and text 
    df= df[['sentiment', 'text']].dropna()

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