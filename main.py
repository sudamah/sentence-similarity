import pandas as pd
import numpy as np
from scipy import spatial
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

def get_embeddings():
    df = pd.read_csv('lyrics.csv')
    df = df.assign(embeddings=df['Lyric'].apply(lambda x: model.encode(str(x))))
    print(df)
    return df

def closest_lyrics(inp):
    data = get_embeddings()
    inp_vector = model.encode(inp)
    s = data['embeddings'].apply(
        lambda x: 1 - spatial.distance.cosine(x, inp_vector))
    data = data.assign(similarity=s)
    return (data.sort_values('similarity', ascending=False))



if __name__ == '__main__':

    print(closest_lyrics('thinking about you'))
