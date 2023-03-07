import csv

import pandas as pd
import numpy as np
from scipy import spatial
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')


def get_embeddings(filename):
    with open(filename) as csv_file:
        # read the csv file
        csv_reader = csv.reader(csv_file)

    # now we can use this csv files into the pandas
    df = pd.DataFrame([csv_reader], index=None)

    df_embedding = df.assign(embeddings=df['Lyric'].apply(
        lambda x: model.encode(str(x))))
    print(df_embedding)
    return df_embedding


def get_similarity_score(inp, filename):
    data = get_embeddings(filename)
    inp_vector = model.encode(inp)
    s = data['embeddings'].apply(
        lambda x: 1 - spatial.distance.cosine(x, inp_vector))
    data = data.assign(similarity=s)
    return (data.sort_values('similarity', ascending=False))


if __name__ == '__main__':

    filename = 'lyrics.csv'     # csv file name

    print(get_similarity_score('thinking about you', filename))
