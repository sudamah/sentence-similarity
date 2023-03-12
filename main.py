import csv

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


def load_data(filename):
    # read the csv file
    df = pd.read_csv(filename)
    return df


def sentence_list(df):
    sentences = []
    for key_str in tqdm(df.Occupation):
        sentences.append(key_str)

    return sentences


def similarity_score(sent_list):

    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained(
        'sentence-transformers/bert-base-nli-mean-tokens')

    # initialize dictionary that will contain tokenized sentences
    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in sent_list:
        # tokenize sentence and append to dictionary lists
        new_tokens = tokenizer.encode_plus(sentence,
                                           max_length=128,
                                           truncation=True,
                                           padding='max_length',
                                           return_tensors='pt'
                                           )

        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

        # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    # convert from PyTorch tensor to numpy array
    mean_pooled = mean_pooled.detach().numpy()
    similarity_matrix = cosine_similarity(
        mean_pooled, mean_pooled)  # calculate cosine_similarity
    return similarity_matrix


def sentence_mapping(score, sentences):
    rows = np.argwhere(score > .7)
    temp_dict = {}
    for i in rows:
        if i[0] not in temp_dict.keys():
            temp_dict[i[0]] = []
        if i[0] == i[1]:
            temp_dict[i[0]].append(sentences[i[0]])
            continue
        temp_dict[i[0]].append(sentences[i[1]])
    return temp_dict


if __name__ == '__main__':

    filename = 'information.csv'     # csv file name
    data = load_data(filename)

    sentences = sentence_list(data)
    score = similarity_score(sentences)

    print(sentence_mapping(score, sentences))
