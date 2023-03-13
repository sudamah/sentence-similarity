import csv
import os
import json

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


def load_data(filepath):
    """
    It takes a filepath as an argument, checks the file extension, and loads the file into a Pandas
    DataFrame

    :param filepath: The path to the file you want to load
    :return: A dataframe
    """
    try:
        # Checking if the filepath ends with .csv and if it does, it loads the file into a dataframe.
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)

        # Checking if the filepath ends with .xlsx and if it does, it loads the file into a dataframe.
        elif filepath.endswith(".xlsx"):
            df = pd.read_excel(filepath)

        # Reading the text file and loading it into a dataframe.
        elif filepath.endswith(".txt"):
            df = pd.read_csv(filepath, sep=',', header=None)

        # Loading the json file into a dataframe.
        elif filepath.endswith(".json"):
            with open(filepath, 'r') as file:
                data = json.load(file)
            df = pd.DataFrame(data)

        else:
            return None

        return df

    except:
        return "Something went wrong"


def similarity_score(sent_list, model, tokenizer):
    """
    It takes a list of sentences, tokenizes them, and then calculates the cosine similarity between each
    sentence

    :param sent_list: list of sentences
    :return: A similarity matrix
    """

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


def sentence_mapping(score, sentences, threshold=0.7):
    """
    It takes the score matrix and the sentences and returns a dictionary with the key as the sentence
    number and the value as a list of sentences that are similar to the key sentence

    :param score: the similarity matrix
    :param sentences: list of sentences
    :return: A dictionary with the keys being the index of the sentence and the values being a list of
    sentences that are similar to the key sentence.
    """

    rows = np.argwhere(score > threshold)
    temp_dict = {}
    for i in rows:
        if i[0] not in temp_dict.keys():
            temp_dict[i[0]] = []
        if i[0] == i[1]:
            temp_dict[i[0]].append(sentences[i[0]])
            continue
        temp_dict[i[0]].append(sentences[i[1]])
    return temp_dict


# A way to tell the interpreter that the code in the block should only be executed if the file is run
# as the main program.
if __name__ == '__main__':

    # The path to the csv file that contains the sentences.
    filename = '/home/heptagon/Desktop/sentence-similarity/information.csv'     # csv file name

    # Loading the pretrained model.
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained(
        'sentence-transformers/bert-base-nli-mean-tokens')

    # Checking if the file is valid or not.
    data = load_data(filename)
    if data is None:
        print("pass the valid file")

    # Taking the column 'Occupation' from the dataframe and converting it into a list.
    sentences = data['Occupation'].to_list()
    score = similarity_score(sentences, model, tokenizer)

    # Printing the sentence mapping.
    print(sentence_mapping(score, sentences))
