import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer, AdamW
import os
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertTokenizer


def tweet_file_spliter(path='Tweets_data/tweetDownloadBE.csv'):

    # We have to split the dataset in subset of 100 000 tweets
    reader = open(path, 'r', encoding='Latin-1')
    # Read all lines
    lines = reader.readlines()

    sub_file_size = 100000
    file_idx = 0
    total_idx = 0
    first = True
    # Open the new file to write
    index = str(file_idx)
    if len(index < 2):
        index = '0{}'.format(index)
    file = open('Tweets_data/Subsets/sub_tweet_{}.csv'.format(index), 'a', encoding='Latin-1')
    # Add headers
    headers = ['idx', 'date', 'text', 'retweets', 'likes', '\n']
    headers = '\t'.join(headers)
    file.write(headers)
    # Store skipped lines
    skipped = 0
    for line in tqdm(lines):
        if total_idx % sub_file_size == 0 and total_idx != 0:
            # Write in a new file
            file.close()
            file_idx += 1
            file = open('Tweets_data/Subsets/sub_tweet_{}.csv'.format(file_idx), 'a', encoding='Latin-1')
            file.write(headers)
        # Avoid headers
        if first:
            first = False
            continue
        line = line.split('\t')
        if len(line) < 4:
            skipped += 1
            continue
        try:
            writer = [str(total_idx),
                      str(line[0]),
                      str(line[1]),
                      str(line[2]),
                      str(line[3]),
                      '\n']
            file.write('\t'.join(writer))
            total_idx += 1
        except:
            skipped += 1
    file.close()
    reader.close()

    print('Skipped lines: {}'.format(skipped))



