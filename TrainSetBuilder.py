import torch
import pandas
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertTokenizer
import pickle


class TrainSetBuilder():

    def __init__(self):

        # Hyper parameters
        self.max_len = 300      # Maximum number of words in a sequence
        self.train_split = 0.8  # Propotion of the dataset for training set
        self.batch_size = 10    # Number of elements in a batch

        # Training dataset
        self.train_dataset = None
        # Testing dataset
        self.test_dataset = None
        # Classes
        self.class_labels = ['Negative', 'Positive']

        # Data handlers
        self.train_handler = None
        self.test_handler = None

        # Load the tokenizer
        self.tokenizer = CamembertTokenizer.from_pretrained(
            'camembert-base',
            do_lower_case=True
        )

    def import_allocine_data(self, path='Train_data/allocine_dataset.pickle.',
                             reduce=None):
        # Unpickle data
        with open(path, 'rb') as reader:
            data = pickle.load(reader)
        # Import data
        print('Get data array...')
        train_rev = data["train_set"]['review'].to_numpy()
        val_rev = data["val_set"]['review'].to_numpy()
        test_rev = data["test_set"]['review'].to_numpy()
        # import labels
        train_labels = data["train_set"]['polarity'].to_numpy()
        val_labels = data["val_set"]['polarity'].to_numpy()
        test_labels = data["test_set"]['polarity'].to_numpy()
        class_names = data['class_names']
        print('...Done')
        del data
        # Concat data
        reviews = np.concatenate([train_rev, val_rev, test_rev])
        # Concat labels
        sentiments = np.concatenate([train_labels, val_labels, test_labels])
        # To list
        reviews = reviews.tolist()

        # If reduce (doesn't load all the dataset
        if reduce is not None:
            print('WARNING: reduced dataset: for deboging purposes only')
            reviews = reviews[0:reduce]
            sentiments = sentiments[0:reduce]

        # Encode the batch of data
        print('Data tokenization...')
        encoded_batch = self.tokenizer.batch_encode_plus(reviews,
                                                         add_special_tokens=True,
                                                         max_length=self.max_len,
                                                         padding=True,
                                                         truncation=True,
                                                         return_attention_mask=True,
                                                         return_tensors='pt')
        print('... Done')
        # Get the spliting index
        split_border = int(len(sentiments)*self.train_split)
        # Get a tensor for sentiments
        sentiments = torch.tensor(sentiments)
        # Now encode datasets tensors
        print('Tensors encoding...')
        self.train_dataset = TensorDataset(
            encoded_batch['input_ids'][:split_border],
            encoded_batch['attention_mask'][:split_border],
            sentiments[:split_border])
        self.test_dataset = TensorDataset(
            encoded_batch['input_ids'][split_border:],
            encoded_batch['attention_mask'][split_border:],
            sentiments[split_border:])
        print('... Done')

        # Get data handler
        print('Data handler encoding...')
        self.train_handler = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.batch_size)

        self.test_handler = DataLoader(
            self.test_dataset,
            sampler=SequentialSampler(self.test_dataset),
            batch_size=self.batch_size)
        print('... Done')
        print('End of dataset encoding.')



