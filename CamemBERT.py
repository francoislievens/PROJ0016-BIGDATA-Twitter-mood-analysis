import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer, AdamW
import pickle
import os
import matplotlib.pyplot as plt
from math import *
from tqdm import tqdm

class CamemBERT():

    def __init__(self, name='Final_Work',
                 colab=False,  # If we are using google colab
                 reset_if_exist=False,  # If we want to delete an existing model to train a new
                 force_cpu=False,  # Force cpu using while cuda is present
                 externe=True,
                 ):

        # Hyper parameters
        self.learning_rate = 5e-7  # Learning rate for Adam optimizer
        self.epslilon = 1e-8  # Epsilon limit for optimizer
        self.pred_batch_size = 1000  # Batch size for predictions

        # Download pre-trained camembert
        self.model = CamembertForSequenceClassification.from_pretrained(
            'camembert-base',
            num_labels=2  # Binary classification
        )
        # Classes labels
        self.class_labels = ['Negative', 'Positive']

        # The optimizer to use
        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.learning_rate,
                               eps=self.epslilon)

        # Get the tokenizer from BERT pre-trained
        self.tokenizer = CamembertTokenizer.from_pretrained(
            'camembert-base',
            do_lower_case=True
        )
        # A softmax function for predictions
        self.soft_pred = torch.nn.Softmax(dim=1)

        # Device:
        self.device = 'cpu'
        if torch.cuda.is_available() and not force_cpu:
            self.device = 'cuda:0'
        self.model.to(self.device)

        # The actual epoch idx and total tracking index
        self.epoch_idx = 0
        self.total_idx = 0

        # Some paths and name:
        self.name = name
        tmp = ''
        if colab:
            tmp = '/content/gdrive/MyDrive/CamemBERT/'
        if externe:
            tmp = 'D:/CamemBERT/'
        self.model_path = '{}Model/{}/Weights'.format(tmp, name)
        self.optimizer_path = '{}Model/{}/Optimizer'.format(tmp, name)
        self.tracking_path = '{}Model/{}/Tracking'.format(tmp, name)

        # Load model if exist
        gen_idx = 0
        mod_lst = os.listdir(self.model_path)
        if len(mod_lst) > 0:
            idx_lst = []
            for itm in mod_lst:
                idx_lst.append(int(itm.replace('.pt', '')))
            gen_idx = int(max(idx_lst))

        try:
            self.model.load_state_dict(torch.load('{}/{}.pt'.format(self.model_path, gen_idx)))
            print('Existing model successfully loaded')
            self.optimizer.load_state_dict(torch.load('{}/{}'.format(self.optimizer_path, gen_idx)))
            print('Optimizer state successfully loaded')
        except:
            print('WARNING: No Model to restore')
            pass
        gen_idx += 1

        # Get last epoch idx
        try:
            df = pd.read_csv('{}/Total_Tracking.csv'.format(self.tracking_path), sep=';', header=None).to_numpy()
            self.epoch_idx = int(np.max(df[:, 0])) + 1
        except:
            pass

        self.total_idx = gen_idx


    def train(self,
              dataset,  # A TrainSetBuilder element
              nb_epoch=100  # The number of epochs to perform
              ):

        # Get data handlers
        train_dataloader = dataset.train_handler
        test_dataloader = dataset.test_handler

        # Get an iterator for the testing loader
        test_iter = iter(test_dataloader)

        # Get epochs index
        e_start = self.epoch_idx
        e_end = self.epoch_idx + nb_epoch

        # The training loop
        for epoch in range(e_start, e_end):
            tmp_train_loss = []
            # Training mode
            self.model.train()
            epoch_train_loss = []
            epoch_test_loss = []

            for step, batch in enumerate(train_dataloader):
                # Delete cache
                torch.cuda.empty_cache()
                # Set batch's data to gpu
                input_id = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                sentiment = batch[2].to(self.device)
                # Swipe gradients
                self.model.zero_grad()
                # Predict
                prd = self.model(input_id,
                                 token_type_ids=None,
                                 attention_mask=attention_mask,
                                 labels=sentiment)
                loss = prd.loss
                logits = prd.logits
                # Backward and optimization
                loss.backward()
                self.optimizer.step()
                # Store temp loss
                tmp_train_loss.append(loss.cpu().detach().item())
                epoch_train_loss.append(loss.cpu().detach().item())

                # Tracking each 50 batchs
                if step % 50 == 0:
                    tmp_test_loss = []
                    # Compute 10 batchs on the test set
                    self.model.eval()
                    with torch.no_grad():
                        for t in range(0, 10):
                            try:
                                test_batch = next(test_iter)
                            except:
                                test_iter = iter(test_dataloader)
                                test_batch = next(test_iter)
                            # Get data
                            input_id = test_batch[0].to(self.device)
                            attention_mask = test_batch[1].to(self.device)
                            sentiment = test_batch[2].to(self.device)
                            # Predict
                            prd = self.model(input_id,
                                             token_type_ids=None,
                                             attention_mask=attention_mask,
                                             labels=sentiment)
                            test_loss = prd.loss
                            tmp_test_loss.append(test_loss.cpu().detach().item())
                            epoch_test_loss.append(test_loss.cpu().detach().item())

                    # Get the mean loss
                    train_loss = np.mean(tmp_train_loss)
                    tmp_train_loss = []
                    test_loss = np.mean(tmp_test_loss)

                    print('Epoch {}, step {} - Total_idx {} - Train Loss: {} - Test Loss: {}'.format(epoch,
                                                                                        step,
                                                                                        self.total_idx,
                                                                                        train_loss,
                                                                                        test_loss))
                    # Write in file
                    tmp = [str(epoch), str(self.total_idx), str(step), str(train_loss), str(test_loss)]
                    file = open('{}/Total_Tracking.csv'.format(self.tracking_path), 'a')
                    file.write('{}\n'.format(';'.join(tmp)))
                    file.close()

                    # Save parameters
                    torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.model_path, self.total_idx))
                    torch.save(self.optimizer.state_dict(), '{}/{}.pt'.format(self.optimizer_path, self.total_idx))
                    # Update general idx
                    self.total_idx += 1

                    # Delete old models
                    mod_lst = os.listdir(self.model_path)
                    for itm in mod_lst:
                        itm_idx = int(itm.replace('.pt', ''))
                        if itm_idx < self.total_idx and self.total_idx - itm_idx > 5:
                            if itm_idx % 5 != 0:
                                try:
                                    os.remove('{}/{}.pt'.format(self.model_path, itm_idx))
                                    os.remove('{}/{}.pt'.format(self.optimizer_path, itm_idx))
                                except:
                                    print('Error during old model deleting')

            # Epoch loss computing
            ep_train_loss = np.mean(epoch_train_loss)
            ep_test_loss = np.mean(epoch_test_loss)

            file = open('{}/Epoch_Tracking.csv'.format(self.tracking_path), 'a')
            file.write('{};{};{}\n'.format(epoch, ep_train_loss, ep_test_loss))
            file.close()
            print('\t* ===================================== *')
            print('\t* Epoch {} :'.format(epoch))
            print('\t* \t Average Train Loss: {}'.format(ep_train_loss))
            print('\t* \t Average Test Loss: {}'.format(ep_test_loss))


    def annot_file(self, file_path='Tweets_data/Subsets'):

        # Open the file
        df = pd.read_csv(file_path, sep='\t')

        # Tokenize a batch tensor
        tmp = df['text'].tolist()
        input_len = len(tmp)
        for i in range(0, len(tmp)):
            if 'str' not in str(type(tmp[i])):
                tmp[i] = ' '

        encoded_batch = self.tokenizer.batch_encode_plus(tmp,
                                                         add_special_tokens=True,
                                                         max_length=100,
                                                         truncation=True,
                                                         padding=True,
                                                         return_attention_mask=True,
                                                         return_tensors='pt')
        # Build a data tensor in the right format
        data_tensor = TensorDataset(
            encoded_batch['input_ids'],
            encoded_batch['attention_mask'])
        # And the data handler
        handler = DataLoader(
            data_tensor,
            shuffle=False,
            batch_size=self.pred_batch_size)

        # Don't keep gradient graph
        self.model.eval()
        # Store predictions:
        nb_batch = ceil(input_len/self.pred_batch_size)
        total_prd = np.zeros((nb_batch, self.pred_batch_size))
        # Predict on the batch
        i = 0

        for step, pred_batch in enumerate(handler):
            # Get data
            input_id = pred_batch[0].to(self.device)
            attention_mask = pred_batch[1].to(self.device)
            # Predict
            with torch.no_grad():
                self.model.eval()
                prd = self.model(input_id,
                                 token_type_ids=None,
                                 attention_mask=attention_mask)
                prd_logits = self.soft_pred(prd.logits)[:, 1].cpu().detach().numpy()

            # Add to total array:
            total_prd[i, 0:len(prd_logits)] = prd_logits
            i += 1

        annot_column = total_prd.flatten(order='C')
        annot_column = annot_column[0:input_len]
        # Add the new column
        df['Unnamed:5'] = annot_column

        new_path = file_path.replace('.csv', '')

        df.to_csv('{}_annoted.csv'.format(new_path), sep='\t')

    def annot_all(self):

        tmp_lst = os.listdir('Tweets_data/Subsets')

        # Keep only not already annoted files
        path_lst = []
        for i in range(0, len(tmp_lst)):
            if not 'annoted' in tmp_lst[i]:
                annoted = False
                tmp_name = tmp_lst[i].replace('.csv', '')
                for j in range(0, len(tmp_lst)):
                    if i != j and tmp_name in tmp_lst[j]:
                        annoted = True
                if not annoted:
                    path_lst.append(tmp_lst[i])


        for i in tqdm(range(0, len(path_lst))):
            self.annot_file('Tweets_data/Subsets/{}'.format(path_lst[i]))



        pass