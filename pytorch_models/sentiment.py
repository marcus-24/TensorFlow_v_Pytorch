import os
import pandas as pd
import torch
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Callable
import re
import string
import itertools
from tqdm import tqdm
import numpy as np

from customtorchutils import get_workers


EPOCHS = 1000
BATCH_SIZE = 64
VOCAB_SIZE = 1000
MAX_LEN = 30
EMBEDDING_DIM = 16
N_CLASSES = 3

def vocab_generator(sentences: List[str]) -> List[str]:
    for sentence in sentences:
        words = [word for word in sentence.split(' ') 
                 if word and word not in string.punctuation]
        for word in words:
            clean_word = re.sub(f'[{string.punctuation}]', "", word).strip().lower()
            yield [clean_word]


class SentimentDataset(Dataset):

    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 transformer: transforms.Sequential,
                 vocab_gen: Callable[List[str], List[str]]=vocab_generator):
        self._texts = texts
        self._labels = labels
        self._transformer = transformer
        self._vocab_gen = vocab_gen

    def __len__(self) -> int:
        return len(self._texts)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        sentence_gen = self._vocab_gen([self._texts[index]])
        parsed_words = list(itertools.chain.from_iterable(sentence_gen))
        transformed_words = self._transformer(parsed_words)
        tensor_label = torch.tensor(self._labels[index])
        return transformed_words, tensor_label
    
class SentimentModel(nn.Module):
        def __init__(self):
            super().__init__()
            # TODO: Make another class to define layer parameters
            self._embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx=0)
            self._lstm = nn.LSTM(EMBEDDING_DIM, 
                             hidden_size=EMBEDDING_DIM, 
                             num_layers=1, 
                             batch_first=True, # batch_first sets input dim-> (batch_size, window, input_size)
                             bidirectional=True)
            self._dropout = nn.Dropout(0.3)
            self._fc1 = nn.Linear(in_features=2 * EMBEDDING_DIM, out_features=10)
            self._fc2 = nn.Linear(in_features=10, out_features=N_CLASSES)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self._embedding(x)
            x, _ = self._lstm(x)
            x = x[:, -1, :] # the last output for each hidden feature in each batch
            x = self._dropout(x)
            x = F.relu(self._fc1(x))
            x = self._fc2(x) # remember loss function does softmax for you
            return x

if __name__ == "__main__":
    device, num_workers = get_workers()

    '''Load data'''
    fname = os.path.join('data', 'Tweets.csv')
    df = pd.read_csv(fname)

    '''Create feature and label arrays'''
    tweets = df['text'].to_list()
    labels = df['airline_sentiment'].to_list()
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    labels_int = [label_map[label] for label in labels]  # convert to numpy for network compatibility

    '''Split data'''
    training_size = int(0.7 * len(tweets))

    training_tweets = tweets[:training_size]
    testing_tweets = tweets[training_size:]
    training_labels = labels_int[:training_size]
    testing_labels = labels_int[training_size:]

    '''Create corpus based on vocab seen in tweets'''
    # reference: https://pytorch.org/tutorials/beginner/torchtext_custom_dataset_tutorial.html
    source_vocab = build_vocab_from_iterator(vocab_generator(training_tweets),
                                            min_freq=10, # skip words that show up less than 2 times 
                                            specials=['<pad>', # <pad> is the padding token.
                                                    '<unk>'], # <unk> for unknown words. An example of unknown word is the one skipped because of min_freq=2.
                                            special_first=True, # we set special_first=True. Which means <pad> will get index 0 and <unk> will get index 1 in the vocabulary.
                                            max_tokens=VOCAB_SIZE -1) # set to max_occ -1 like tensorflow
    source_vocab.set_default_index(source_vocab['<unk>'])  # we set default index as index of <unk>. That means if some word is not in vocabulary, we will use <unk> instead of that unknown word.

    '''Define transformer functions'''
    text_transform = transforms.Sequential(
        transforms.VocabTransform(vocab=source_vocab),  # transform words to int using twitter vocab fit above
        transforms.Truncate(max_seq_len=MAX_LEN), # truncate to max length
        transforms.ToTensor(), 
        transforms.PadTransform(max_length=MAX_LEN, pad_value=0)) # pad values (only takes tensor as input)
 

    '''Create Datasets for Model'''
    train_set = SentimentDataset(texts=training_tweets, 
                                 labels=training_labels, 
                                 transformer=text_transform)
    
    test_set = SentimentDataset(texts=training_tweets, 
                                labels=training_labels, 
                                transformer=text_transform)
    
    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=num_workers,
                                        generator=torch.Generator(device=device))
    
    testloader = DataLoader(test_set, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=num_workers,
                                    generator=torch.Generator(device=device))
    
    '''Define network'''
    net = SentimentModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    
    n_train_batches = len(trainloader)
    n_val_batches = len(testloader)
    early_stop_thresh = 10
    best_loss = 1e100
    best_epoch = -1

    # TODO: Break up script below into functions
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss, running_loss_val, correct, correct_val, total_samples, total_samples_val = np.zeros(6)
        for i, data in tqdm(enumerate(trainloader), desc=f'Epoch: {epoch + 1}/{EPOCHS}', total=n_train_batches):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad() # resets the gradients for new batch

            # forward + backward + optimize
            outputs = net(inputs) # predicted output
            loss = criterion(outputs, labels) # calulate loss for batch
            loss.backward() # perform backprogation to calculate gradients
            optimizer.step() # gradient descent - update network weights and biases

            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = running_loss / n_train_batches
        avg_acc = 100 * correct / total_samples  # TODO: Double check math
        print(f'Average loss={avg_loss:.4f}  Average accuracy={avg_acc:.3f}%', end="\t")

        net.eval() # activate testing mode
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss_val = criterion(outputs, labels)
            running_loss_val += loss_val.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_val += (predicted == labels).sum().item()
            total_samples_val += labels.size(0)
        
        avg_loss_val = running_loss_val / n_val_batches
        avg_acc_val = 100 * correct_val / total_samples_val  # TODO: Double check math
        print(f'Average val loss={avg_loss_val:.4f}  Average val accuracy={avg_acc_val:.3f}%')

        if avg_loss_val < best_loss:
            best_loss = avg_loss_val
            best_epoch = epoch
            torch.save(net.state_dict(), os.path.join('saved_models', 'torch_best_sentiment_model.pt'))

        elif epoch - best_epoch > early_stop_thresh:
            print("Early stopped training at epoch ", epoch)
            break  # terminate the training loop



    print('Finished Training')
