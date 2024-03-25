import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import torch.optim as optim
from tqdm import tqdm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from customtorchutils import get_workers

DATA_COLS = ["Open", "High", "Low", "Close"]
BATCH_SIZE = 32
WIN_SIZE = 10
EPOCHS = 1000
N_FEATURES = len(DATA_COLS)

class TimeSeriesDataset(Dataset):

    def __init__(self, 
                 data: np.ndarray,
                 window_size: int, 
                 target_idx: int):
        self._data = np.float32(data)  # needs to be float32 to feed into network
        self._window_size = window_size
        self._target_idx = target_idx

    def __len__(self) -> int:
        return self._data.shape[0] - self._window_size # handles tf drop_remainder arg
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self._data[index:index + self._window_size, :]
        # target = self._data[index + 1:index + self._window_size + 1, self._target_idx] for seq to seq
        target = self._data[index + self._window_size, self._target_idx]
        return torch.tensor(feature), torch.tensor(target)


class LSTMNet(nn.Module):
    def __init__(self, 
                 input_size: int, # aka: number of features
                 hidden_size: int,
                 num_layers: int,
                 num_classes: int,
                 bidirectional: bool = True):
        super().__init__()
        self._num_layers = num_layers  # number of LSTM layers to stack together
        self._lstm = nn.LSTM(input_size, 
                             hidden_size, 
                             num_layers, 
                             batch_first=True, # batch_first sets input dim-> (batch_size, window, input_size)
                             bidirectional=bidirectional)
        lstm_output_size = 2 * hidden_size if bidirectional else hidden_size # number of features in hidden state (sets output size)
        self._fc = nn.Linear(lstm_output_size, num_classes) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self._lstm(x)  # out dim -> (batch_size, window, hidden_size)
        out = out[:, -1, :] # the last output for each hidden feature in each batch
        out = self._fc(out)

        return out


if __name__ == "__main__":
    device, num_workers = get_workers()

    '''Load stock data'''
    data = yf.download("AAPL", start="2020-01-01", end="2024-01-30")

    '''Split Data'''
    split_idx = int(0.75 * data.shape[0])
    split_time = data.index[split_idx]

    x_train = data.loc[:split_time, DATA_COLS]
    train_time = x_train.index.to_numpy()
    x_val = data.loc[split_time:, DATA_COLS]
    val_time = x_val.index.to_numpy()

    '''Normalize data'''
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    train_set = TimeSeriesDataset(x_train,
                                window_size=WIN_SIZE,
                                target_idx=-1)
    val_set = TimeSeriesDataset(x_val,
                                window_size=WIN_SIZE,
                                target_idx=-1)


    train_loader = DataLoader(train_set,
                            batch_size=BATCH_SIZE,
                            generator=torch.Generator(device=device))

    val_loader = DataLoader(val_set,
                            batch_size=BATCH_SIZE,
                            generator=torch.Generator(device=device))

    net = LSTMNet(N_FEATURES,
                  hidden_size=30,
                  num_layers=2,
                  num_classes=1)
    
    print('model summary: ', net)
    
    criterion = nn.HuberLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    early_stop_thresh = 10
    best_loss = 1e100
    best_epoch = -1

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss, running_loss_val, abs_errors, abs_errors_val, total_samples, total_samples_val = np.zeros(6)

        net.train() # activate training model 
        for data in tqdm(train_loader, desc=f'Epoch: {epoch + 1}/{EPOCHS}', total=n_train_batches):
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
            abs_errors += (outputs - labels).abs().sum().item()
            total_samples += labels.size(0)

        avg_loss = running_loss / n_train_batches
        mae = abs_errors / total_samples  # TODO: Double check math
        print(f'Average loss={avg_loss} \t MAE={mae}', end =" ")

        net.eval() # activate testing mode
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            predicted = net(inputs)
            abs_errors_val += (outputs - labels).abs().sum().item()
            total_samples_val += labels.size(0)
        
        avg_loss_val = running_loss / n_val_batches
        mae_val = abs_errors_val / total_samples_val  # TODO: Double check math
        print(f'Average val loss={avg_loss_val} \t Val MAE={mae_val}')

        if avg_loss_val < best_loss:
            best_loss = avg_loss_val
            best_epoch = epoch
            torch.save(net.state_dict(), os.path.join('saved_models', 'torch_best_forecast_model.pt'))

        elif epoch - best_epoch > early_stop_thresh:
            print("Early stopped training at epoch ", epoch)
            break  # terminate the training loop



    print('Finished Training')
