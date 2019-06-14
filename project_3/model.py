import torch as tor
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence



class CategoryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cate_size, fc_size=2**9, dropout=0.5):
        super(CategoryClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        
        self.fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, fc_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(fc_size, fc_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(fc_size, cate_size),
        )


    def forward(self, x):
        # print(x)
        x, (h_n, c_n) = self.lstm(x)
        x, x_len = pad_packed_sequence(x, batch_first=True)
        gather_idx = (x_len - 1).unsqueeze(1).unsqueeze(2).repeat(1, 1, x.size(2))
        x = tor.gather(x, dim=1, index=gather_idx).squeeze(1)
        # print(x)
        o = self.fc(x)

        return o