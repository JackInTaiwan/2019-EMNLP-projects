import torch as tor
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence



class CategoryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cate_size, fc_size=2**9, dropout=0.5):
        super(CategoryClassifier, self).__init__()

        # self.lstm = nn.LSTM(
        #     input_size=input_size,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     dropout=dropout,
        #     batch_first=True,
        #     bias=False,
        # )

        self.lstm = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
            bias=False,
        )
        
        self.fc = nn.Sequential(
            # # nn.Tanh(),
            # nn.Sigmoid(),
            nn.Linear(hidden_size * 2, fc_size, bias=False),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Dropout(p=dropout),
            # nn.Sigmoid(),
            # nn.Linear(fc_size, fc_size, bias=False),
            # nn.Dropout(p=dropout),
            # nn.ReLU(),
            # nn.Sigmoid(),
            nn.Linear(fc_size, cate_size, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        # print(x)
        # x, (h_n, c_n) = self.lstm(x)
        x, h_n = self.lstm(x)
        x, x_len = pad_packed_sequence(x, batch_first=True)
        x = tor.mean(x, dim=1).squeeze(1) * 100
        # gather_idx = (x_len - 1).unsqueeze(1).unsqueeze(2).repeat(1, 1, x.size(2))
        # x = tor.gather(x, dim=1, index=gather_idx).squeeze(1)
        # if not self.training: print(x)
        o = self.fc(x)
        # if not self.training: print(o)

        return o