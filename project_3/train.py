import nltk
import torch as tor
import torch.nn as nn
import numpy as np
from utils import word_voc, produce_labels
from utils import SequenceDataset
from model import CategoryClassifier
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence
from torch.utils.data import DataLoader



np.random.seed(123)
UNKNOWN = np.random.randn(300)



def batch_to_packed_sequence(word_table, sentences):
    vec_dim = 300
    s_vec_list, s_len_list = [], []

    for sentence in sentences:
        s_vec = []
        for word in nltk.word_tokenize(sentence, language='english'):
            try:
                v_vec = word_table[word]
            except:
                v_vec = UNKNOWN
            s_vec.append(v_vec)
        s_len_list.append(len(s_vec))
        s_vec_list.append(tor.tensor(s_vec, dtype=tor.float))

    s_vec_list, s_len_list = zip(*sorted(zip(s_vec_list, s_len_list), reverse=True, key=lambda x: x[1]))
    s_vec_list = pack_padded_sequence(pad_sequence(list(s_vec_list), batch_first=True), s_len_list, batch_first=True)

    return s_vec_list



def calculate_valid_acc(x_valid, y_valid, model, word_table):
    batch_size = 2 ** 8
    dataset = SequenceDataset(x_valid, y_valid)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=2,
    )
    acc_list = []

    model.eval()

    for ((x, idx_entity_s, idx_entity_e), y) in data_loader:
        x_ts, y_ts = batch_to_packed_sequence(word_table, x), y
        pred = model(x_ts)
        acc = tor.mean((tor.max(pred, dim=1)[1] == y_ts).type(tor.float))
        print((tor.max(pred, dim=1)[1]))
        acc_list.append(acc.detach().numpy())
    
    acc_output = np.mean(np.array(acc_list))
    model.train()
    
    return acc_output



def train(gpu, lr_lstm, lr_fc, word_table_limit, hidden_size, fc_size, batch_size):
    # nltk.download('punkt', download_dir='./')
    word_table = word_voc('./cc.en.300.vec', word_table_limit)
    label_list, cate_list = produce_labels('./data.train')
    
    model = CategoryClassifier(
        input_size=300,
        hidden_size=hidden_size,
        num_layers=1,
        fc_size=fc_size,
        cate_size=len(label_list)
    )

    data = np.load('./data.transform.train.npy', allow_pickle=True)
    
    x_train, y_train = data[:, 0: 3], data[:, 4]
    epoch = 100

    dataset = SequenceDataset(x_train, y_train)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )
    loss_func = nn.CrossEntropyLoss()
    optim = tor.optim.SGD([
                {'params': model.lstm.parameters(), 'lr': lr_lstm},
                {'params': model.fc.parameters(), 'lr': lr_fc}
            ])
    lr_scheduler = tor.optim.lr_scheduler.StepLR(
        optim,
        step_size=10,
        gamma=0.5,
    )
    
    if gpu:
        print('|Use GPU')
        loss_func = loss_func.cuda()
        model = model.cuda()

    for epoch_ in range(epoch):
        print('|Epoch:', epoch_)
        for step, ((x, idx_entity_s, idx_entity_e), y) in enumerate(data_loader):
            x_ts = batch_to_packed_sequence(word_table, x) if not gpu else batch_to_packed_sequence(word_table, x).cuda()
            y_ts = y if not gpu else y.cuda()
            optim.zero_grad()
            pred = model(x_ts)

            loss = loss_func(pred, y_ts)
            acc = tor.mean((tor.max(pred, dim=1)[1] == y_ts).type(tor.float))
            loss.backward()
            optim.step()
            if step % 50 == 0:
                print('Loss', loss.detach().numpy(), flush=True)
            
            if step % 1000 == 0:
                optim.zero_grad()
                valid_sample_size = 2000
                acc = calculate_valid_acc(x_train[:valid_sample_size], y_train[:valid_sample_size], model, word_table)
                print('Acc:', acc, flush=True)
        lr_scheduler.step()



if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu')
    parser.add_argument('--lr-1', dest='lr_lstm', type=float, default=1e-3, help='lr for lstm')
    parser.add_argument('--lr-2', dest='lr_fc', type=float, default=1e-4, help='lr for fc')
    parser.add_argument('--word-table', dest='word_table_limit', type=int, default=400000, help='size of required word table')
    parser.add_argument('--hidden-size', type=int, default=2 ** 9, help='hidden size of lstm model')
    parser.add_argument('--fc-size', type=int, default=2 ** 9, help='fc size of fc model')
    parser.add_argument('--batch', dest='batch_size', type=int, default=2 ** 4, help='batch size')
    args = parser.parse_args()

    train(**vars(args))