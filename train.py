import torch
import argparse
from torch import optim
from DataProcessor import DataProcessor
from model import BiLSTM_CRF
from utils import *


def train(embedding_dim, hidden_dim, epochs, device):
    processor = DataProcessor(".\\data\\data_renmin.txt_utf8")
    content = processor.get_words()
    label = processor.get_tags()

    train_data = []
    for i in range(len(label)):
        train_data.append((content[i], label[i]))

    word2idx = {}
    for sentence, tags in train_data:
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    model = BiLSTM_CRF(len(word2idx), processor.word2idx_map, embedding_dim, hidden_dim, device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for sentence, tags in train_data:
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word2idx, device=device)
            targets = torch.tensor([processor.word2idx_map[i] for i in tags], dtype=torch.long)
            loss = model.neg_log_likelihood(sentence_in, targets)

            loss.backward()
            optimizer.step()
            print('epoch/epochs: {}/{}, loss: {:.6f}'.format(epoch + 1, epochs, loss.data[0]))

    torch.save(model, './checkpoints/cws.model')
    torch.save(model.state_dict(), './checkpoints/cws_params.model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",type=int, default=100)
    parser.add_argument("--embed_dim", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()

    train(args.embed_dim, args.hidden_dim, args.epoch, args.device)


