import torch
import torch.nn as nn
from utils import *


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim, device):
        super(BiLSTM_CRF, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tag_set_size = len(tag2id)
        self.device = device

        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # LSTM Outputs -> Tag Space
        self.hidden2tag = nn.Linear(hidden_dim, self.tag_set_size)

        self.transitions = nn.Parameter(torch.randn(self.tag_set_size, self.tag_set_size))

        self.transitions.data[tag2id[self.START_TAG], :] = -10000
        self.transitions.data[:, tag2id[self.STOP_TAG]] = -10000

        self.hidden = self._init_hidden_()

    def _init_hidden_(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(self.device),
                torch.randn(2, 1, self.hidden_dim // 2).to(self.device))

    def _forward_alg(self, features):
        init_alphas = torch.full((1, self.tag_set_size), -10000).to(self.device)
        init_alphas[0][self.tag2id[self.START_TAG]] = 0

        forward_var = init_alphas

        for feature in features:
            alpha_t = []
            for next_tag in range(self.tag_set_size):
                emit_score = feature[next_tag].view(1, -1).expand(1, self.tag_set_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alpha_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alpha_t).view(1, -1).to(self.device)
        terminal_var = forward_var + self.transitions[self.tag2id[self.STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self._init_hidden_()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_features = self.hidden2tag(lstm_out)
        return lstm_features

    def _score_sentence(self, features, tags):
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.tag2id[self.START_TAG]], dtype=torch.long).to(self.device), tags]).to(
            self.device)
        for i, feature in enumerate(features):
            score = score + self.transitions[tags[i + 1], tags[i]] + feature[tags[i + 1]]
        score += self.transitions[self.tag2id[self.STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, features):
        backpointers = []
        init_vars = torch.full((1, self.tag_set_size), -10000).to(self.device)
        init_vars[0][self.tag2id[self.START_TAG]] = 0

        forward_var = init_vars

        for feature in features:
            bptrs_t = []
            viterbi_vars_t = []
            for next_tag in range(self.tag_set_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbi_vars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbi_vars_t) + feature).to(self.device).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag2id[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag2id[self.START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence, tags):
        features = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(features)
        gold_score = self._score_sentence(features, tags)
        return forward_score - gold_score

    def test(self, sentence):
        lstm_features = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_features)
        return score, tag_seq
