import torch

START_TAG = '<START>'
STOP_TAG = '<STOP>'


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.word2idx_map = {'B': 0, 'M': 1, 'E': 2, 'S': 3, '<START>': 4, '<STOP>': 5}
        self.sentences = []
        self.tags = []
        self.idxs = []
        self.words = []

        self._read_file_()
        self._get_tags_()
        self._word2idx_()

    def _read_file_(self):
        with open(self.file_path, "r", encoding='utf-8') as f:
            for line in f.readlines():
                self.sentences.append(line.strip().split(' '))  # pku 的split要用两个空格
                self.words.append([c for c in line.strip() if c != ' '])

    def _get_tags_(self):
        for line in self.sentences:
            tmp_tag = []
            for word in line:
                if len(word) == 1:
                    tmp_tag.append('S')
                elif len(word) == 2:
                    tmp_tag.append('B')
                    tmp_tag.append('E')
                else:
                    tmp_tag.append('B')
                    tmp_tag.extend('M' * (len(word) - 2))
                    tmp_tag.append('E')
            self.tags.append(tmp_tag)

    def _word2idx_(self):
        for sentence in self.tags:
            idx = [self.word2idx_map[w] for w in sentence]
            self.idxs.append(torch.tensor(idx, dtype=torch.long))

    def get_words(self):
        return self.words

    def get_sentences(self):
        return self.sentences

    def get_tags(self):
        return self.tags

    def get_indexes(self):
        return self.idxs


# processor = DataProcessor('../data/data_renmin.txt_utf8')
# print(processor.get_words())
# print(processor.get_tags())
