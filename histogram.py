import random
import re


class Histogram:
    sep = {}
    train_set = {}
    test_set = {}

    def load_data(self, filename):
        with open(filename, "r") as f:
            text = f.readlines()
        _lines = [line.rstrip("\n") for line in text]
        _lines = self.remove_unwanted(_lines)
        _lines = self.switch_simple_case(_lines)
        return _lines

    @staticmethod
    def remove_unwanted(_lines):
        _lines = list(filter(None, _lines))
        _lines = [re.sub('[^A-Za-z: ]+', '', line) for line in _lines]
        return _lines

    @staticmethod
    def switch_simple_case(_lines):
        for i in range(len(_lines)):
            _lines[i] = _lines[i].lower()
        return _lines

    @staticmethod
    def separate_by_classes(_lines, separater):
        separated = {}
        for l in _lines:
            data = l.split(separater)
            if data[0] not in separated:
                separated[data[0]] = []
            separated[data[0]].append(data[1].split())
        return separated

    def get_seperated(self):
        return self.sep

    def get_num_classes(self):
        return len(self.sep.keys())

    def get_train_set(self):
        return self.train_set

    def get_histogram(self):
        histo = {}
        for class_name, lines in self.sep.items():
            histo[class_name] = {}
            r = histo[class_name]
            for line in lines:
                for word in line:
                    if word not in r:
                        r[word] = 1
                    else:
                        r[word] += 1
        return histo

    def get_statisics_data(self):
        histo = self.get_histogram()

        for cls, words in histo.items():
            total_elements = float(len(words))
            for word, count in words.items():
                words[word] = float(count) / total_elements

        return histo

    @staticmethod
    def split_data(data_set, split_ratio):
        train_set_size = int(len(data_set) * split_ratio)
        train_set = []
        original = data_set
        while len(train_set) < train_set_size:
            idx = random.randrange(0, len(original))
            train_set.append(original.pop(idx))

        return [train_set, original]

    def get_test_set(self):
        return self.test_set

    def __init__(self, filename="dialog.txt", separater=":", split_ratio=0.67):
        lines = self.load_data(filename)
        self.train_set, self.test_set = self.split_data(lines, split_ratio)
        self.sep = self.separate_by_classes(self.train_set, separater)
