UNK_token = 0  # Unknown '<unk>'
UNK_symbol = "<unk>"


class Vocab:
    """Class that defines a vocab with methods to move from numeric index to word

    Public Methods:
    get_words
    num_words
    word2index
    index2word
    word2cound
    add_sentence
    add_word
    """

    def __init__(self, name=""):
        """Construction that sets a name for this vocab

        Args:
          name
        """
        self.name = name
        self._word2index = {UNK_symbol: UNK_token}
        self._word2count = {UNK_symbol: 0}
        self._index2word = {UNK_token: UNK_symbol}
        self._n_words = 1

    def get_words(self) -> list[str]:
        return list(self._word2count.keys())

    def num_words(self):
        return self._n_words

    def word2index(self, word):
        if word in self._word2index:
            return self._word2index[word]
        else:
            return self._word2index[UNK_symbol]

    def index2word(self, word):
        return self._index2word[word]

    def word2count(self, word):
        return self._word2count[word]

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        if word not in self._word2index:
            self._word2index[word] = self._n_words
            self._word2count[word] = 1
            self._index2word[self._n_words] = word
            self._n_words += 1
        else:
            self._word2count[word] += 1
