import math

# ignore all warnings
import warnings
import fire
from utils import load_dataset
from constants import ModelNames
import yaml

import gensim
import gensim.downloader
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.data import get_tokenizer

from cbow import CBOW
from skip_gram import SkipGram
from vocab import Vocab

warnings.filterwarnings("ignore")

# TODO: given the size of this file would likely have split it up into multiple files, but
# useful for code review


class Word2Vec:
    """Parent class for CBOW and SKIP_GRAM with shared functionality"""

    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Current device is {self.DEVICE}")

    def get_batch(
        self, data: np.ndarray, index: int, batch_size: int = 10
    ) -> tuple(torch.Tensor, torch.Tensor):
        """A shared function to generate batches of data as tensors on the GPU

        Note: if a regular list of lists is passed in instead of an np.ndarray performance is 10x worse,
        ~21ms vs. 290 ms

        Args:
            data (np.ndarray): numpy array of training pairs
            index (int): the batching index
            batch_size (int): the batch size

        Returns:
            A tensor for the training data x and y placed on the GPU device (ideally)
        """
        data = np.array(data)
        x = data[index : index + batch_size, 0]
        y = data[index : index + batch_size, 1]
        # Citation: how to deal with numpy.object issue
        # https://stackoverflow.com/questions/55724123/typeerror-cant-convert-np-ndarray-of-type-numpy-object
        # This is a shared function between SKIP-GRAM and CBOW.
        # Due to the homework restrictions, the data types passed into this by the training and then
        # unit testing code were widely variable and difficult to use type inspection
        # Decided to use instead forgiveness rather than permission
        try:
            x = np.vstack(x).astype(np.int)
            y = y.astype(np.int)
        except Exception as e:
            x = x.astype(np.int)
            y = np.vstack(y).astype(np.int)

        return torch.from_numpy(x).to(self.DEVICE), torch.from_numpy(y).to(self.DEVICE)

    @staticmethod
    def get_tokenizer(text: str, tokenizer_name: str = "basic_english") -> list[str]:
        """
        Params:
            text (str): The text string to be tokenized
            tokenizer_name (str): The name of the tokenizer

        Returns:
            A list[str] of tokens
        """
        tokenizer = get_tokenizer(tokenizer_name)
        return tokenizer(text)


class CBOW_Wrapper(Word2Vec):
    """A class to instantiate a continuous bag of words (CBOW) model"""

    def __init__(
        self,
    ):
        super().__init__()
        self.model = None
        self.cbow_optimizer = None
        self.cbow_criterion = None

    def prep_cbow_data(
        self,
        data_frame: pd.DataFrame,
        text_col: str,
        tokenizer_fn: callable = None,
        window: int = 2,
        max_length: int = 50,
    ) -> tuple[np.array, Vocab]:
        """Prep CBOW data returns a list of training examples, [(x1, y1), (x2, y2)...]
        where the x consists of the surrounding i +/- window words and the y the ith word

        E.g. For the sentence "The quick brown fox jumped over the lazy dog" we would generate

        x=[the, quick, fox, jumped], y=brown
        x=[quick, brown, jumped, over], y=fox
        x=[brown, fox, over, the], y=jumped
        x=[fox, jumped, the, lazy], y=over
        x=[jumped, over, lazy, dog], y=the

        Args:
            data_frame (pd.DataFrame): A dataframe to tokenize then return training_pairs
            text_col (str): A string which indicates which column in the dataframe has the text
            tokenizer_fn (callable): A function to tokenize input, can either be default or use your own
            window (int): How many words in either direction from the ith word to use as context
            max_length (int): How many pairs of training examples to create from each document

        Returns:
            A numpy array of numpy arrays containing the training examples,
            i.e. np.array(np.array(np.array(x1), np.array(y1))

        """

        # Likely not best practice, but wanted to demonstrate typing a callable function and function as
        # first class citizens in python
        if not tokenizer_fn:
            tokenizer_fn = self.get_tokenizer

        data_out = []
        vocab = Vocab()
        # The homework required us to iterate through dataframe, obviously bad practice
        for text in data_frame[text_col].values:
            words = tokenizer_fn(text)
            vocab.add_sentence(" ".join(words))
            for i in range(window, min(len(words) - window, max_length - window)):
                indices = [
                    vocab.word2index(words[i])
                    for i in range(i - window, i + window + 1)
                ]
                y = indices.pop(window)
                # This awkward construction is due to homework requirements
                # As a side note, interesting to note that native python list vs. np.array
                # results in 10x poorer performance in subsequent function call
                data_out.append((np.array(indices), np.array(y)))

        return np.array(data_out), vocab

    def load_model(self, vocab_size: int, embed_size: int, learning_rate: float):
        """Helper function to load model, needs to occur after prep_cbow_data
        because we need the vocabulary size

        Params:
            vocab_size (int): The vocabulary size
            embed_size (int): The embedding dimensions
            learning_rate (float): The learning_rate
        """
        self.model = CBOW(vocab_size, embed_size)
        self.model.to(self.DEVICE)
        self.cbow_criterion = nn.NLLLoss()
        self.cbow_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate
        )

    def train_cbow(
        self, data: tuple[torch.Tensor, torch.Tensor], num_epochs: int, batch_size: int
    ):
        """Function to train cbow model

        Note: this code aparted from course code provided

        Params:
            data (tuple[torch.Tensor, torch.Tensor]: Training data in x, y pairs
            num_epochs (int): Number of epochs
            batch_size (int): Batch size
        :return:
        """
        # TODO: obviously would want a more robust check here
        if not self.model:
            raise ValueError("You need to initialize the model!")

        for epoch in range(num_epochs):
            losses = []
            for i in range(len(data) // batch_size):
                x, y = self.get_batch(data, i, batch_size)
                y_hat = self.model(x)
                # TODO: interesting to consider whether criterion and optimized should be part of class or not
                loss = self.cbow_criterion(y_hat, y)
                self.cbow_optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                self.cbow_optimizer.step()
                if i % 100 == 0:
                    print("iter", i, "loss", np.array(losses).mean())
            print("epoch", epoch, "loss", np.array(losses).mean())

    # TODO: add in test code, but frankly that was provided by course so I omitted it

    class SkipGram_Wrapper(Word2Vec):
        """A class to instantiate a continuous bag of words (CBOW) model

        Note: did not fill this class in as completely

        """

        def __init__(self, vocab_size, embed_dim, learning_rate: int):
            super().__init__()
            self.model = SkipGram(vocab_size, embed_dim)
            self.skip_criterion = nn.NLLLoss()
            self.skip_optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=learning_rate
            )

        def train_skipgram(self, data, num_epochs, batch_size):
            for epoch in range(num_epochs):
                losses = []
                for i in range(len(data) // batch_size):
                    x, y = self.get_batch(data, i, batch_size)
                    y_hat = self.model(x)
                    loss = None
                    # Calculate loss for every word in the context
                    for word in y.T:
                        if loss is None:
                            loss = self.skip_criterion(y_hat, word)
                        else:
                            loss += self.skip_criterion(y_hat, word)
                    self.skip_optimizer.zero_grad()
                    loss.backward()
                    losses.append(loss.item() / y.shape[1])
                    self.skip_optimizer.step()
                    if i % 100 == 0:
                        print("iter", i, "loss", np.array(losses).mean())
                print("epoch", epoch, "loss", np.array(losses).mean())


def main(model_name: str):
    # Load config
    # TODO: with more time would likely create a proper config object
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    # Load data
    WIKI_TRAIN, WIKI_TEST = load_dataset()

    if model_name == ModelNames.CBOW:
        cbow = CBOW_Wrapper()
        data, vocab = cbow.prep_cbow_data(WIKI_TRAIN, "test")
        cbow.load_model(
            vocab.num_words(),
            config.cbow.CBOW_EMBED_DIMENSIONS,
            config.cbow.CBOW_LEARNING_RATE,
        )
        cbow.train_cbow(data, config.cbow.CBOW_NUM_EPOCHS, config.cbow.CBOW_BATCH_SIZE)
    elif model_name == ModelNames.SKIP_GRAM:
        # TODO: add same as above
        pass
    else:
        raise ValueError("Incorrect model name passed")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    # Allows easy passing of command line arguments
    fire.Fire(main)
