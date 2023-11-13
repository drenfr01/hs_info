import datasets
import pandas as pd
from datasets import load_dataset


def import_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A utility function to download datasets for training purposes

    Returns:
         a tuple containing the training dataset and a test_dataset
    """
    wiki_data_train = load_dataset("wikitext", "wikitext-2-v1", split="train").shuffle()
    wiki_data_test = load_dataset("wikitext", "wikitext-2-v1", split="test").shuffle()
    WIKI_TRAIN = pd.DataFrame(wiki_data_train)
    WIKI_TEST = pd.DataFrame(wiki_data_test)

    return WIKI_TRAIN, WIKI_TEST
