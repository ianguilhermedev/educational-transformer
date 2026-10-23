import pandas as pd
from pathlib import Path

def load_data(
    path=None
):
    '''
    return: se path=None retorna os dados processados
    '''

    if path is None:
        path = Path("../data/processed/tatoeba_data/en-pt-pairs.tsv")

    return pd.read_csv(path, sep="\t")
