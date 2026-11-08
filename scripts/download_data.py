from pathlib import Path
import urllib.request

#buscar url correta posteriormente
BASE_URL = "https://tatoeba.org/pt-br/exports/download/79728/Pares%20de%20frase%20em%20Ingl%C3%AAstraduzidas%20para%20o%20Portugu%C3%AAs-%20-%202026-11-08.tsv"

dir_path = Path("data/raw")

def download():

    #verifico a existência dos diretórios corretos
    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    #local exato do arquivo
    path = dir_path / "Pares%20de%20frase%20em%20Ingl%C3%AAstraduzidas%20para%20o%20Portugu%C3%AAs-%20-%202026-11-08.tsv"

    if not path.exists():
        print("ARQUIVO NÃO ENCONTRADO! BAIXANDO...")
        urllib.request.urlretrieve(BASE_URL, path)
        print("ARQUIVO BAIXADO!")

    else:
        print("ARQUIVO JÁ NO SISTEMA!")

download()

