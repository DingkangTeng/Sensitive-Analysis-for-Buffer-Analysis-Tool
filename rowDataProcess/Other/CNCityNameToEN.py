import pandas as pd
from pypinyin import lazy_pinyin, Style

cityListPath = r"rowDataProcess\\Other\\CNCity.csv"

def U(a):
    return ''.join(lazy_pinyin(a, style=Style.FIRST_LETTER))[:2].upper()

df = pd.read_csv(cityListPath)

df["citynameUP"] = df["cityname"].apply(U)
df["prUP"] = df["pname"].apply(U)

df.to_csv(cityListPath, encoding="utf-8")