import pandas as pd

STANDER_NAME = {
    "": "",
}

def CITY_STANDER() -> dict:
    stander = pd.read_csv("analysis\\cityList.csv", encoding="utf-8")
    return dict(zip(stander["city"], stander["UID"]))