import pandas as pd

STANDER_NAME = {
    "ratioTrans500": "% of EVCS around transport metro stations within 500 meters",
    "ratioTerminal500": "% of EVCS around terminal metro stations within 500 meters",
    "ratioNormal500": "% of EVCS around normal metro stations within 500 meters",
    "ratioAll500": "% of EVCS around all metro stations within 500 meters"
}

COLOR = {
    "US": 'teal',
    "CN": 'r',
    "EU": 'b'
}

TITLE_FONT = {"style":"italic","weight":"bold"}

def CITY_STANDER() -> dict:
    stander = pd.read_csv("analysis\\cityList.csv", encoding="utf-8")
    return dict(zip(stander["city"], stander["UID"]))