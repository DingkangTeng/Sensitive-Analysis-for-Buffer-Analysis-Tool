import textwrap
import pandas as pd
import matplotlib.colors as mcolors

STANDER_NAME = {
    "ratioTrans500": "% of EVCS around transport metro stations within 500 meters",
    "ratioTerminal500": "% of EVCS around terminal metro stations within 500 meters",
    "ratioNormal500": "% of EVCS around normal metro stations within 500 meters",
    "ratioAll500": "% of EVCS around all metro stations within 500 meters",
    "ratioAll": "EVCS ratio for all MTR stations",
    "ratioAll_Baseline": "Average distributuion ratio for all MTR stations",
    "ratioAll_PaR": "Parking lots ratio for all MTR stations",
    "ratioNormal": "EVCS ratio for normal MTR stations",
    "ratioTerminal": "EVCS ratio for terminal MTR stations",
    "ratioTrans": "EVCS ratio for interchange MTR stations",
}

COLOR = {
    "US": "teal",
    "CN": "orangered",
    "EU": "deepskyblue"
}

TITLE_FONT = {"style":"italic","weight":"bold"}

def CITY_STANDER() -> dict:
    stander = pd.read_csv("analysis\\cityList.csv", encoding="utf-8")
    return dict(zip(stander["city"], stander["UID"]))

def adjustBrightness(color: str, factor: float):
    # Convert the color to HSV
    hsv = mcolors.rgb_to_hsv(mcolors.to_rgb(color))
    # Adjust the value (brightness)
    hsv[2] = max(0, min(1, hsv[2] * factor))  # Ensure value is between 0 and 1
    # Convert back to RGB
    return mcolors.hsv_to_rgb(hsv)

def wrapLabels(labels: list[str], width: int) -> list[str]:
    """
    width: The max number of characters in one line
    """
    return [textwrap.fill(label, width=width) for label in labels]