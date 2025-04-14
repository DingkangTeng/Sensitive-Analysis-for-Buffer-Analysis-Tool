import textwrap
import pandas as pd
import matplotlib.colors as mcolors

STANDER_NAME = {
    "ratioTrans500": "% of EVCS around interchange metro stations within 500 meters",
    "ratioTerminal500": "% of EVCS around terminal metro stations within 500 meters",
    "ratioNormal500": "% of EVCS around normal metro stations within 500 meters",
    "ratioAll500": "% of EVCS around all metro stations within 500 meters",
    "ratioAll": "EVCS ratio for all metro stations",
    "ratioAll_Baseline": "Average distributuion ratio for all metro stations",
    "ratioAll_PaR": "Parking lots ratio for all metro stations",
    "ratioNormal": "EVCS ratio for normal metro stations",
    "ratioTerminal": "EVCS ratio for terminal metro stations",
    "ratioTrans": "EVCS ratio for interchange metro stations",
    "All": "All metro stations within 500 meters",
    "Normal": "Normal metro stations within 500 meters",
    "Terminal": "Terminal metro stations within 500 meters",
    "Trans": "Interchange metro stations wtihin 500 meters",
    "_PaR": "Parking lots"
}

COLOR = {
    "US": "teal",
    "CN": "orangered",
    "EU": "deepskyblue",
}

# Font control
TITLE_FONT = {"style": "italic", "weight": "bold", "size": 16}
TICK_FONT_INT = 14
TICK_FONT = {"size": TICK_FONT_INT}
MARK_FONT_INT = 16
MARK_FONT = {"size": MARK_FONT_INT}

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