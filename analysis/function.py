import textwrap
import pandas as pd
import matplotlib.colors as mcolors

STANDER_NAME = {
    "ratioTrans500": "% of EVCS around interchange stations within 500 meters",
    "ratioTerminal500": "% of EVCS around terminal stations within 500 meters",
    "ratioNormal500": "% of EVCS around normal stations within 500 meters",
    "ratioAll500": "% of EVCS around all stations within 500 meters",
    "ratioAll": "EVCS ratio for all stations",
    "ratioAll_Baseline": "Average distributuion ratio for all stations",
    "ratioAll_PaR": "Parking lots ratio for all stations",
    "ratioNormal": "EVCS ratio for normal stations",
    "ratioTerminal": "EVCS ratio for terminal stations",
    "ratioTrans": "EVCS ratio for interchange stations",
    "All": "All stations within 500 meters",
    "Normal": "Normal stations within 500 meters",
    "Terminal": "Terminal stations within 500 meters",
    "Trans": "Interchange stations wtihin 500 meters",
    "_PaR": "Parking lots"
}

COLOR = {
    "US": "teal",
    "CN": "orangered",
    "EU": "deepskyblue",
}

# Font control
TICK_FONT_INT = 21
TICK_FONT = {"size": TICK_FONT_INT}
MARK_FONT_INT = 24
MARK_FONT = {"size": MARK_FONT_INT}
TITLE_FONT = {"font": "DejaVu Sans", "style": "italic", "weight": "bold", "size": MARK_FONT_INT}

def CITY_STANDER() -> dict:
    stander = pd.read_csv("analysis\\cityList.csv", encoding="utf-8")
    return dict(zip(stander["city"], stander["UID"]))

def adjustBrightness(color: str, factor: float) -> str:
    # Convert the color to HSV
    hsv = mcolors.rgb_to_hsv(mcolors.to_rgb(color))
    # Adjust the value (brightness)
    hsv[2] = max(0, min(1, hsv[2] * factor))  # Ensure value is between 0 and 1
    # Convert back to RGB
    rgb = mcolors.hsv_to_rgb(hsv)
    return mcolors.to_hex(tuple(rgb))

def wrapLabels(labels: list[str], width: int) -> list[str]:
    """
    width: The max number of characters in one line
    """
    return [textwrap.fill(label, width=width) for label in labels]