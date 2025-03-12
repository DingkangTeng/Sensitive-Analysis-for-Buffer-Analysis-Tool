import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

from globalAnalysis import analysis as GA
from function import CITY_STANDER, STANDER_NAME

# # Setting Chinese front
# mpl.rcParams["font.sans-serif"] = ["SimHei"]
# mpl.rcParams["axes.unicode_minus"] = False

class analysis(GA):
    dataList = []

    def __init__(self, metro: pd.DataFrame):
        metro.drop(metro.loc[metro["city"] == "San Juan"].index, inplace=True) # Delete San Juan
        # metro change city name using CITY_STANDER
        metro["city"] = metro["city"].map(CITY_STANDER())
        self.cities = metro["city"].unique().tolist()
        self.cities.sort()
        self.metro = metro

    def addData(self, dataPath: str, name: str, sub: str = "") -> None:
        """
        Add data into memory

        Parameter:
        dataPath: Input data path
        name: Input data category ("All", "Normal", "Terminal", "Trans")
        sub: Input sub data, default is Null string.

        Return:
        None
        """
        # Read data
        ratioName = "ratio" + name
        data = pd.read_csv(dataPath + ".csv", encoding="utf-8")
        data.drop(data.loc[data["city"] == "San Juan"].index, inplace=True) # Delete San Juan
        data[ratioName] = data["Num"] / data["totalNum"]
        data.rename(columns={"Num": "Num" + name}, inplace=True)
        dataBaseline = pd.read_csv(dataPath + "_Baseline.csv", encoding="utf-8")
        dataBaseline.drop(dataBaseline.loc[dataBaseline["city"] == "San Juan"].index, inplace=True) # Delete San Juan
        # Some buffer zone exceed the district area, recalculate the ratio into 100%
        dataBaseline.loc[dataBaseline["totalNum"] == 0, "Num"] = 1
        dataBaseline.loc[dataBaseline["totalNum"] == 0, "totalNum"] = 1
        dataBaseline[ratioName + "_Baseline"] = dataBaseline["Num"] / dataBaseline["totalNum"]
        dataBaseline = dataBaseline[["city", "distance", ratioName + "_Baseline"]]
        data = pd.merge(data, dataBaseline, how="inner", on=["city", "distance"])
        # Add sub data
        if sub != "":
            subData = pd.read_csv(dataPath.replace("CaR", sub) + ".csv", encoding="utf-8")
            subData[ratioName+ '_' + sub] = subData["Num"] / subData["totalNum"]
            subData.rename(columns={"totalNum": "total" + name + '_' + sub, "Num": "Num" + name + '_' + sub}, inplace=True)
            data = pd.merge(data, subData, how="inner", on=["city", "distance"])
        # data change city name using CITY_STANDER
        data["city"] = data["city"].map(CITY_STANDER())
        self.dataList.append(data)

        return
    
    # Merge all data into one csv
    def merge(self, path: str) -> None:
        self.data = pd.concat(self.dataList)
        self.data = self.data.groupby(["city", "distance", "totalNum"], as_index=False).agg("sum")
        self.data.sort_values(["city", "distance"], inplace=True)
        self.data.to_csv(path, encoding="utf-8", index=False)
        self.dataList = []

        return
    
    def saveFig(self, i: int, axs: plt.Axes, fig: plt.Figure, lines: list, labels: list, savePath: str):
        # Hide unused subplots
        for j in range(i, len(axs)):
            fig.delaxes(axs[j])
        
        # Add one legend
        # Change labels into a friendly name
        labels = [STANDER_NAME[x] if x in STANDER_NAME else x for x in labels]
        fig.legend(lines, labels, loc = "lower right")
        plt.tight_layout()
        plt.savefig(savePath, bbox_inches="tight")
        plt.close()

        return

    def skipFirst(self, threshold: int) -> int:
        j = 0
        while True:
            metro = self.metro.loc[self.metro["city"] == self.cities[j]]
            if metro["FREQUENCY"].iloc[0] >= threshold:
                break
            j += 1
        
        return j
    
    def calRowCol(self) -> Tuple[int, int]:
        cityNum = len(self.cities)
        colNum = int(cityNum ** 0.5)
        rowNum = (cityNum + colNum - 1) // colNum

        return colNum, rowNum

    def drawCurveAcc(self, path: str, columnList: list[str], threshold: int = 0) -> None:
        def plotDatas(i: int, axs: plt.Axes, city: str, columnStation: list[str], columnBaseline: list[str]) -> None:
            dataStation = self.data.loc[self.data["city"] == city, columnStation].set_index("distance")
            dataBaseline = self.data.loc[self.data["city"] == city, columnBaseline].set_index("distance")
            dataBaseline.plot(ax=axs[i], marker=',', color="gray")
            dataStation.plot(ax=axs[i], marker='.', title=city, ylabel="ratio")
            axs[i].set_xlabel("distance")
            axs[i].set_yticks(self.yticks)

            return
        
        colNum, rowNum = self.calRowCol()
        
        # Draw different condition
        for column in columnList:
            columnStation = ["distance", column, column + "_PaR"]
            columnBaseline = ["distance", column + "_Baseline"]
            fig, axs = plt.subplots(rowNum, colNum, figsize=(20, 20))
            axs = axs.flatten() # Flatten the 2D array of axes to 1D for easy indexing

            # Plot the first sub plot
            j = self.skipFirst(threshold)
            plotDatas(0, axs, self.cities[j], columnStation, columnBaseline)
            lines, labels = fig.axes[0].get_legend_handles_labels()
            fig.axes[0].get_legend().remove()

            i = 1
            for city in self.cities[j + 1:]: 
                # Skip city whose metro station number is less than the thresold
                metro = self.metro.loc[self.metro["city"] == city]
                if metro["FREQUENCY"].iloc[0] < threshold:
                    continue
                # Plot curve
                plotDatas(i, axs, city, columnStation, columnBaseline)
                fig.axes[i].get_legend().remove()
                i += 1
            
            savePath = path + "\\accum" + column + ".png"
            self.saveFig(i, axs, fig, lines, labels, savePath)

        return

    def drawCurveAll(self, path: str, columnList: list[str], threshold: int = 0) -> None:
        def plotDatas(i: int, axs: plt.Axes, city: str, columns: list[str]) -> None:
            dataStation = self.data.loc[self.data["city"] == city, columns].set_index("distance")
            dataStation.plot(ax=axs[i], marker='.', title=city, ylabel="ratio")
            axs[i].set_xlabel("distance")
            axs[i].set_yticks(self.yticks)

            return
        
        colNum, rowNum = self.calRowCol()
        
        columns = ["distance"] + columnList
        fig, axs = plt.subplots(rowNum, colNum, figsize=(20, 20))
        axs = axs.flatten() # Flatten the 2D array of axes to 1D for easy indexing

        # Plot the first sub plot
        j = self.skipFirst(threshold)
        plotDatas(0, axs, self.cities[j], columns)
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.axes[0].get_legend().remove()

        i = 1
        for city in self.cities[j + 1:]: 
            # Skip city whose metro station number is less than the thresold
            metro = self.metro.loc[self.metro["city"] == city]
            if metro["FREQUENCY"].iloc[0] < threshold:
                continue
            # Plot curve
            plotDatas(i, axs, city, columns)
            fig.axes[i].get_legend().remove()
            i += 1
        
        savePath = path + "\\sensative.png"
        self.saveFig(i, axs, fig, lines, labels, savePath)

        return

# Analysis using saved output csv file
class analysisAll(analysis):
    def __init__(self, metro: pd.DataFrame, data: pd.DataFrame):
        super().__init__(metro)
        self.data = data.sort_values(by=["city", "distance"])

def runAnalysis(a: analysis | analysisAll, path: str) -> None:
    # a.drawCurveAcc(path, ["ratioAll", "ratioNormal", "ratioTerminal", "ratioTrans"], 5) # Draw the comparation cruve bteween charging, parking and baseline
    # a.drawCurveAll(path, ["ratioAll", "ratioNormal", "ratioTerminal", "ratioTrans"], 5) # Draw charging and riding cruve
    # a.drawCurveAll(path, ["ratioAll_PaR", "ratioNormal_PaR", "ratioTerminal_PaR", "ratioTrans_PaR"], 5) # Draw parking and riding cruve
    
    return

if __name__ == "__main__":
    # First round analysis
    for country in ["CH", "US", "EU"]:
        path = "..\\Export\\" + country + "\\"
        metroPath = path + country + "_Metro.csv"
        metro = pd.read_csv(metroPath, encoding="utf-8")
        a = analysis(metro)

        '''
        All - All station
        Terminal - Only terminal station
        Trans - Only transfer staiton
        Normal - Neither terminal station nor transfer station
        '''
        for stationType in ["All", "Normal", "Terminal", "Trans"]:
            dataPath = path + country + "_CaR_" + stationType
            a.addData(dataPath, stationType, "PaR")
        
        # Merger all different data and save to one csv file
        a.merge(path + country + ".csv")

        # # Run analysis method
        # runAnalysis(a, path)
    
    # # Analysis using saved csv file (country by country)
    # for country in ["CH", "US", "EU"]:
    #     path = "..\\Export\\" + country + "\\"
    #     metroPath = path + country + "_Metro.csv"
    #     dataPath = path + country + ".csv"
    #     metro = pd.read_csv(metroPath, encoding="utf-8")
    #     data = pd.read_csv(dataPath, encoding="utf-8")
    #     a = analysisAll(metro, data)
        
    #     # Run analysis method
    #     runAnalysis(a, path)