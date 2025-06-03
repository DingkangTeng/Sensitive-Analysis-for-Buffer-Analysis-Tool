import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from typing import Tuple, Hashable
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from numpy._typing import NDArray
from scipy.interpolate import PchipInterpolator

from globalAnalysis import analysis as GA
from function import CITY_STANDER, STANDER_NAME, wrapLabels, TICK_FONT_INT, TICK_FONT, MARK_FONT

# # Setting Chinese front
# mpl.rcParams["font.sans-serif"] = ["SimHei"]
# mpl.rcParams["axes.unicode_minus"] = False

class analysis(GA):
    dataList = []

    def __init__(self, metro: pd.DataFrame, interval: int = 10):
        metro.drop(metro.loc[metro["city"] == "San Juan"].index, inplace=True) # Delete San Juan
        # metro change city name using CITY_STANDER
        metro["city"] = metro["city"].map(CITY_STANDER())
        self.cities = metro["city"].unique().tolist()
        self.cities.sort()
        self.metro = metro
        self.interval = interval

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
    
    def saveFig(self, i: int, axs: list[plt.Axes] | NDArray, fig: Figure, lines: list, labels: list, savePath: str):
        # Hide unused subplots
        for j in range(i, len(axs)):
            fig.delaxes(axs[j])
        
        # Add one legend
        # Change labels into a friendly name
        labels = wrapLabels([STANDER_NAME[x] if x in STANDER_NAME else x for x in labels], 25)
        # fig.legend(lines, labels, loc="lower right")
        fig.legend(lines, labels, bbox_to_anchor=(1, 0.18))
        plt.tight_layout()
        plt.savefig(savePath, bbox_inches="tight", dpi=300)
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

    def drawCurveAcc(self, path: str, columnList: list[str], threshold: int = 0, distance: int = 500) -> None:
        def plotDatas(i: int, axs: list[plt.Axes] | NDArray, city: str, columnStation: list[str], columnBaseline: list[str]) -> None:
            quary = (self.data["city"] == city) & (self.data["distance"] <= distance)
            dataStation = self.data.loc[quary, columnStation].set_index("distance")
            dataBaseline = self.data.loc[quary, columnBaseline].set_index("distance")
            dataBaseline.plot(ax=axs[i], marker=',', color="gray")
            dataStation.plot(ax=axs[i], marker='.', title=city, ylabel="ratio", color=["#5F8B4C", "#EB5B00"])
            axs[i].set_xlabel("distance")
            axs[i].set_yticks(self.yticks)
            axs[i].set_xticks([0, 100, 200, 300, 400, 500])

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
            l = fig.axes[0].get_legend()
            assert l is not None
            l.remove()

            i = 1
            for city in self.cities[j + 1:]: 
                # Skip city whose metro station number is less than the thresold
                metro = self.metro.loc[self.metro["city"] == city]
                if metro["FREQUENCY"].iloc[0] < threshold:
                    continue
                # Plot curve
                plotDatas(i, axs, city, columnStation, columnBaseline)
                l = fig.axes[i].get_legend()
                assert l is not None
                l.remove()
                i += 1
            
            savePath = path + "\\accum" + column + ".png"
            self.saveFig(i, axs, fig, lines, labels, savePath)

        return

    def drawCurveAll(self, path: str, columnList: list[str], threshold: int = 0, distance: int = 500) -> None:
        def plotDatas(i: int, axs: list[plt.Axes] | NDArray, city: str, columns: list[str], distance: int) -> None:
            dataStation = self.data.loc[(self.data["city"] == city) & (self.data["distance"] <= distance), columns].set_index("distance")
            dataStation.plot(ax=axs[i], marker='.', title=city, ylabel="ratio", color=["#5F8B4C", "#FFDDAB", "#FF9A9A", "#945034"])
            axs[i].set_xlabel("distance")
            axs[i].set_yticks(self.yticks)
            axs[i].set_xticks([0, 100, 200, 300, 400, 500])

            return
        
        colNum, rowNum = self.calRowCol()
        
        columns = ["distance"] + columnList
        fig, axs = plt.subplots(rowNum, colNum, figsize=(20, 20))
        axs = axs.flatten() # Flatten the 2D array of axes to 1D for easy indexing

        # Plot the first sub plot
        j = self.skipFirst(threshold)
        plotDatas(0, axs, self.cities[j], columns, distance)
        lines, labels = fig.axes[0].get_legend_handles_labels()
        l = fig.axes[0].get_legend()
        assert l is not None
        l.remove()

        i = 1
        for city in self.cities[j + 1:]: 
            # Skip city whose metro station number is less than the thresold
            metro = self.metro.loc[self.metro["city"] == city]
            if metro["FREQUENCY"].iloc[0] < threshold:
                continue
            # Plot curve
            plotDatas(i, axs, city, columns, distance)
            l = fig.axes[i].get_legend()
            assert l is not None
            l.remove()
            i += 1
        
        savePath = path + "\\sensative.png"
        self.saveFig(i, axs, fig, lines, labels, savePath)

        return
    
    def drawHeatMap(self, path: str, columnList: list[str], sub: list[str] = [], threshold: int = 0, distance: int = 500) -> None:
        #Front for heatmap
        __MARK_FONT = MARK_FONT.copy()
        __MARK_FONT["size"] = 24
        __TICK_FONT_INT = 16
        __TICK_FONT = TICK_FONT.copy()
        __TICK_FONT["size"] = __TICK_FONT_INT
        __SUB_SIZE = 16

        columns = ["distance"] + columnList
        if sub != []:
            for i in sub:
                columns += [x + i for x in columnList]

        # Get data
        baseNum = len(columnList)
        count = 0
        sortCity = []
        cities = self.metro.loc[self.metro["FREQUENCY"] >= threshold, "city"].to_list()
        data = self.data.loc[(self.data["city"].isin(cities)) & (self.data["distance"] <= distance)].copy()

        # Change city name into identifier and save the table
        origCities = data["city"].unique().tolist()
        ## Save tabel
        cityTable = pd.DataFrame({"Region": origCities})
        cityTable.index = pd.Index(cityTable["Region"].str[0] + cityTable.index.astype(str))
        cityTable.index.name = "ID"
        discription = pd.read_csv(r"analysis//cityList.csv", usecols=[3,4,5,6])
        discription.set_index("UID", inplace=True)
        cityTable = cityTable.join(discription, on="Region")
        cityTable.to_csv(path + "cityIndex.csv", encoding="utf-8")
        ## Change city name to identifer
        data.replace({"city": dict(zip(origCities, cityTable.index))}, inplace=True)

        for column in columns[1:]:
            dataPivot = data.pivot(index="city", columns="distance", values=column)
            # Main data
            if count < baseNum:
                dataPivot.sort_values(by=distance, ascending=False, inplace=True)
                sortCity = dataPivot.index
            # Sub data
            else:
                dataPivot = dataPivot.loc[sortCity] # Sort sub data based on the main data

            plt.figure(figsize=(15, 15))  # Set the figure size
            heatmap = sns.heatmap(
                dataPivot,
                cmap="Reds",
                annot=False, cbar=True,
                vmin=0, vmax=1,
                linewidths=0.05, linecolor="white",
                cbar_kws={"pad": 0.07}, # gap between colorbar and heatmap
            )

            # Adjust the color bar
            colorbar = heatmap.collections[0].colorbar
            colorbar.ax.tick_params(
                labelsize=__TICK_FONT_INT, # Set the font size for color bar ticks
                left=True, labelleft=True, # Show left axis
                right=False, labelright=False # Hide right axis
            )
            colorbar.ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            colorbar.ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
            colorbar.ax.set_xlabel(
                "% of\n{}".format(STANDER_NAME[column][:-29].replace(' ', "\n")),
                fontdict={"size": __SUB_SIZE}
            )

            # Add titles and labels
            plt.xticks(
                ticks=[x for x in range(distance // self.interval + 1)],
                labels=[10 * x if x % 10 == 0 else '' for x in range(distance // self.interval)] + [distance],
                rotation=0,
                fontdict=__TICK_FONT
                )
            plt.yticks(rotation=0, size=__TICK_FONT_INT)
            # Adjust minor x axis
            ax = plt.gca()
            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))  # Major ticks every 10 unit
            ax.tick_params(axis='x', which="major", length=5)
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))  # Minor ticks every 1 units
            ax.tick_params(axis='x', which="minor", length=3, color='gray')
            plt.xlabel("Distance", fontdict=__MARK_FONT)
            plt.ylabel('Study Unites ID', fontdict=__MARK_FONT)

            # Create a new axis for the curve
            curveAx = plt.gca().inset_axes([1.2, 0, 0.11, 1])  # [x0, y0, width, height(multiple of existing length)] of the curveAx
            # Generate data for the curve
            valueCounts = dataPivot[distance].round(1).value_counts().sort_index()
            valueSum = valueCounts.sum()
            valueCounts = valueCounts / valueSum
            x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            tmp = pd.DataFrame(index=x)
            y = tmp.join(valueCounts).fillna(0)["count"].to_list()
            # Create a smooth curve using interpolation
            xSmooth = np.linspace(0, 1, 100)  # 100 points for smoothness
            pchipInterpolator = PchipInterpolator(x, y)  # Cubic spline interpolation
            ySmooth = np.maximum(pchipInterpolator(xSmooth), 0)
            ySmooth = np.minimum(max(valueCounts), ySmooth)
            # Plot the curve
            curveAx.plot(ySmooth, xSmooth, color="darkblue")
            curveAx.tick_params(labelsize=__TICK_FONT_INT)
            curveAx.set_ylim(0, 1)
            curveAx.set_xlim(0, 0.5)
            curveAx.set_yticks([])
            curveAx.set_xlabel("PDF of\nStudy Unites", fontdict={"size":__SUB_SIZE})
            # curveAx.axis('off')  # Hide the axis

            # Show the plot
            count += 1
            # plt.show()
            plt.savefig(path + column + "_HeatMap.jpg", bbox_inches="tight", dpi=300)
            plt.close()

        return

# Analysis using saved output csv file
class analysisAll(analysis):
    def __init__(self, metro: pd.DataFrame, data: pd.DataFrame):
        super().__init__(metro)
        self.data = data.sort_values(by=["city", "distance"])

def runAnalysis(a: analysis | analysisAll, path: str) -> None:
    # a.drawCurveAcc(path, ["ratioAll"], 7) # Draw the comparation cruve bteween charging, parking and baseline
    # a.drawCurveAll(path, ["ratioAll", "ratioNormal", "ratioTerminal", "ratioTrans"], 7) # Draw charging and riding cruve
    # a.drawCurveAll(path, ["ratioAll_PaR", "ratioNormal_PaR", "ratioTerminal_PaR", "ratioTrans_PaR"], 7) # Draw parking and riding cruve
    a.drawHeatMap(path, ["ratioAll"], ["_PaR"], 7)
    
    return

if __name__ == "__main__":
    # # First round analysis
    # for country in ["CH", "US", "EU"]:
    #     path = "..\\Export\\" + country + "\\"
    #     metroPath = path + country + "_Metro.csv"
    #     metro = pd.read_csv(metroPath, encoding="utf-8")
    #     a = analysis(metro)

    #     '''
    #     All - All station
    #     Terminal - Only terminal station
    #     Trans - Only interchange staiton
    #     Normal - Neither terminal station nor interchange station
    #     '''
    #     for stationType in ["All", "Normal", "Terminal", "Trans"]:
    #         dataPath = path + country + "_CaR_" + stationType
    #         a.addData(dataPath, stationType, "PaR")
        
    #     # Merger all different data and save to one csv file
    #     a.merge(path + country + ".csv")

    #     # # Run analysis method
    #     # runAnalysis(a, path)
    
    # Analysis using saved csv file (country by country)
    for country in ["CH", "US", "EU"]: #
        path = "..\\Export\\" + country + "\\"
        metroPath = path + country + "_Metro.csv"
        dataPath = path + country + ".csv"
        metro = pd.read_csv(metroPath, encoding="utf-8")
        data = pd.read_csv(dataPath, encoding="utf-8")
        a = analysisAll(metro, data)
        
        # Run analysis method
        runAnalysis(a, path)