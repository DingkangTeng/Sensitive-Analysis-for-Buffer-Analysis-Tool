import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker
from matplotlib.patches import Patch

from function import CITY_STANDER, COLOR, STANDER_NAME, TITLE_FONT

class analysis:
    metro = pd.DataFrame()
    data = pd.DataFrame()
    cities = []
    yticks = [0, 0.5, 1] # Unify y-axis

    def __init__(self, metro: pd.DataFrame = metro, data: pd.DataFrame = data):
        if not metro.empty:
            # metro & data change city name using CITY_STANDER
            metro["city"] = metro["city"].map(CITY_STANDER())
            self.cities += metro["city"].unique().tolist()
            data.sort_values(by=["city", "distance"], inplace=True)
        self.metro = pd.concat([self.metro, metro])
        self.data = pd.concat([self.data, data])
            
    def append(self, metro: pd.DataFrame, data: pd.DataFrame) -> None:
        self.__init__(metro, data)

        return

    def compareRatio(self, distance: list[int], typelist: list[str] | str, path: str = "", threshold: int = 0) -> pd.DataFrame:
        # Initialize statice dataframe
        result = pd.DataFrame({"city": self.cities})
        if type(typelist) is list:
            typestr = typelist[-1]
        else:
            typestr = typelist
        
        col = "ratio" + typestr
        for i in distance:
            result[col + str(i)] = 0

        for city in self.cities:
            # Skip city whose metro station number is less than the thresold
            metro = self.metro.loc[self.metro["city"] == city]
            if metro["FREQUENCY"].iloc[0] < threshold:
                result.drop(result.loc[result["city"] == city].index, inplace=True)
                continue
            
            for i in distance:
                data = self.data.loc[(self.data["city"] == city) & (self.data["distance"] == i), [col]]
                result.loc[result["city"] == city, [col + str(i)]] = data.iloc[0,0]
        
        # Recursion for the rest elements
        if type(typelist) is list and len(typelist) > 1:
            typelist.pop()
            lastResult = self.compareRatio(distance, typelist, "", threshold)
            result = result.join(lastResult.set_index("city"), on="city")
        
        # Save result
        if path != "":
            result.sort_values(by=["city"]).to_csv(path, encoding="utf-8", index=False)

        return result
    
    def calculateAccumulation(self, path: str, interval: int = 10, threshold: int = 500, sub: str = "") -> None:
        def formular(sequencesA: list[int | float], sequencesB: list[int | float], interval: int) -> float:
            segmaA = np.sum(sequencesA[:-1])
            resultA = (segmaA + 0.5 * sequencesA[-1]) * interval
            segmaB = np.sum(sequencesB[:-1])
            resultB = (segmaB + 0.5 * sequencesB[-1]) * interval
            if resultA == 0 and resultB == 0:
                return None
            else:
                return np.around(resultA - resultB, 4)
        
        # Initialize results
        cities = self.data["city"].unique().tolist()
        cities.sort()
        cityNum = len(cities)
        result = pd.DataFrame({"city": cities,
                               "Accumulation of all stations": [0] * cityNum,
                               "Accumulaiton of normal stations": [0] * cityNum,
                               "Accululation of transfer stations": [0] * cityNum,
                               "Accumulation of terminal stations": [0] * cityNum
                            })
        if sub != "":
            result["Accumulation of all stations_" + sub] = 0
            result["Accumulaiton of normal stations_" + sub] = 0
            result["Accululation of transfer stations_" + sub] = 0
            result["Accumulation of terminal stations_" + sub] = 0
        
        for i in range(cityNum):
            data = self.data.loc[(self.data["city"] == cities[i]) & (self.data["distance"] <= threshold)]
            result.loc[i, "Accumulation of all stations"] = formular(data["ratioAll"].to_list(), data["ratioAll_Baseline"].to_list(), interval)
            result.loc[i, "Accumulaiton of normal stations"] = formular(data["ratioNormal"].to_list(), data["ratioNormal_Baseline"].to_list(), interval)
            result.loc[i, "Accululation of transfer stations"] = formular(data["ratioTerminal"].to_list(), data["ratioTerminal_Baseline"].to_list(), interval)
            result.loc[i, "Accumulation of terminal stations"] = formular(data["ratioTrans"].to_list(), data["ratioTrans_Baseline"].to_list(), interval)
            if sub != "":
                result.loc[i, "Accumulation of all stations_" + sub] = formular(data["ratioAll_" + sub].to_list(), data["ratioAll_Baseline"].to_list(), interval)
                result.loc[i, "Accumulaiton of normal stations_" + sub] = formular(data["ratioNormal_" + sub].to_list(), data["ratioNormal_Baseline"].to_list(), interval)
                result.loc[i, "Accululation of transfer stations_" + sub] = formular(data["ratioTerminal_" + sub].to_list(), data["ratioTerminal_Baseline"].to_list(), interval)
                result.loc[i, "Accumulation of terminal stations_" + sub] = formular(data["ratioTrans_" + sub].to_list(), data["ratioTrans_Baseline"].to_list(), interval)
        
        result.sort_values(by=["city"]).to_csv(path, encoding="utf-8", index=False)

        return
    
    # Distributuin Plot
    def distributionPlot(self, path: str, areas: list[str], distance: int = 500, threshold: int = 0) -> None:
        metro = self.metro.loc[self.metro["FREQUENCY"] >= threshold].copy()
        metro["MTRPer"] = metro["hasCharging" + str(distance)] / metro["FREQUENCY"]
        metro = metro[["city", "MTRPer"]]
        data = self.data.loc[self.data["distance"] == distance].copy()
        data = data.merge(metro, how="inner", on="city")
        
        plt.figure(figsize=(10, 10))
        ax = plt.axes()
        for area in areas:
            subData = data.loc[data["city"].str[0:2] == area].copy()
            subData.plot.scatter("ratioAll", "MTRPer", color=COLOR.get(area), label=area, ax=ax)
        
        # Set Plots
        ## 1:1 Line
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        ax.set_xlabel("% of EVCS")
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(1,0))
        ax.set_ylabel("% of MTR Station \n who has EVCS withing {} meters".format(distance))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1,0))
        plt.legend(loc=4)

        plt.show()
        # plt.savefig(path)
        # plt.close()

    def drawGolbalBar(self, path: str, typelist: list[str], sub :str, distances: int = 500, threshold: int = 0) -> None:
        fullList = typelist + [x + sub for x in typelist]
        result = self.compareRatio([distances], fullList, threshold=threshold)
        result.sort_values(by=["city"], inplace=True)
        result["color"] = result["city"].str[0:2]
        countries = result["color"].unique().tolist()
        result["color"] = result["color"].replace(COLOR)
        result.set_index("city", inplace=True)
        types = ["ratio" + x + str(distances) for x in typelist]
        typesLen = len(types)

        # Plot
        fig, axs = plt.subplots(typesLen, 1, figsize=(10, 3 * typesLen))
        axs = axs.flatten() # Flatten the 2D array of axes to 1D for easy indexing
        for i in range(typesLen):
            result.sort_values(by=["city", types[i]], inplace=True)
            result[types[i]].plot.bar(width=0.8, ax=axs[i], color=result["color"])

            # Add parking
            result[types[i].replace(str(distances),"") + sub + str(distances)].plot(ax=axs[i], marker=',', color='black')

            axs[i].set_yticks(self.yticks)
            axs[i].set_xlabel(None)
            axs[i].set_xticks([])
            axs[i].set_title("({}) {}".format(chr(i+97), STANDER_NAME.get(types[i])), fontdict=TITLE_FONT) # unicode 97 is a

        # Add legend
        customLegend = []
        for i in countries:
            customLegend.append(
                Patch(color=COLOR.get(i), label=i)
            )
        plt.legend(handles=customLegend, bbox_to_anchor=(0.5, -0.3), loc=8, ncol = len(countries))
        
        plt.show()
        # plt.savefig(path)
        # plt.close()

if __name__ == "__main__":
    a = analysis()
    
    for country in ["US", "CH", "EU"]:
        path = "..\\Export\\" + country + "\\"
        metroPath = path + country + "_Metro.csv"
        dataPath = path + country + ".csv"
        metro = pd.read_csv(metroPath, encoding="utf-8")
        data = pd.read_csv(dataPath, encoding="utf-8")
        a.append(metro, data)

    # a.calculateAccumulation("..\\Export\\Accumulation.csv")
    a.compareRatio([500], ["All", "Normal", "Terminal", "Trans"], "..\\Export\\global500.csv", 5)
    # a.distributionPlot("..\\Export\\G-generlaDistribution.jpg", ["US", "CN", "EU"], threshold=5)
    # a.drawGolbalBar("..\\Export\\G-bar.jpg", ["All", "Normal", "Terminal", "Trans"], "_PaR", threshold=5)