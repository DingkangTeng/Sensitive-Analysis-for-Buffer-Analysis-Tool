import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl

from globalAnalysis import analysis as GA
from function import CITY_STANDER

# # Setting Chinese front
# mpl.rcParams["font.sans-serif"] = ["SimHei"]
# mpl.rcParams["axes.unicode_minus"] = False

class analysis(GA):
    data = []

    def __init__(self, metro: pd.DataFrame):
        # metro change city name using CITY_STANDER
        metro["city"] = metro["city"].map(CITY_STANDER())
        self.cities = metro["city"].unique()
        self.metro = metro

    def addData(self, dataPath: str, name: str) -> None:
        # Read data
        ratioName = "ratio" + name
        data = pd.read_csv(dataPath + ".csv", encoding="utf-8")
        dataBaseline = pd.read_csv(dataPath + "_Baseline.csv", encoding="utf-8")
        data[ratioName] = data["Num"] / data["totalNum"]
        data.rename(columns={"Num": "Num" + name}, inplace=True)
        # Read baseline
        dataBaseline[ratioName + "_Baseline"] = dataBaseline["Num"] / dataBaseline["totalNum"]
        data = data.set_index("city").join(dataBaseline[ratioName + "Baseline"], on="city")
        # data change city name using CITY_STANDER
        data["city"] = data["city"].map(CITY_STANDER())
        self.data.append(data)

        return
    
    # Merge all data into one csv
    def merge(self, path: str) -> None:
        self.data = pd.concat(self.data)
        self.data = self.data.groupby(["city", "distance", "totalNum"], as_index=False).agg("sum")
        self.data.to_csv(path, encoding="utf-8", index=False)

        return
    
    # 加上baseline后线太多了，想想咋重新画图
    def drawCurve(self, path: str, threshold: int = 0) -> None:
        columns = ["distance", "ratioAll", "ratioNormal", "ratioTerminal", "ratioTrans"]
        cityNum = len(self.cities)
        colNum = int(cityNum ** 0.5)
        rowNum = (cityNum + colNum - 1) // colNum
        fig, axs = plt.subplots(rowNum, colNum, figsize=(20, 20))
        axs = axs.flatten() # Flatten the 2D array of axes to 1D for easy indexing

        i = 0
        for city in self.cities: 
            # Skip city whose metro station number is less than the thresold
            metro = self.metro.loc[self.metro["city"] == city]
            if metro["FREQUENCY"].iloc[0] < threshold:
                continue

            data = self.data.loc[self.data["city"] == city, columns].set_index("distance")

            # Plot curve
            if i == 0:
                data.plot(ax=axs[i], marker='o', title=city, ylabel="ratio")
                axs[i].set_xlabel("distance")
                axs[i].set_yticks(self.yticks)
                lines, labels = fig.axes[i].get_legend_handles_labels()
                fig.axes[i].get_legend().remove()
                i = 1
                continue
            data.plot(ax=axs[i], marker='.', title=city, ylabel="ratio", legend=False)
            axs[i].set_xlabel("distance")
            axs[i].set_yticks(self.yticks)
            i += 1
        
        # Hide unused subplots
        for j in range(i, len(axs)):
            fig.delaxes(axs[j])
        
        # Add one legend
        # Change labels into a friendly name
        # ...
        fig.legend(lines, labels, loc = "lower right")
        plt.tight_layout()
        plt.savefig(path + '\\multiple_dataframes_plot.png', bbox_inches='tight')
        plt.close()

        return
    
    # Draw the distribution of the PCR ration in different area
    def drawBar(self, distance: list[int], typestr: str, path: str, threshold: int = 0) -> None:
        data = self.compareRatio(distance, typestr, threshold=threshold)
        data.set_index("city", inplace=True)
        data.sort_values("ratio" + typestr + str(distance[0]), ascending=False, inplace=True)
        # Only display the top 20 cities
        data.iloc[:20].plot.bar()
        plt.yticks(self.yticks)
        plt.savefig(path + '\\rankingsTop20.png', bbox_inches='tight')
        plt.close()

        # All cities
        data.plot.bar(figsize=(20,15))
        plt.savefig(path + '\\rankings.png', bbox_inches='tight')
        plt.close()

        return

# Analysis using saved output csv file
class analysisAll(analysis):
    def __init__(self, metro: pd.DataFrame, data: pd.DataFrame):
        data["city"] = data["city"].map(CITY_STANDER())
        super().__init__(metro)
        self.data = data

def runAnalysis(a: analysis | analysisAll, path: str) -> None:
    a.drawCurve(path, 5)
    a.drawBar([500], "All", path, 5)
    
    return

if __name__ == "__main__":
    # First round analysis
    for country in ["EU", "US", "CH"]:
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
            a.addData(dataPath, stationType)
        
        # Merger all different data and save to one csv file
        a.merge(path + country + ".csv")

        # Run analysis method
        runAnalysis(a, path)
    
    # Analysis using saved csv file
    for country in []:
        path = "..\\Export\\" + country + "\\"
        metroPath = path + country + "_Metro.csv"
        dataPath = path + country + ".csv"
        metro = pd.read_csv(metroPath, encoding="utf-8")
        data = pd.read_csv(dataPath, encoding="utf-8")
        a = analysisAll(metro, data)
        
        # Run analysis method
        runAnalysis(a, path)