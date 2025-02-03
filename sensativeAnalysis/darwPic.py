import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl

# Setting Chinese front
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

class analysis:
    data = []

    def __init__(self, metro: pd.DataFrame):
        self.cities = metro["city"].unique()
        self.metro = metro

    def addData(self, data: pd.DataFrame, name: str) -> None:
        data["ratio" + name] = data["Num"] / data["totalNum"]
        data.rename(columns={"Num": "Num" + name}, inplace=True)
        self.data.append(data)

        return
    
    # Merge all data into one csv
    def merge(self, path: str) -> None:
        self.data = pd.concat(self.data)
        self.data = self.data.groupby(["city", "distance", "totalNum"], as_index=False).agg("sum")
        self.data.to_csv(path, encoding="utf-8", index=False)

        return
    
    def drawCurve(self, path: str, threshold: int = 0) -> None:
        columns = ["distance", "ratioAll", "ratioNormal", "ratioTerminal", "ratioTrans"]
        cityNum = len(self.cities)
        colNum = int(cityNum ** 0.5)
        rowNum = (cityNum + colNum - 1) // colNum
        fig, axs = plt.subplots(rowNum, colNum, figsize=(20, 20))
        axs = axs.flatten() # Flatten the 2D array of axes to 1D for easy indexing
        yticks = [0, 0.5, 1]

        i = 0
        for city in self.cities:
            data = self.data.loc[self.data["city"] == city, columns].set_index("distance")
            metro = self.metro.loc[self.metro["city"] == city]
            
            # Skip city whose metro station number is less than the thresold
            if metro["FREQUENCY"].iloc[0] < threshold:
                continue

            # Plot curve
            if i == 0:
                data.plot(ax=axs[i], marker='o', title=city, ylabel="ratio")
                axs[i].set_xlabel("distance")
                axs[i].set_yticks(yticks)
                lines, labels = fig.axes[i].get_legend_handles_labels()
                fig.axes[i].get_legend().remove()
                i = 1
                continue
            data.plot(ax=axs[i], marker='o', title=city, ylabel="ratio", legend=False)
            axs[i].set_xlabel("distance")
            axs[i].set_yticks(yticks)
            i += 1
        
        # Hide unused subplots
        for j in range(i, len(axs)):
            fig.delaxes(axs[j])
        
        # Add one legend
        fig.legend(lines, labels, loc = "lower right")
        plt.tight_layout()
        plt.savefig(path + '\\multiple_dataframes_plot.png', bbox_inches='tight')

        return

# Analysis using saved output csv file
class analysisAll(analysis):
    def __init__(self, metro: pd.DataFrame, data: pd.DataFrame):
        super().__init__(metro)
        self.data = data

def runAnalysis(a: analysis | analysisAll, path: str) -> None:
    a.drawCurve(path, 5)
    
    return

if __name__ == "__main__":
    # First round analysis
    for country in ["CH", "EU"]:
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
            dataPath = path + country + "_CaR_" + stationType + ".csv"
            data = pd.read_csv(dataPath, encoding="utf-8")
            a.addData(data, stationType)
        
        # Merger all different data and save to one csv file
        a.merge(path + country + ".csv")

        # Run analysis method
        runAnalysis(a, path)
    
    # Analysis using saved csv file
    for country in ["US"]:
        path = "..\\Export\\" + country + "\\"
        metroPath = path + country + "_Metro.csv"
        dataPath = path + country + ".csv"
        metro = pd.read_csv(metroPath, encoding="utf-8")
        data = pd.read_csv(dataPath, encoding="utf-8")
        a = analysisAll(metro, data)
        
        # Run analysis method
        runAnalysis(a, path)