import pandas as pd
import numpy as np

from function import CITY_STANDER

class analysis:
    metro = pd.DataFrame()
    data = pd.DataFrame()
    cities = []
    yticks = [0, 0.5, 1] # Unify y-axis

    def __init__(self, metro: pd.DataFrame = metro, data: pd.DataFrame = data):
        self.metro = pd.concat([self.metro, metro])
        self.data = pd.concat([self.data, data])
        if not metro.empty:
            # metro & data change city name using CITY_STANDER
            self.metro["city"] = self.metro["city"].map(CITY_STANDER())
            self.cities += self.metro["city"].unique().tolist()
            self.data.sort_values(by=["city", "distance"], inplace=True)
            
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
    
    def calculateAccumulation(self, path: str, interval: int = 10, threshold: int = 2000, sub: str = "") -> None:
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

if __name__ == "__main__":
    a = analysis()
    
    for country in ["US", "CH", "EU"]:
        path = "..\\Export\\" + country + "\\"
        metroPath = path + country + "_Metro.csv"
        dataPath = path + country + ".csv"
        metro = pd.read_csv(metroPath, encoding="utf-8")
        data = pd.read_csv(dataPath, encoding="utf-8")
        a.append(metro, data)
    
    # a.compareRatio([500, 1000], ["All", "Normal", "Terminal", "Trans"], "..\\Export\\global.csv", 5)
    # a.calculateAccumulation("..\\Export\\Accumulation.csv")
    a.calculateAccumulation("..\\Export\\Accumulation500.csv", threshold=500, sub="PaR")
    a.calculateAccumulation("..\\Export\\Accumulation1000.csv", threshold=1000, sub="PaR")