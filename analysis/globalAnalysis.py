import pandas as pd

from function import CITY_STANDER

class analysis:
    metro = pd.DataFrame()
    data = pd.DataFrame()
    cities = []
    yticks = [0, 0.5, 1] # Unify y-axis

    def __init__(self, metro: pd.DataFrame = metro, data: pd.DataFrame = data):
        if not metro.empty:
            # metro & data change city name using CITY_STANDER
            metro["city"] = metro["city"].map(CITY_STANDER())
            data["city"] = data["city"].map(CITY_STANDER())
            self.cities += metro["city"].unique().tolist()
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
            result.to_csv(path, encoding="utf-8", index=False)

        return result

if __name__ == "__main__":
    a = analysis()
    
    for country in ["US", "CH", "EU"]:
        path = "..\\Export\\" + country + "\\"
        metroPath = path + country + "_Metro.csv"
        dataPath = path + country + ".csv"
        metro = pd.read_csv(metroPath, encoding="utf-8")
        data = pd.read_csv(dataPath, encoding="utf-8")
        a.append(metro, data)
    
    a.compareRatio([500, 1000], ["All", "Normal", "Terminal", "Trans"], "..\\Export\\global.csv", 5)