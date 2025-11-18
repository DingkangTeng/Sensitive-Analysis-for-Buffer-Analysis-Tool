# Amap metro: https://map.amap.com/subway/index.html?&4401
# Amap json: https://map.amap.com/service/subway?_1734508912342&srhdata=1100_drw_beijing.json
import time, os, copy, ast
import pandas as pd

from _toolClass.crawler import crawler
from _toolClass.coordTransform import gcj02_to_wgs84
from _toolClass.rightRound import rightRound

class getAllMetro(crawler):
    __result = {"city": [], "lineName": [], "lineShortName": [], "lineID": [],
              "station": [], "whetherTrans": [], "transStation": [],
              "lat": [], "long": []}
    csv = pd.DataFrame(__result)

    def __init__(self):
        pass

    def getTime(self) -> int:
        millis = rightRound(time.time() * 10000)
        return int(millis)
    
    def getAllCities(self, name: str, path: str = "") -> None:
        cityListUrl = "https://map.amap.com/service/subway?_{}&srhdata=citylist.json".format(self.getTime())
        super().__init__(cityListUrl)
        data = self.rget().json()
        cityList = data["citylist"]
        for city in cityList:
            adcode = city["adcode"]
            cityName = city["spell"]
            url = "https://map.amap.com/service/subway?_{}&srhdata={}_drw_{}.json".format(self.getTime(), adcode, cityName)
            result = self.getOneCity(url)

            #Save in one file
            self.csv = pd.concat([self.csv, result])
        
        #Save in one file
        self.csv.to_csv(os.path.join(path, name + "CH.csv"), encoding="utf-8", index=False)
        self.csv.to_csv(os.path.join(path, name + "CHGBK.csv"), encoding="ansi", index=False)

        return
    
    def getOneCity(self, url: str) -> pd.DataFrame:
        super().__init__(url)
        data = self.rget().json()
        city = data["s"]
        lines = data["l"]
        result = copy.deepcopy(self.__result)

        # Get line
        for line in lines:
            lineName = line["kn"]
            lineSortName = line["ln"]
            lineID = line["ls"]
            stations = line["st"]
            num = 0

            for station in stations:
                result["city"].append(city)
                result["lineName"].append(lineName)
                result["lineShortName"].append(lineSortName)
                result["lineID"].append(lineID)

                stationName = station['n']
                wheterTrans = '0'
                allLine = station['r'].split("|")
                if lineID in allLine:
                    allLine.remove(lineID)
                if len(allLine) > 0:
                    wheterTrans = '1'
                location = station["sl"].split(',')
                coordinate = gcj02_to_wgs84(float(location[0]), float(location[1]))

                result["station"].append(stationName)
                result["whetherTrans"].append(wheterTrans)
                result["transStation"].append(allLine)
                result["long"].append(coordinate[0])
                result["lat"].append(coordinate[1])
        
        return pd.DataFrame(result)
    
def drop(file: str, path: str) -> None:
    data = pd.read_csv(file, dtype=str)
    allLine = data[["lineName", "lineID"]].copy()
    # allLine.loc[allLine.shape[0]] = ["磁悬浮", "310100104512"]
    allLine.loc[allLine.shape[0]] = ["屯马线", "810000020819"]
    allLine.loc[allLine.shape[0]] = ["轨道交通5号线", "900000070367"]
    allLine.loc[allLine.shape[0]] = ["轨道交通11号线（上海/昆山）", "320500022611"]
    allLine.drop_duplicates(inplace=True)
    data["wheterTerminal"] = '0'
    for i in range(data.shape[0]):
        lines = ast.literal_eval(data["transStation"][i])
        result = []
        for line in lines:
            exception = [
                "900000086378", "900000113361", "900000079032", "120100021326", "500100021567",
                "900000076252", "210200031348", "900000143028", "900000211171"
            ]
            if line in exception:
                continue
            tmp = allLine.loc[allLine["lineID"] == line]
            result.append(tmp.iloc[0]["lineName"])
        result.append(data["lineName"][i])
        data["transStation"][i] = ", ".join(result)
    
    data = data.copy()
    data.iloc[0]["wheterTerminal"] = '1'
    i = 1
    for i in range(1, data.shape[0]):
        a = data["lineName"][i]
        b = data["lineName"][i - 1]
        if a != b:
            data.iloc[i]["wheterTerminal"] = '1'
            data.iloc[i - 1]["wheterTerminal"] = '1'
    data.iloc[i]["wheterTerminal"] = '1'
    
    data["UID"] = data["city"] + data["station"]
    data.to_csv(path + ".csv", encoding="utf-8", index=False)
    data.to_csv(path + "GBK.csv", encoding="ansi", index=False)

# Change all station with the same name
def ter(file: str, path: str) -> None:
    data = pd.read_csv(file, dtype=str)
    cities = data[["city"]].copy().drop_duplicates()["city"].to_list()
    for city in cities:
        stations = data.loc[data["city"] == city]["station"].to_list()
        for station in stations:
            tmp = data.loc[(data["city"] == city) & (data["station"] == station)]
            for i in tmp["wheterTerminal"].to_list():
                if i == "1":
                    data["wheterTerminal"].loc[(data["city"] == city) & (data["station"] == station)] = '1'
                    break
    
    data.to_csv(path + ".csv", encoding="utf-8", index=False)
    data.to_csv(path + "GBK.csv", encoding="ansi", index=False)

if __name__ == "__main__":
    a = getAllMetro().getAllCities("2024", "__SampleData")
    drop("__SampleData\\2024CH.csv", "__SampleData\\2024CH-Processed")
    ter("__SampleData\\2024CH-Processed.csv", "__SampleData\\2024CH-Processed")