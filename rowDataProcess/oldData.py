import os
import pandas as pd

class oldData:
    def __init__(self):
        pass

    def merge(self, path: str, place: str) -> None:
        if place == "CH":
            result = pd.DataFrame({"city": [], "ID": [], "Station": [], "Lat": [], "Lon": []})
        elif place in ["US", "EU"]:
            result = pd.DataFrame({"city": [], "地铁站名": [], "线路名称": [], "是否是换乘站": [], "Lat": [], "Lon": []})
        else:
            raise ValueError("Wrnng place input!")
        
        files = os.listdir(path)
        for file in files:
            if file[-4:] != "xlsx":
                continue
            filePath = os.path.join(path, file)
            city = pd.read_excel(filePath)
            if place == "CH":
                city["city"] = file[:-5] + "市"
            else:
                city["city"] = file[:-5]
            result = pd.concat([result, city])
        result.to_csv(os.path.join(path, "2022" + place + "Old.csv"), encoding="utf-8", index=False)

        return

if __name__ == "__main__":
    a = oldData()
    a.merge(u"__SampleData\\GlobalMetroOld\\CH", "CH")