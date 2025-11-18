import os
import pandas as pd

GISDATA = r"__SampleData"

class analysis:
    def __init__(self, newData: pd.DataFrame, oldData: pd.DataFrame):
        self.newData = newData
        self.oldData = oldData
        self.oldData["chart"] = oldData["city"].str.split('-', expand=True)[0]
        self.citis = self.newData["chart"].drop_duplicates()
    
    def ratio(self, path: str) -> None:
        result = pd.DataFrame({
            "city": [], "totalStop": [],
            "transStop": [], "transRatio": [],
            "terminalStop": [], "terminalRatio": []
        })
        for city in self.citis:
            data = self.newData.loc[self.newData["chart"] == city]
            totalStop = data.shape[0]
            transStop = data.loc[data["whetherTrans"] == 1].shape[0]
            terminalStop = data.loc[(data["wheterTerminal"] == 1) | (data["wheterTerminal"] == '1')].shape[0]
            tmp = pd.DataFrame({
                "city": [city], "totalStop": [totalStop],
                "transStop": [transStop], "transRatio": [str(round(transStop/totalStop, 4) * 100) + "%"],
                "terminalStop": [terminalStop], "terminalRatio": [str(round(terminalStop/totalStop, 4)) + "%"]
            })
            result = pd.concat([result, tmp])

        result.to_csv(path, encoding="ansi", index=False)

        return

    def compare(self, path: str) -> None:
        result = pd.DataFrame({"city": [], "old": [], "new": [], "difference": []})
        for city in self.citis:
            newData = self.newData.loc[self.newData["chart"] == city].shape[0]
            oldData = self.oldData.loc[self.oldData["chart"] == city].shape[0]
            tmp = pd.DataFrame({"city": [city], "old": [oldData], "new": [newData], "difference": [newData - oldData]})
            result = pd.concat([result, tmp])
        
        result.to_csv(path, encoding="ansi", index=False)

        return

if __name__ == "__main__":
    for PLACE in ["CH", "EU", "US"]:
        newData = pd.read_csv(os.path.join(GISDATA, "2024"+ PLACE + "-Processed.csv"))
        oldData = pd.read_csv(os.path.join(GISDATA, "GlobalMetroOld", PLACE, "2022" + PLACE + "Old.csv"))
        a = analysis(newData, oldData)
        a.ratio(os.path.join(GISDATA, "Analysis", PLACE + "-ratio.csv"))
        a.compare(os.path.join(GISDATA, "Analysis", PLACE + "-compare.csv"))