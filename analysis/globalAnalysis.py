import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib import ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

from function import (
    CITY_STANDER, COLOR, STANDER_NAME, adjustBrightness,
    TITLE_FONT, TICK_FONT, TICK_FONT_INT, MARK_FONT, MARK_FONT_INT
)

LEGEND_TITLE = {
    "US": "U.S.",
    "EU": "Europe",
    "CN": "China"
}

class analysis:
    metro = pd.DataFrame()
    data = pd.DataFrame()
    cities = []
    yticks = [0, 0.5, 1] # Unify y-axis
    DROP_DATA = ["San Juan", u"香港特别行政区", u"澳门特别行政区"] # Delete San Juan, Hong Kong and Macao

    def __init__(self, metro: pd.DataFrame = metro, data: pd.DataFrame = data):
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.sans-serif"] = "Sans Serif Collection"
        plt.rcParams["legend.facecolor"] = "white"
        plt.rcParams["legend.edgecolor"] = "lightgray"
        plt.rcParams['legend.frameon'] = True
        plt.rcParams["legend.framealpha"] = 1.0
        if not metro.empty:
            # metro & data change city name using CITY_STANDER
            metro["city"] = metro["city"].map(CITY_STANDER())
            self.cities += metro["city"].unique().tolist()
            data.sort_values(by=["city", "distance"], inplace=True)
        self.metro = pd.concat([self.metro, metro])
        self.data = pd.concat([self.data, data])
            
    def append(self, metro: pd.DataFrame, data: pd.DataFrame) -> None:
        data = data.drop(data.loc[data["city"].isin(self.DROP_DATA)].index) # Delete San Juan, Hong Kong and Macao
        metro = metro.drop(metro.loc[metro["city"].isin(self.DROP_DATA)].index) # Delete San Juan, Hong Kong and Macao
        self.__init__(metro, data)

        return
    
    @staticmethod
    def customLegend(tag: list[str]) -> list[Patch | Line2D]:
        customLegend = []
        tag.sort(reverse=True)
        for i in tag:
            customLegend.append(
                Patch(color=COLOR[i], label=LEGEND_TITLE.get(i, i))
            )
        
        return customLegend

    def compareRatio(self, distance: list[int], typelist: list[str] | str, path: str = "", threshold: int = 0) -> pd.DataFrame:
        # Initialize statice dataframe
        result = pd.DataFrame({"city": self.cities})
        if type(typelist) is list:
            typestr = typelist[-1]
        else:
            assert typelist is str
            typestr = typelist
        
        col = "ratio" + typestr
        for i in distance:
            result[col + str(i)] = 0.0

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
    
    # Distributuin Plot
    def distributionPlot(self, path: str, areas: list[str], compare: str, distance: int = 500, threshold: int = 0) -> None:
        cities = self.metro.loc[self.metro["FREQUENCY"] >= threshold, "city"].to_list()
        data = self.data.loc[(self.data["city"].isin(cities)) & (self.data["distance"] == distance)].copy()
        
        plt.figure(figsize=(10, 10))
        ax = plt.axes()
        for area in areas:
            subData = data.loc[data["city"].str[0:2] == area].copy()
            color = COLOR.get(area)
            subData.plot.scatter("ratioAll", compare, color=color, label=LEGEND_TITLE.get(area, area), ax=ax)
            # Draw boundary
            hull = ConvexHull(subData[["ratioAll", compare]].values)
            # Get the hull points
            hullPoints = subData[["ratioAll", compare]].values[hull.vertices]
            # Close the boundary by appending the first point to the end
            hullPoints = np.vstack([hullPoints, hullPoints[0]])
            # Create a B-spline representation of the hull points
            tck, u = splprep(hullPoints.T, s=0.002)  # s=0 means no smoothing
            xSmooth, ySmooth = splev(np.linspace(0, 1, 100), tck)
            # Fill the area inside the convex hull
            plt.fill(xSmooth, ySmooth, color=color, alpha=0.2)
            # Plot the smooth boundary
            plt.plot(xSmooth, ySmooth, color=color, alpha=0.2)
        
        # Set Plots
        # 1:1 Line
        ax.set_xlim(-0.05,1.05)
        ax.set_ylim(-0.05,1.05)
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_xlabel("% of EVCS", fontdict=MARK_FONT)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(1,0))
        ax.tick_params(axis='x', labelsize=TICK_FONT_INT)
        tmp={"ratioAll_Baseline": "% of buffer areas", "ratioAll_PaR": "% of parking lots"}
        ax.set_ylabel(tmp[compare], fontdict=MARK_FONT)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1,0))
        ax.tick_params(axis='y', labelsize=TICK_FONT_INT)
        plt.legend(loc="upper right", fontsize=TICK_FONT_INT)

        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

    # Distributuin Plot ( metro station number )
    def distributionPlot_Num(self, path: str, areas: list[str], distance: int = 500, threshold: int = 0) -> None:
        cities = self.metro.loc[self.metro["FREQUENCY"] >= threshold, "city"].to_list()
        data = self.data.loc[(self.data["city"].isin(cities)) & (self.data["distance"] == distance)].copy()
        data = data.join(self.metro.set_index("city"), on="city")
        data["FREQUENCY"] = data["FREQUENCY"] / distance
        
        plt.figure(figsize=(10, 10))
        ax = plt.axes()
        for area in areas:
            subData = data.loc[data["city"].str[0:2] == area].copy()
            color = COLOR.get(area)
            subData.plot.scatter("ratioAll", "FREQUENCY", color=color, label=LEGEND_TITLE.get(area, area), ax=ax)
            # Draw boundary
            hull = ConvexHull(subData[["ratioAll", "FREQUENCY"]].values)
            # Get the hull points
            hullPoints = subData[["ratioAll", "FREQUENCY"]].values[hull.vertices]
            # Close the boundary by appending the first point to the end
            hullPoints = np.vstack([hullPoints, hullPoints[0]])
            # Create a B-spline representation of the hull points
            tck, u = splprep(hullPoints.T, s=0.001)  # s=0 means no smoothing
            xSmooth, ySmooth = splev(np.linspace(0, 1, 100), tck)
            # Fill the area inside the convex hull
            plt.fill(xSmooth, ySmooth, color=color, alpha=0.2)
            # Plot the smooth boundary
            plt.plot(xSmooth, ySmooth, color=color, alpha=0.2)
        
        # Set Plots
        ax.set_xlabel("% of EVCS", fontdict=MARK_FONT)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        ax.set_xlim((-0.02, 1.02))
        ax.set_ylim((-0.02, 1.02))
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(1,0))
        ax.set_ylabel("Number of metro stations", fontdict=MARK_FONT)
        ax.tick_params(axis='x', labelsize=TICK_FONT_INT)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticklabels(["0", "100", "200", "300", "400", "500"], fontdict=TICK_FONT)
        plt.legend(loc="upper right", fontsize=TICK_FONT_INT)

        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

    def drawGolbalBar(self, path: str, typelist: list[str], sub :str, distances: int = 500, threshold: int = 0) -> None:
        fullList = typelist + [x + sub for x in typelist]
        result = self.compareRatio([distances], fullList, threshold=threshold)
        result["country"] = result["city"].str[0:2]
        countries = result["country"].unique().tolist()
        result["color"] = result["country"].replace(COLOR)
        result.set_index("city", inplace=True)
        types = ["ratio" + x + str(distances) for x in typelist]
        typesLen = len(types)

        # Sort
        result.sort_values(by=["country", types[0]], ascending=(False, False), inplace=True)
        result = result.join(self.metro.set_index("city"), on="city")

        # Plot
        fig, axs = plt.subplots(typesLen, 1, figsize=(20, 5 * typesLen), sharex=True, sharey=True)
        axs = axs.flatten() # Flatten the 2D array of axes to 1D for easy indexing
        ## Prepared for high light different type with different count
        highLight = ["FREQUENCY"] * typesLen
        for i in range(typesLen):
            result2 = result.copy()
            # Count metro station num
            # Change the brightness of the data after top ten in each area
            topTenIndexs = pd.Index([], name="city")
            for country in countries:
                topTenIndex = result2.loc[result2["country"] == country].nlargest(5, highLight[i]).index
                topTenIndexs = topTenIndexs.append(topTenIndex)
            result2.loc[~result2.index.isin(topTenIndexs), "color"] = result2.loc[~result2.index.isin(topTenIndexs), "color"].apply(lambda c: adjustBrightness(c, 0.6))
            result2[types[i]].plot.bar(width=0.8, ax=axs[i], color=result2["color"].tolist())

            # Add parking
            result2[types[i].replace(str(distances),"") + sub + str(distances)].plot(ax=axs[i], marker='.', color="black")

            axs[i].set_yticks(self.yticks)
            axs[i].tick_params(axis='y', labelsize=TICK_FONT_INT)
            axs[i].set_xlabel(None)
            axs[i].set_xticks([])
            axs[i].set_title("({}) {}".format(i + 1, STANDER_NAME.get(types[i])), fontdict=TITLE_FONT) # unicode 97 is a

        # Add legend and fig text
        customLegend = self.customLegend(countries)
        customLegend.append(
            Line2D([0], [0], marker='.', color="black", label=STANDER_NAME.get(sub))
        )
        plt.legend(handles=customLegend, bbox_to_anchor=(0.5, -0.3), loc=8, ncol = len(countries) + 1, fontsize=MARK_FONT_INT)
        plt.xlabel("Study unit", fontsize=MARK_FONT_INT)
        fig.supylabel("%  of  EVCS or parking lot", fontsize=MARK_FONT_INT, x=-0.0000001)
        
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

        return
    
    # Draw separately by location
    def drawGlobalBoxplot(self, path: str, typelist: list[str], sub :str, distances: int = 500, threshold: int = 0, ylim: float = 0.8) -> None:
        fullList = typelist + [x + sub for x in typelist]
        result = self.compareRatio([distances], fullList, threshold=threshold)
        result["country"] = result["city"].str[0:2]
        countries = result["country"].unique().tolist()

        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        axs = axs.flatten() # Flatten the 2D array of axes to 1D for easy indexing
        
        axsN = 0
        for i in typelist:
            result2 = []
            columns = ["ratio" + i + str(distances), "ratio" + i + sub + str(distances)]
            for country in countries:
                subdata = result.loc[result["country"] == country, columns].copy()
                subdata.columns = pd.MultiIndex.from_arrays([[country] * 2, ["EVCS", "Parking"]])
                result2.append(subdata)
            result2 = pd.concat(result2, axis=1)
            result2.sort_index(axis=1, ascending=(False, True), inplace=True)
            bplot = result2.boxplot(
                ax=axs[axsN],
                patch_artist=True,
                showfliers=False,
                whiskerprops={"linewidth": 1.5},
                capprops={"linewidth": 1.5},
                medianprops={"color": "lime"},
            )

            # Modify the label
            ## Modify the main label
            xticks = []
            for x in result2.columns:
                if x[0] not in xticks:
                    xticks.append(x[0])
            topXTicks = [x[1] for x in result2.columns]
            axs[axsN].set_title("({}) {}".format(axsN + 1, STANDER_NAME.get(i)), fontdict=TITLE_FONT) # unicode 97 is a
            ## Add a top label for areas
            if axsN == 0 or axsN == 1:
                ax2 = axs[axsN].twiny() # Create a twin Axes sharing the y-axis
                ax2.set_xticks(axs[axsN].get_xticks())
                ax2.set_xticklabels(topXTicks, fontdict=TICK_FONT)
                ax2.set_xlim(axs[axsN].get_xlim()) # Set the start location of new top ax
                ax2.xaxis.set_ticks_position('top')
                ax2.xaxis.set_label_position('top')
            ## Change bottom xticks
            axs[axsN].set_xticks([1.5 + x + x for x in range(len(countries))])
            if axsN == 2 or axsN == 3:
                axs[axsN].set_xticklabels([LEGEND_TITLE.get(x, x) for x in xticks], fontdict=TICK_FONT)
            else:
                axs[axsN].set_xticklabels([None] * len(xticks))
            
            axs[axsN].set_ylim(0, ylim)

            # Chage color:
            color = [COLOR[x[0]] for x in result2.columns]
            adjColor = [adjustBrightness(x, 0.6) for x in color]
            ## Change other color
            children = bplot.get_children()
            for i, box in enumerate(children):
                INTERVAL = 6 # How many color in one group, it depends
                if i % INTERVAL in [1, 2, 3, 4] and i // INTERVAL < len(color):  # 1, 2 are the bottom and top tails, and 3, 4 are bottom and top tails end
                    box.set_color(color[i // INTERVAL])  # Set the color for each cap
            ## Change box and its edge color
            for i, patch in enumerate(bplot.patches):
                # Box and its edge color
                if i % 2 == 0:
                    patch.set(facecolor=color[i], edgecolor=color[i])
                else:
                    patch.set(hatch='///', facecolor=adjColor[i], edgecolor=color[i]) # Add /// hatch
                # Background color
                rect = patches.Rectangle(
                    (i + 0.5, axs[axsN].get_ylim()[0]), # The anchor point
                    1, # Rectangle width
                    1.1, # Rectangle height # get automatically: axs[axsN].get_ylim()[1] - axs[axsN].get_ylim()[0]
                    color=color[i],
                    alpha=0.1 # Adjust alpha for transparency
                )
                axs[axsN].add_patch(rect)

            # Add y label
            n = int(ylim * 10 + 1)
            axs[axsN].set_yticks([x/10 for x in range(n)]) # y axis 0-0.8
            axs[axsN].tick_params(axis='y', labelsize=MARK_FONT_INT)
            if axsN == 0 or axsN == 2:
                axs[axsN].set_ylabel("% of EVCS or parking lots", fontdict=MARK_FONT)
            else:
                axs[axsN].set_yticklabels([None] * n)
            axsN += 1

        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

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
    
    allList = ["All", "Normal", "Terminal", "Trans"]
    for i in allList.copy():
        allList.append(i + "_PaR")
        allList.append(i + "_Baseline")
    # a.compareRatio([500], allList, "..\\Export\\global500.csv", 7) # Compaer ratio and export csv, not used in paper
    
    # Figure 6
    a.drawGolbalBar("..\\Export\\G-bar.jpg", ["All", "Normal", "Terminal", "Trans"], "_PaR", threshold=7)
    # Figure 7
    a.drawGlobalBoxplot("..\\Export\\G-box.jpg", ["All", "Normal", "Terminal", "Trans"], "_PaR", threshold=7, ylim=0.8)
    # Figures 8
    a.distributionPlot("..\\Export\\G-generlaDistribution_PaR.jpg", ["US", "CN", "EU"], "ratioAll_PaR", threshold=7)
    # Figures 9
    a.distributionPlot_Num("..\\Export\\G-generlaDistribution_number.jpg", ["US", "CN", "EU"], threshold=7)
    # Figures 10
    a.distributionPlot("..\\Export\\G-generlaDistribution.jpg", ["US", "CN", "EU"], "ratioAll_Baseline", threshold=7)
   