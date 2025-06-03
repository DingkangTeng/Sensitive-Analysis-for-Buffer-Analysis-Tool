import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib import ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

from function import CITY_STANDER, COLOR, STANDER_NAME, adjustBrightness, \
    TITLE_FONT, TICK_FONT, TICK_FONT_INT, MARK_FONT, MARK_FONT_INT


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
    
    def customLegend(self, tag: list[str]) -> list[Patch]:
        customLegend = []
        tag.sort(reverse=True)
        for i in tag:
            customLegend.append(
                Patch(color=COLOR[i], label=i)
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
    
    # def calculateAccumulation(self, path: str, interval: int = 10, threshold: int = 500, sub: str = "") -> None:
    #     def formular(sequencesA: list[int | float], sequencesB: list[int | float], interval: int) -> float:
    #         segmaA = np.sum(sequencesA[:-1])
    #         resultA = (segmaA + 0.5 * sequencesA[-1]) * interval
    #         segmaB = np.sum(sequencesB[:-1])
    #         resultB = (segmaB + 0.5 * sequencesB[-1]) * interval
    #         if resultA == 0 and resultB == 0:
    #             return None
    #         else:
    #             return np.around(resultA - resultB, 4)
        
    #     # Initialize results
    #     cities = self.data["city"].unique().tolist()
    #     cities.sort()
    #     cityNum = len(cities)
    #     result = pd.DataFrame({"city": cities,
    #                            "Accumulation of all stations": [0] * cityNum,
    #                            "Accumulaiton of normal stations": [0] * cityNum,
    #                            "Accululation of interchange stations": [0] * cityNum,
    #                            "Accumulation of terminal stations": [0] * cityNum
    #                         })
    #     if sub != "":
    #         result["Accumulation of all stations_" + sub] = 0
    #         result["Accumulaiton of normal stations_" + sub] = 0
    #         result["Accululation of interchange stations_" + sub] = 0
    #         result["Accumulation of terminal stations_" + sub] = 0
        
    #     for i in range(cityNum):
    #         data = self.data.loc[(self.data["city"] == cities[i]) & (self.data["distance"] <= threshold)]
    #         result.loc[i, "Accumulation of all stations"] = formular(data["ratioAll"].to_list(), data["ratioAll_Baseline"].to_list(), interval)
    #         result.loc[i, "Accumulaiton of normal stations"] = formular(data["ratioNormal"].to_list(), data["ratioNormal_Baseline"].to_list(), interval)
    #         result.loc[i, "Accululation of interchange stations"] = formular(data["ratioTerminal"].to_list(), data["ratioTerminal_Baseline"].to_list(), interval)
    #         result.loc[i, "Accumulation of terminal stations"] = formular(data["ratioTrans"].to_list(), data["ratioTrans_Baseline"].to_list(), interval)
    #         if sub != "":
    #             result.loc[i, "Accumulation of all stations_" + sub] = formular(data["ratioAll_" + sub].to_list(), data["ratioAll_Baseline"].to_list(), interval)
    #             result.loc[i, "Accumulaiton of normal stations_" + sub] = formular(data["ratioNormal_" + sub].to_list(), data["ratioNormal_Baseline"].to_list(), interval)
    #             result.loc[i, "Accululation of interchange stations_" + sub] = formular(data["ratioTerminal_" + sub].to_list(), data["ratioTerminal_Baseline"].to_list(), interval)
    #             result.loc[i, "Accumulation of terminal stations_" + sub] = formular(data["ratioTrans_" + sub].to_list(), data["ratioTrans_Baseline"].to_list(), interval)
        
    #     result.sort_values(by=["city"]).to_csv(path, encoding="utf-8", index=False)

    #     return
    
    # Distributuin Plot
    def distributionPlot(self, path: str, areas: list[str], compare: str, distance: int = 500, threshold: int = 0) -> None:
        cities = self.metro.loc[self.metro["FREQUENCY"] >= threshold, "city"].to_list()
        data = self.data.loc[(self.data["city"].isin(cities)) & (self.data["distance"] == distance)].copy()
        
        plt.figure(figsize=(10, 10))
        ax = plt.axes()
        for area in areas:
            subData = data.loc[data["city"].str[0:2] == area].copy()
            color = COLOR.get(area)
            subData.plot.scatter("ratioAll", compare, color=color, label=area, ax=ax)
            # Draw boundary
            hull = ConvexHull(subData[["ratioAll", compare]].values)
            # # Fill the area inside the convex hull
            # plt.fill(subData[["ratioAll"]].values[hull.vertices], subData[["MTRPer"]].values[hull.vertices], color=color, alpha=0.2)
            # # Plot the hull edges
            # for simplex in hull.simplices:
            #     plt.plot(subData[["ratioAll"]].values[simplex], subData[["MTRPer"]].values[simplex], color=color, alpha=0.2)
            # Draw boundary in smooth
            # Get the hull points
            hullPoints = subData[["ratioAll", compare]].values[hull.vertices]
            # Close the boundary by appending the first point to the end
            hullPoints = np.vstack([hullPoints, hullPoints[0]])
            # Create a B-spline representation of the hull points
            tck, u = splprep(hullPoints.T, s=0.0001)  # s=0 means no smoothing
            xSmooth, ySmooth = splev(np.linspace(0, 1, 100), tck)
            # Fill the area inside the convex hull
            plt.fill(xSmooth, ySmooth, color=color, alpha=0.2)
            # Plot the smooth boundary
            plt.plot(xSmooth, ySmooth, color=color, alpha=0.2)
        
        # Set Plots
        # 1:1 Line
        # ax.set_xlim((-0.05,1.05))
        # ax.set_ylim((-0.05,1.05))
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
        plt.legend(loc=4, fontsize=TICK_FONT_INT)

        # plt.show()
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
            subData.plot.scatter("ratioAll", "FREQUENCY", color=color, label=area, ax=ax)
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
        ax.set_yticklabels([0, 100, 200, 300, 400, 500], fontdict=TICK_FONT)
        plt.legend(loc=4, fontsize=TICK_FONT_INT)

        # plt.show()
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
        fig, axs = plt.subplots(typesLen, 1, figsize=(20, 3 * typesLen))
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
            result2.loc[~result2.index.isin(topTenIndexs), "color"] = result2["color"].apply(lambda c: adjustBrightness(c, 0.6))
            result2[types[i]].plot.bar(width=0.8, ax=axs[i], color=result2["color"])

            # Add parking
            result2[types[i].replace(str(distances),"") + sub + str(distances)].plot(ax=axs[i], marker='.', color="black")

            axs[i].set_yticks(self.yticks)
            axs[i].tick_params(axis='y', labelsize=TICK_FONT_INT)
            axs[i].set_xlabel(None)
            axs[i].set_xticks([])
            axs[i].set_title("({}) {}".format(chr(i+97), STANDER_NAME.get(types[i])), fontdict=TITLE_FONT) # unicode 97 is a

        # Add legend and fig text
        customLegend = self.customLegend(countries)
        customLegend.append(
            Line2D([0], [0], marker='.', color="black", label=STANDER_NAME.get(sub))
        )
        plt.legend(handles=customLegend, bbox_to_anchor=(0.5, -0.4), loc=8, ncol = len(countries) + 1)
        fig.text(0.1, 0.5, "%  of  EVCS or parking lot",
                 fontsize=MARK_FONT_INT, verticalalignment="center", horizontalalignment="right", rotation=90)
        fig.text(0.5, 0.1, "Study areas",
                 fontsize=MARK_FONT_INT, verticalalignment="top", horizontalalignment="center")
        
        # plt.show()
        plt.savefig(path, dpi=300)
        plt.close()

        return
    
    # Draw separately by location
    def drawGlobalBoxplot2(self, path: str, typelist: list[str], sub :str, distances: int = 500, threshold: int = 0) -> None:
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
                subdata.columns = [[country] * 2, ["EVCS", "Parking"]]
                result2.append(subdata)
            result2 = pd.concat(result2, axis=1)
            result2.sort_index(axis=1, ascending=(False, True), inplace=True)
            bplot = result2.boxplot(
                ax=axs[axsN],
                patch_artist=True,
                showfliers=False,
                # boxprops={"linewidth": 0},
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
            axs[axsN].set_title("({}) {}".format(chr(axsN+97), STANDER_NAME.get(i)), fontdict=TITLE_FONT) # unicode 97 is a
            ## Add a top label for areas
            ax2 = axs[axsN].twiny() # Create a twin Axes sharing the y-axis
            ax2.set_xticks(axs[axsN].get_xticks())
            ax2.set_xticklabels(topXTicks, fontdict=TICK_FONT)
            ax2.set_xlim(axs[axsN].get_xlim()) # Set the start location of new top ax
            ax2.xaxis.set_ticks_position('top')
            ax2.xaxis.set_label_position('top')
            ## Change bottom xticks
            axs[axsN].set_xticks([1.5 + x + x for x in range(len(countries))])
            axs[axsN].set_xticklabels(xticks, fontdict=TICK_FONT)

            # Chage color:
            color = [COLOR.get(x[0]) for x in result2.columns]
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
            axs[axsN].set_ylabel("% of EVCS or parking lots", fontdict=MARK_FONT)
            axs[axsN].set_yticks([x/10 for x in range(9)]) # y axis 0-0.8
            axs[axsN].tick_params(axis='y', labelsize=TICK_FONT_INT)
            axsN += 1

        # Add legend
        customLegend = self.customLegend(countries)
        plt.legend(handles=customLegend, bbox_to_anchor=(0.1, -0.1), loc=0, ncol=3, fontsize=TICK_FONT_INT)

        # plt.show()
        plt.savefig(path, dpi=300)
        plt.close()

        return
    
    # # Draw separately by type
    # def drawGlobalBoxplot(self, path: str, typelist: list[str], sub :str, distances: int = 500, threshold: int = 0) -> None:
    #     fullList = typelist + [x + sub for x in typelist]
    #     result = self.compareRatio([distances], fullList, threshold=threshold)
    #     result["country"] = result["city"].str[0:2]
    #     countries = result["country"].unique().tolist()
    #     evcs = ["ratio" + x + str(distances) for x in typelist]
    #     parking = ["ratio" + x + sub + str(distances) for x in typelist]
    #     typelen = len(typelist)

    #     fig, axs = plt.subplots(2, 1, figsize=(20, 20))
        
    #     axsN = 0
    #     yLable = ["EVCS", "Parking"]
    #     for i in [evcs, parking]:
    #         result2 = []
    #         for country in countries:
    #             subdata = result.loc[result["country"] == country, i].copy()
    #             subdata.columns = [typelist, [country] * typelen]
    #             result2.append(subdata)
    #         result2 = pd.concat(result2, axis=1)
    #         result2.sort_index(axis=1, ascending=(True, False), inplace=True)
    #         bplot = result2.boxplot(
    #             ax=axs[axsN],
    #             patch_artist=True,
    #             showfliers=False,
    #             # boxprops={"linewidth": 0},
    #             whiskerprops={"linewidth": 1.5},
    #             capprops={"linewidth": 1.5}
    #         )

    #         # Modify the label
    #         ## Modify the main label
    #         xticks = [x[0] for x in result2.columns]
    #         topXTicks = [x[1] for x in result2.columns]
    #         group = len(countries)
    #         middle = (group - 1) // 2
    #         colorBack = []
    #         for i in range(len(xticks) // group):
    #             start = i * group
    #             for j in range(group):
    #                 if j != middle:
    #                     xticks[start + j] = None
    #         axs[axsN].set_xticklabels(xticks)
    #         ## Add a top label for areas
    #         ax2 = axs[axsN].twiny() # Create a twin Axes sharing the y-axis
    #         ax2.set_xticks(axs[axsN].get_xticks())
    #         ax2.set_xticklabels(topXTicks)
    #         ax2.set_xlim(axs[axsN].get_xlim()) # Set the start location of new top ax
    #         ax2.xaxis.set_ticks_position('top')
    #         ax2.xaxis.set_label_position('top')

    #         # Chage color:
    #         color = [COLOR.get(x[1]) for x in result2.columns]
    #         adjColor = [adjustBrightness(x, 0.8) for x in color]
    #         ## Background color is fixd for four group, if the group number exceed four, the index would out of range
    #         colorBack = ["lightblue"] * group + ["lightgreen"] * group + ["lightcoral"] * group + ["lightyellow"] * group
    #         ## Change other color
    #         children = bplot.get_children()
    #         for i, box in enumerate(children):
    #             INTERVAL = 6 # How many color in one group, it depends
    #             if i % INTERVAL in [1, 2, 3, 4] and i // INTERVAL < len(color):  # 1, 2 are the bottom and top tails, and 3, 4 are bottom and top tails end
    #                 box.set_color(color[i // INTERVAL])  # Set the color for each cap
    #         ## Change box and its edge color
    #         for i, patch in enumerate(bplot.patches):
    #             # Box and its edge color
    #             patch.set_facecolor(adjColor[i])
    #             patch.set_edgecolor(color[i])
    #             # Background color
    #             rect = patches.Rectangle(
    #                 (i + 0.5, axs[axsN].get_ylim()[0]), # The anchor point
    #                 1, # Rectangle width
    #                 1.1, # Rectangle height # get automatically: axs[axsN].get_ylim()[1] - axs[axsN].get_ylim()[0]
    #                 color=colorBack[i],
    #                 alpha=0.5 # Adjust alpha for transparency
    #             )
    #             axs[axsN].add_patch(rect)

    #         # Add y label
    #         axs[axsN].set_ylabel(yLable[axsN], fontdict=MARK_FONT)
    #         axs[axsN].set_yticks([x/10 for x in range(9)]) # y axis 0-0.8
    #         axsN += 1

    #     # Add legend
    #     customLegend = self.customLegend(countries)
    #     plt.legend(handles=customLegend, bbox_to_anchor=(0.5, -0.1), loc=8, ncol=group, fontsize=TICK_FONT_INT)

    #     # plt.show()
    #     plt.savefig(path, dpi=300)
    #     plt.close()

    #     return

if __name__ == "__main__":
    a = analysis()
    
    for country in ["US", "CH", "EU"]:
        path = "..\\Export\\" + country + "\\"
        metroPath = path + country + "_Metro.csv"
        dataPath = path + country + ".csv"
        metro = pd.read_csv(metroPath, encoding="utf-8")
        data = pd.read_csv(dataPath, encoding="utf-8")
        a.append(metro, data)
    
    # # a.calculateAccumulation("..\\Export\\Accumulation.csv")
    allList = ["All", "Normal", "Terminal", "Trans"]
    for i in allList.copy():
        allList.append(i + "_PaR")
        allList.append(i + "_Baseline")
    a.compareRatio([500], allList, "..\\Export\\global500.csv", 7)
    # a.distributionPlot("..\\Export\\G-generlaDistribution.jpg", ["US", "CN", "EU"], "ratioAll_Baseline", threshold=7)
    # a.distributionPlot("..\\Export\\G-generlaDistribution_PaR.jpg", ["US", "CN", "EU"], "ratioAll_PaR", threshold=7)
    # a.distributionPlot_Num("..\\Export\\G-generlaDistribution_number.jpg", ["US", "CN", "EU"], threshold=7)
    # a.drawGolbalBar("..\\Export\\G-bar.jpg", ["All", "Normal", "Terminal", "Trans"], "_PaR", threshold=7)
    # a.drawGlobalBoxplot2("..\\Export\\G-box.jpg", ["All", "Normal", "Terminal", "Trans"], "_PaR", threshold=7)