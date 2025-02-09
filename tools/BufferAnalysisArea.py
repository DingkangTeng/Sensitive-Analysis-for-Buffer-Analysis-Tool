# -*- coding: utf-8 -*-

import arcpy, os, random
import pandas as pd

class BufferAnalysisArea(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Buffer Analysis(Area)"
        self.description = "Analysis the the proporation of area of the buffer zone."

    def getParameterInfo(self):
        """Define the tool parameters."""
        metroLayer = arcpy.Parameter(
            displayName="Metro Layer",
            name="metroLayer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        zoneFieldM = arcpy.Parameter(
            displayName="Select zone name field of metro",
            name="zoneFieldM",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        zoneFieldM.parameterDependencies = [metroLayer.name]
        zoneFieldM.filter.list = ["Text"]
        districtLayer = arcpy.Parameter(
            displayName="Disrict Layer",
            name="districtLayer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        zoneFieldD = arcpy.Parameter(
            displayName="Select zone name field of district layer",
            name="zoneFieldD",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        zoneFieldD.parameterDependencies = [districtLayer.name]
        zoneFieldD.filter.list = ["Text"]
        bufferSyntax = arcpy.Parameter(
            displayName="Buffer (Meter) Syntax (Python)",
            name="bufferSyntax",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        bufferSyntax.value = "[10 * x for x in range(1,201)]"
        savePath = arcpy.Parameter(
            displayName="Save path",
            name="savePath",
            datatype="DEDiskConnection", #DEFolder
            parameterType="Required",
            direction="Output"
        )

        return [metroLayer, zoneFieldM, districtLayer, zoneFieldD, bufferSyntax, savePath]

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        metroLayer = parameters[0].valueAsText # Metro Station Layer
        zoneFieldM = parameters[1].valueAsText # Metro Station Zone Field
        districtLayer = parameters[2].valueAsText # Charging Station Layer
        zoneFieldD = parameters[3].valueAsText # Charging Station Zone Field
        exec("self.setDistance(" + parameters[4].valueAsText + ")") # Set buffer distance
        savePath = parameters[5].valueAsText # Saving path of calculation csv file
        parts = len(self.distance)

        # Get unique city set
        allCities = set()
        with arcpy.da.SearchCursor(metroLayer, [zoneFieldM]) as cursor:
            for row in cursor:
                allCities.add(row[0])
        totalWorking = len(allCities)
        
        # Initialize null dataframe
        results = pd.DataFrame({"city": [], "distance": [], "Num": [], "totalNum": []})

        # Calculate total Area
        totalAreaName = self.randomName("tA")
        arcpy.management.AddField(districtLayer, totalAreaName, "DOUBLE")
        arcpy.management.CalculateField(districtLayer, totalAreaName, "!shape.geodesicArea!", "PYTHON3")
        
        # Set buffer using city
        num = 1
        for city in allCities:
            # Initialize meaasge and null result
            arcpy.AddMessage("Processing city {} ({}/{})".format(city, num, totalWorking))
            num += 1
            result = {"city": [city] * parts, "distance": self.distance, "Num": [0] * parts, "totalNum": [0] * parts}
            memoryName = self.randomName()

            # Select data in one city
            expression = arcpy.AddFieldDelimiters(metroLayer, zoneFieldM)+"=\'" + city +'\''
            arcpy.management.SelectLayerByAttribute(metroLayer, "NEW_SELECTION", expression)
            expression = arcpy.AddFieldDelimiters(districtLayer, zoneFieldD)+"=\'" + city +'\''
            arcpy.management.SelectLayerByAttribute(districtLayer, "NEW_SELECTION", expression)

            # Total Area
            with arcpy.da.SearchCursor(districtLayer, [totalAreaName]) as cursor:
                for row in cursor:
                    totalArea = row[0]
            
            # Creat Buffer (Rings)
            arcpy.AddMessage("Creating buffer.")
            pathBuffer = os.path.join("memory", "Buffer" + memoryName) # Save intermediate result in memory
            arcpy.analysis.MultipleRingBuffer(metroLayer, pathBuffer, self.distance, "Meters", "distance", "ALL", "FULL", "GEODESIC")

            # Cut the outside area
            arcpy.AddMessage("Processing buffer.")
            pathIntersect = os.path.join("memory", "Intersect" + memoryName) # Save intermediate result in memory
            arcpy.analysis.PairwiseIntersect([pathBuffer, districtLayer], pathIntersect, "All")

            # Sort using disatnce
            pathSort = os.path.join("memory", "Sort" + memoryName) # Save intermediate result in memory
            arcpy.management.Sort(pathIntersect, pathSort, [["distance", "Ascending"]])
            arcpy.management.Delete(pathIntersect)
            
            # Calculate Area
            arcpy.AddMessage("Calculating area.")
            areaName = self.randomName("Area")
            arcpy.management.AddField(pathSort, areaName, "DOUBLE")
            arcpy.management.CalculateField(pathSort, areaName, "!shape.geodesicArea!", "PYTHON3")
            
            # Change feature into table
            arcpy.AddMessage("Coverting results into table.")
            with arcpy.da.SearchCursor(pathSort, ["distance", areaName]) as cursor:
            # Useful Field name: distance in [4] & Count of Points in [2] (Spatial Join)
                i = 0
                for data in cursor:
                    result["distance"][i] = data[0]
                    # Buffer shape is disk, so the number shound be sum up
                    if i == 0:
                        result["Num"][i] = data[1]
                    else:
                        result["Num"][i] = result["Num"][i - 1] + data[1]
                    result["totalNum"][i] = totalArea
                    i += 1
            arcpy.management.Delete(pathSort)

            # Append result in dataframe
            results = pd.concat([results, pd.DataFrame(result)])

        arcpy.management.DeleteField(districtLayer, totalAreaName)
        results.to_csv(savePath + ".csv", encoding="utf-8", index=False)
        
        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return

    def randomName(self, name: str = "") -> str:
        return name+"_"+"".join(random.sample('zyxwvutsrqponmlkjihgfedcba1234567890',5))
    
    def setDistance(self, distance: list) -> None:
        self.distance = distance

        return
