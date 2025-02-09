# -*- coding: utf-8 -*-

import arcpy, os, random
import pandas as pd

class BufferAnalysis(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Buffer Analysis"
        self.description = "Analysis the relationship between Metro station and charging station or parking lots."

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
        chargingLayer = arcpy.Parameter(
            displayName="Charging Station Layer",
            name="chargingLayer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        zoneFieldC = arcpy.Parameter(
            displayName="Select zone name field of charging station",
            name="zoneFieldC",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        zoneFieldC.parameterDependencies = [chargingLayer.name]
        zoneFieldC.filter.list = ["Text"]
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

        return [metroLayer, zoneFieldM, chargingLayer, zoneFieldC, bufferSyntax, savePath]

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
        chargingLayer = parameters[2].valueAsText # Charging Station Layer
        zoneFieldC = parameters[3].valueAsText # Charging Station Zone Field
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
            expression = arcpy.AddFieldDelimiters(chargingLayer, zoneFieldC)+"=\'" + city +'\''
            arcpy.management.SelectLayerByAttribute(chargingLayer, "NEW_SELECTION", expression)
            
            # Get the total number of charging station in one city
            totalCharing = int(arcpy.management.GetCount(chargingLayer).getOutput(0))
            # No charging station, save 0 and skip to next city
            if totalCharing == 0:
                # Append result in dataframe
                results = pd.concat([results, pd.DataFrame(result)])
                continue
            
            # Creat Buffer (Rings)
            arcpy.AddMessage("Creating buffer.")
            pathBuffer = os.path.join("memory", "Buffer" + memoryName) # Save intermediate result in memory
            arcpy.analysis.MultipleRingBuffer(metroLayer, pathBuffer, self.distance, "Meters", "distance", "ALL", "FULL", "GEODESIC")

            # Calculate the number of charger in the buffer area
            arcpy.AddMessage("Calculation station number.")
            pathWithin = os.path.join("memory", "Within" + memoryName) # Save intermediate result in memory
            ## SummarizeWithin has performance problem
            # arcpy.analysis.SummarizeWithin(pathBuffer, chargingLayer, pathWithin, "KEEP_ALL")
            ## Using SpatialJoin instead
            # Add fiedmappings for input layer
            fieldmappings = arcpy.FieldMappings()
            fieldmappings.addTable(pathBuffer)
            arcpy.analysis.SpatialJoin(pathBuffer, chargingLayer, pathWithin, "JOIN_ONE_TO_ONE", "KEEP_ALL", fieldmappings, "CONTAINS")
            arcpy.management.Delete(pathBuffer)

            # Sort using disatnce
            pathSort = os.path.join("memory", "Sort" + memoryName) # Save intermediate result in memory
            arcpy.management.Sort(pathWithin, pathSort, [["distance", "Ascending"]])
            arcpy.management.Delete(pathWithin)
            
            # Change feature into table
            arcpy.AddMessage("Converting results into table.")
            numpyArry = arcpy.da.TableToNumPyArray(pathSort, '*')
            # Useful Field name: distance in [2] & Count of Points in [3] (Summarize within)
            # Useful Field name: distance in [4] & Count of Points in [2] (Spatial Join)
            for i in range(len(numpyArry)):
                data = numpyArry[i]
                result["distance"][i] = data[4]
                # Buffer shape is disk, so the number shound be sum up
                if i == 0:
                    result["Num"][i] = data[2]
                else:
                    result["Num"][i] = result["Num"][i - 1] + data[2]
                result["totalNum"][i] = totalCharing
            arcpy.management.Delete(pathSort)

            # Append result in dataframe
            results = pd.concat([results, pd.DataFrame(result)])

        results.to_csv(savePath + ".csv", encoding="utf-8", index=False)
        
        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return

    def randomName(self, name: str = "") -> str:
        return name+"_"+"".join(random.sample('zyxwvutsrqponmlkjihgfedcba1234567890',10))
    
    def setDistance(self, distance: list) -> None:
        self.distance = distance

        return