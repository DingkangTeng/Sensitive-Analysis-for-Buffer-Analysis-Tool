# -*- coding: utf-8 -*-

import arcpy

class DistrictDivid(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "District Divid"
        self.description = "Add district classification result into the database."

    def getParameterInfo(self):
        """Define the tool parameters."""
        inLayer = arcpy.Parameter(
            displayName="Input Layer",
            name="inputLayer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        zoneLayer = arcpy.Parameter(
            displayName="Zone Division Layer",
            name="zoneLayer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        zoneField = arcpy.Parameter(
            displayName="Select zone name field",
            name="zoneField",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        zoneField.parameterDependencies = [zoneLayer.name]
        zoneField.filter.list = ["Text"]
        savePath = arcpy.Parameter(
            displayName="Save path",
            name="savePath",
            datatype="DEDiskConnection",
            parameterType="Required",
            direction="Output"
        )

        return [inLayer, zoneLayer, zoneField, savePath]

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
        inLayer = parameters[0].valueAsText
        zoneLayer = parameters[1].valueAsText
        zoneField = parameters[2].valueAsText
        savePath = parameters[3].valueAsText

        # Add fiedmappings for input layer
        fieldmappings = arcpy.FieldMappings()
        fieldmappings.addTable(inLayer)

        # Join specific field in zoneLayer
        zoneFieldmappings = arcpy.FieldMappings()
        zoneFieldmappings.addTable(zoneLayer)
        # OID cannot be find using index (return -1)
        zoneNameIndex = zoneFieldmappings.findFieldMapIndex(zoneField)
        zoneName = zoneFieldmappings.getFieldMap(zoneNameIndex)
        fieldmappings.addFieldMap(zoneName)

        arcpy.analysis.SpatialJoin(inLayer, zoneLayer, savePath, "JOIN_ONE_TO_ONE", "KEEP_ALL", fieldmappings, "WITHIN")

        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return