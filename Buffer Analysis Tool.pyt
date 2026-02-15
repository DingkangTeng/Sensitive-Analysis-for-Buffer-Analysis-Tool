# -*- coding: utf-8 -*-

import arcpy

# # # Test parts
# import os, random
# import pandas as pd

# Inport analysis method
from tools.DistrictDivid import DistrictDivid as DD
from tools.BufferAnalysis import BufferAnalysis as BA
from tools.BufferAnalysisArea import BufferAnalysisArea as BAA

# Set some environment settings
arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = True  

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Parking and Charging Analysis Toolbox"
        self.alias = "ParkingAndCharging"

        # List of tool classes associated with this toolbox
        self.tools = [DD, BA, BAA]