# Overview
This repository contains all codes and (sample) dataset of the paper - 
***A global insight into integration of metro and electric vehicle charging stations***. 

Note that the **full dataset** can be requested through our [Global EV Data Initiative](https://globalevdata.github.io/data.html).

# Requirements and Installation
## Sensitive-Analysis-for-Buffer-Analysis-Tool 
ParkingAndCharging.pyt is an ArcGIS Tool Box for multi-ring buffer analysis, which can be used to creat multi-ring buffer and calculate how many features are counted in different buffer rings. The tool only tested in ArcGIS Pro 3.3.1

## Analysis code
The whole analysis-related codes is in `rowDataProcess` and `analysis` folders and should run with a **Python** environment with a version higher than **3.9**. 
We successfully execute all the codes in Windows (Win11) machines.

More detailed info is as below:

## Prerequisites 
It is highly recommended to install and use the following versions of python/packages to run the codes:
- ``matplotlib`` == 3.10.7
- ``numpy`` == 2.3.5
- ``pandas`` == 2.3.3
- ``pypinyin`` == 0.55.0
- ``Requests`` == 2.32.5
- ``scipy`` == 1.16.3
- ``seaborn`` == 0.13.2
- ``arcpy``: It is highly recommended to install [ArcGIS Pro](https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview) 
 **with version 2.8** to run the **arcpy-related codes**. 
 Even though the related-functions could also be implemented by other packages (e.g., geopandas), 
 the arcpy package is more efficient and convenient for the spatial analysis and geometry operations.

## Installation
It is highly recommended to download [AnaConda](https://www.anaconda.com) to create/manage Python environments.
You can create a new Python environment and install required aforementioned packages (except for `arcpy`) via both the GUI or Command Line.
Typically, the installation should be prompt (around _10-20 min_ from a "_clean_" machine to "_ready-to-use_" machine, but highly dependent on the Internet speed)
- via **Anaconda GUI**
  1. Open the Anaconda
  2. Find and click "_Environments_" at the left sidebar
  3. Click "_Create_" to create a new Python environment
  4. Select the created Python environment in the list, and then search and install all packages one by one.


- via **Command Line** (using **_Terminal_** for macOS machine and **_Anaconda Prompt_** for Windows machine, respectively)
  1. Create your new Python environment
     ```
     conda create --name <input_your_environment_name> python=3.10.6
     ```
  2. Activate the new environment 
     ```
     conda activate <input_your_environment_name>
     ```
  3. Install all packages one by one 
     ```
     conda install <package_name>=<specific_version>
     ```

# Usage
1. Git clone/download the repository to your local disk.
2. Unzip the full datasets (which can be provided upon request, see [Overview](https://github.com/DingkangTeng/Sensitive-Analysis-for-Buffer-Analysis-Tool?tab=readme-ov-file#overview))
   > The structure of the provided full datasets should look like as below:
   > 
   > ```
   > - __SampleData
   > - analysis
   > - rowDataProcess
   > - tools
   > - ParkingAndCharging.pyt
   > ```
3. Run
   1. **preprocessing**: run each script in the dir ``./rowDataProcess``
      > - amapMetro.py: get metro data from Amap Metro Map
      > - analysis.py: compare data value from different data sources
      > - oldData.py: merge data from old data sources
   2. **buffer analysis**: run ``./ParkingAndCharging.pyt`` under ``ArcGIS`` enovironmet, python script in ``./tools`` serve for this tool.
   3. **analysis**: run each script (shown as below) in the dir ``./analysis/``
      > - sensativeAnalysis.py: creat heat map for buffer-rings
      > - globalAnalysis.py: generate analysis results
      > - function.py: basic function serve for other python scripts
5. Outputs (including text files and figures) will be stored in the dir ``./__SampleData/Export/``.

# Contact
- Leave questions in [Issues on GitHub](https://github.com/DingkangTeng/Sensitive-Analysis-for-Buffer-Analysis-Tool/issues)
- Get in touch with the Corresponding Author: [Dr. Chengxiang Zhuge](mailto:chengxiang.zhuge@polyu.edu.hk)
or visit our research group website: [The TIP](https://thetipteam.editorx.io/website) for more information

# License
This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
