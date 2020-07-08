# Galaxy Resource Predictions


## Overview

This tool is able to predict

- Memory Usage
- Job Runtime

of galaxy tools.

The tool basically works using a 3 step approach.
The first step is to scan the data. If multiple versions of the same tool are present,
the tool will detect them and automatically put them together in one folder.

The second step combines training the models and evaluation the results.

The third and last step contains generating reports and plots. Reports and plots will be 
placed inside the tool and version folder they are connected to. 
This will result in a clean and easy to understand folder/file structure.

## Installation

A full list of a requirements is available in the requirements.txt

### Mac:

Use the start.sh to run the application. It will automatically create a virtual env,
activates it, install the requirements, check your python version and afterwards starts the tool.


### Windows:
If you are using windows create a virtual environment, install all requirements, 
create the config by copying the config.ini.dist and rename the copy to config.ini
then start the applications main file ResourcePredictor.py.



## Command Line Arguments

CLI arguments are provided to override the config settings. 
This is due to the fact that one would like to run an evaluation in a modified state
without changing the whole config all the time.
The following table lists all available arguments:

|   Argument	|  Short 	|  Description 	|
|---	|---	|---	|
|  --remove 	|   -r	|   Activates the percentage removal of data from the data set. For each iteration 10% of the data will be removed randomly until 0 rows are left.	|
|  --verbose  	|   -v	|   Activates the verbose mode.	|
|  --merge 	|   -mg	|   Enables the merging of all files of a tool. The merged file will then be treated as normal file and evaluated accordingly.	|
|  --memory 	|   -m	|   Activates the memory saving mode.	|
|  --debug 	|   -d	|   Activates the debug mode.	|


## Sample Data

For sample data please have a look at the ExampleData folder.

