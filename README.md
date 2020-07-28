# Bachelorthesis
GIT repository for my Bachelorthesis about "Deep Learning-Based Price Prediction for Online Game Trade Markets".

## Structure

The project contains two different python projects: <br />
* A selenium based website-scraper to extract the player base data of the FIFA20 players in the folder *Selenium_Script*. <br />
* The regression implementation in the folder *Regression*, which contains the four models: <br />
*Linear Regression, Support Vector Regression, Random Forest* and *Deep Neural Network* 

### Selenium Script


To setup Selenium for Python using Chrome follow the instructions given at the following link: <br />
https://selenium-python.readthedocs.io/installation.html 
<br />

Afterwards you just have to set two variables:<br />
* The variable *path_to_chromedriver* is assigned with the total path to the chromedriver on your computer 
* The variable *export_filepath* is assigned with the total export-path, where the extracted data is saved in a .csv-file


### Regression

The regression project uses different libraries. The main libraries are: <br />
*scikit-learn, keras, pandas*
<br />

The code is split up into four parts:
* dicts.py: contains all dictionaries that are used
* preprocessing.py: contains all code fragments that import and preprocess the data for the models and tests
* model.py: contains all model definitions and functions
* main.py: contains the main method which is called when running<br />

Again you need to assign you personal paths to the variables in the first region *predefining* in the models.py and main.py
