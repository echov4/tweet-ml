# Twitter Classification Project


## Description:
This project uses trained ML model to determine the offensiveness of a tweet inputted. It determines the following criteria
1. Is it **offensive** or **not**
2. If it is offensive, is it **targeted** or **untargeted**
3. if it is **targeted**, does it target an **individual**, **group** or **other**


The project is split into two main parts:
1. The Machine Learning
   * This is where the data is trained on by the models and then saved to be used in the web application
2. The Web Application
   * This is where the web application takes the saved models and utilizes it on new tweets in the forms




## Data
The dataset used to train the ML models, was downloaded from [joeykay9's Github](https://github.com/joeykay9/offenseval?tab=readme-ov-file). This dataset is from the OLID collection that was used in the OﬀensEval 2019 challenge (task 6).

### Task Description:
* Task A:
  * Determine if a tweet is offensive (OFF) or not (NOT)
* Task B:
  * Determine if a tweet is targeted insult (TIN) or untargeted insult (UNT)
* Task C:
  * Determine if a tweet is targeted to an individual (IND), group (GRP), or other (OTH)


### Files
| File                   | Description                                                           |
| ---------------------- | --------------------------------------------------------------------- |
| `olid-training-v1.tsv` | Contains 13,240 annotated tweets                                      |
| `olid-annotation.txt`  | Contains a short summary of the annotation guidelines                 |
| `testset-levela.tsv`   | Contains the test set instances of level a.                           |
| `testset-levelb.tsv`   | Contains the test set instances of level b                            |
| `testset-levelc.tsv`   | Contains the test set instances of level c                            |
| `labels-levela.csv`    | Contains the gold labels and IDs of the instances in test set layer a |
| `labels-levelb.csv`    | Contains the gold labels and IDs of the instances in test set layer b |
| `labels-levelc.csv`    | Contains the gold labels and IDs of the instances in test set layer c |


### Task Labels:
* Task A: Offensive language identification
  * (NOT) Not Offensive - This post does not contain offence or profanity
  * (OFF) Offensive - This post contains offensive language or a targeted (veiled or direct) offence
    * In our annotation, we label a post as offensive (OFF) if it contains any form of non-acceptable language (profanity) or a targeted offence, which can be veiled or direct

* Task B: Automatic categorization of offence types
  * (TIN) Targeted Insult and Threats - A post containing an insult or threat to an individual, a group, or others (see categories in sub-task C)
  * (UNT) Untargeted - A post containing non-targeted profanity and swearing
    * Posts containing general profanity are not targeted, but they contain non-acceptable language

* Task C: Offence target identification
  * (IND) Individual - The target of the offensive post is an individual: a famous person, a named individual or an unnamed person interacting in the conversation
  * (GRP) Group - The target of the offensive post is a group of people considered as a unity due to the same ethnicity, gender or sexual orientation, political affiliation, religious belief, or something else
  * (OTH) Other – The target of the offensive post does not belong to any of the previous two categories (e.g., an organization, a situation, an event, or an issue)

## Installation
Need to have the following installed:
* Python 3
* pip
* python3-venv

To initialize the project:
1. Make a virtual environment `python3 -m venv venv`
2. Activate the virtual environment `source venv/bin/activate`
3. Download the libraries required `pip install -r requirements.txt` in the directory where "requirements.txt" is stored. Usually under `application/`

	``` bash
	python3 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt
	```



## Run:
* To run the web application:
  * In the `/application` (where the `run.py` file is located) directory run the command below. This will start the web application on localhost at port 5000 or at **127.0.0.1:5000**
  ``` bash
  flask run
  ```




## Basic Directory Structure:
### Application Structure
| Directory / File                             | Description                                               |
| -------------------------------------------- | --------------------------------------------------------- |
| `app/`                                       | Contains all relevant files and directories               |
| `app/forms.py`                               | Contains forms used in the web application                |
| `app/machine_learning/`                      | Contains files related to ML models                       |
| `app/machine_learning/functions.py`          | Contains functions for data processing and model building |
| `app/machine_learning/ml_models/`            | Contains saved models                                     |
| `app/machine_learning/original_data/`        | Contains raw data for ML models                           |
| `app/machine_learning/twitter_classifier.py` | Runs the ML models and saves results                      |
| `app/static/`                                | Contains images and CSS files                             |
| `app/templates/`                             | Contains HTML templates for the web application           |
| `app/views.py`                               | Contains main logic for the web application               |
| `config.py`                                  | Contains web application configurations                   |
| `run.py`                                     | Script to run the web application                         |


### Data Structure
| Directory / File         | Description                                                       |
| ------------------------ | ----------------------------------------------------------------- |
| `original_data/`         | Contains raw, untouched data for the ML models                    |
| `weka_data/`             | Contains output data from `weka_data.py`                          |
| `weka_data/arff_format/` | Contains cleaned-up data in ARFF format for WEKA                  |
| `weka_data/csv_format/`  | Contains cleaned-up data in CSV format for viewing                |
| `weka_data.py`           | Script to clean up and merge data, output in CSV and ARFF formats |




## Machine Learning
### `functions.py` and `twitter_classifier.py`
* `functions.py`
  * Contains all the functions required to load the data, clean the data, build the model and run it. It also saves the best model and the corresponding TF-IDF using the pickle library so that it can be used in the Web App. The ML model and the TF-IDF is stored in the `project/application/app/machine_learning/ml_models/`

* `twitter_classifier.py`
  * Uses the functions defined in the `functions.py` to load, clean the dataset. Train and save the models. Calculate the metrics


## Web Application
* Uses the saved models in the `app/machine_learning/ml_models/` directory. Loads the model in and depending on the data inputted in the form by the user, it will test on that data and provide metrics to view
* To use a different model in the Web App, you will need to carry out the following steps:
1. In the `functions.py` move the pickle lines from the current model being used in the Web App into the desired model you want to use, for each task. The lines that need moving are:
``` python
    pickle.dump(rfc, open('ml_models/taskA_best.sav', 'wb'))
    pickle.dump(tfidf_vectorizer, open('ml_models/taskA_vector.sav', 'wb'))
```
 * The fist line saves the model for that task
 * the second line saves the corresponding TF-IDF for that task

2. Run the `twitter_classifier.py` again, to build the model, and the TF-IDF. Then the script will save it in the `/project/application/app/machine_learning/ml_models/` directory. This will automatically be imported into the Web App





## WEKA Process
* If you wish to use the data for WEKA and play around with it. You can do so with the following steps that I took
* To get the data for WEKA in ARFF format, you will need to run the `weka_data.py` in the `data` directory. This will output the `weka_data` directory with the data in CSV format and ARFF format
1. Load the ARFF data in and remove the unwanted classifiers for task B and task C

2. Click in classify tab and select the "filtered classifier"
   * select the filter as "StringToWord" with the following settings:
     * IDF Transform = True
     	* TF Transform = True
     	* lower case tokens = True
     	* Stemmer = SnowballStemmer
     	* Stopwords handler = Multistopworda
     	* tokeniser = wordtokeniser
         	* delimeters = .,;:'"()?!@/\
     	* Words to keep = 2000
   * Select the following algorithms:
   	* ZeroR
   	* NaiveB Bayes Multinominal
   	* RandomForest

3. Run for each task
   * NOTE: For taskb and taskc remove the nan values
   * Click the task
   * Select filter in preprocess tab remove with values
   	* "RemoveWithValues"
   		* modify headers = true
   		* select attribute (last)
   		* select nominal indices (last)




## Known Issues:
* If there are issues with the Web Application not working.
  * It is possible you may need to run the `twitter_classifier.py` to save the models, so they can be used

* `InconsistentVersionWarning:` Trying to unpickle estimator "X" from version "Y" when using version "Z".
  * **Reason:**: It is due to the pickle object having a model saved that was built on the older version of the sci-kit learn library
  * **Solution**: You may need to re-run the `twitter_classifier.py` for the sci-kit learn to create a new pickle object that can be loaded in

