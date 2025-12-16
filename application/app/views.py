import time
from app import app
from flask import render_template, redirect, url_for, request, session
from .forms import TweetForm

import pickle
import pandas as pd

from app.machine_learning import functions as tc


# redirects the normal route to the homepage
@app.route('/')
@app.route('/homepage')
def index():
    return render_template("homepage.html")

# homepage route
@app.route('/homepage')
def homepage():
    return render_template("homepage.html")

# classification of a tweet
@app.route('/classify_tweet', methods=['GET','POST'])
def classify_tweet():
    form = TweetForm()
    if form.validate_on_submit():
        session['tweet'] = form.tweet.data
        return(redirect(url_for("results")))
    return render_template("classify_tweet.html", form=form)


# results of the tweet classification
@app.route('/results')
def results():
    original_tweet = session["tweet"]
    # save tweet into pandas dataframe as as string
    input_tweet = save_new_tweet_pandas(str(original_tweet))

    # clean the new_tweet and get the tweet at each stage of cleaning
    stopwords_tweet, preprocessed_tweet, lemmatised_tweet, input_tweet = clean_text(input_tweet)


    # load in the models from pickle
    taskA_model = pickle.load(open('app/machine_learning/ml_models/taskA_best.sav', 'rb'))
    taskB_model = pickle.load(open('app/machine_learning/ml_models/taskB_best.sav', 'rb'))
    taskC_model = pickle.load(open('app/machine_learning/ml_models/taskC_best.sav', 'rb'))

    # load in the vectorisers from pickle
    taskA_vectoriser = pickle.load(open('app/machine_learning/ml_models/taskA_vector.sav', 'rb'))
    taskB_vectoriser = pickle.load(open('app/machine_learning/ml_models/taskB_vector.sav', 'rb'))
    taskC_vectoriser = pickle.load(open('app/machine_learning/ml_models/taskC_vector.sav', 'rb'))



    # taskA prediction and probability of that classifier on new tweet
    # gets the prediction on the class
    taskA_pred = taskA_model.predict(taskA_vectoriser.transform(input_tweet["clean_tweet"]))[0]
    # gets the probability of the class identified by the model
        # transforms the tweet.
        # predicts the probability of all possible classes - returns an array of possibilites for each class
        # get the probability of the single tweet
        # list gets the index of the predicted class of the model in taskA_pred
    taskA_prob = taskA_model.predict_proba(taskA_vectoriser.transform(input_tweet["clean_tweet"]))[0][list(taskA_model.classes_).index(taskA_pred)]


     # taskb prediction and probability of that classifier on new tweet
    # gets the prediction on the class
    taskB_pred = taskB_model.predict(taskB_vectoriser.transform(input_tweet["clean_tweet"]))[0]
    # gets the probability of the class identified by the model
        # transforms the tweet.
        # predicts the probability of all possible classes - returns an array of possibilites for each class
        # get the probability of the single tweet
        # list gets the index of the predicted class of the model in taskB_pred
    taskB_prob = taskB_model.predict_proba(taskB_vectoriser.transform(input_tweet["clean_tweet"]))[0][list(taskB_model.classes_).index(taskB_pred)]

     # taskC prediction and probability of that classifier on new tweet
    # gets the prediction on the class
    taskC_pred = taskC_model.predict(taskC_vectoriser.transform(input_tweet["clean_tweet"]))[0]
    # gets the probability of the class identified by the model
        # transforms the tweet.
        # predicts the probability of all possible classes - returns an array of possibilites for each class
        # get the probability of the single tweet
        # list gets the index of the predicted class of the model in taskC_pred
    taskC_prob = taskC_model.predict_proba(taskC_vectoriser.transform(input_tweet["clean_tweet"]))[0][list(taskC_model.classes_).index(taskC_pred)]


    return render_template("results.html",
                                            original_tweet = original_tweet,
                                            stopwords_tweet=stopwords_tweet,
                                            preprocessed_tweet = preprocessed_tweet,
                                            lemmatised_tweet = lemmatised_tweet,

                                            taskA_model = type(taskA_model).__name__,
                                            taskA_pred = taskA_pred,
                                            taskA_prob = taskA_prob,

                                            taskB_model = type(taskB_model).__name__,
                                            taskB_pred = taskB_pred,
                                            taskB_prob = taskB_prob,

                                            taskC_model = type(taskC_model).__name__,
                                            taskC_pred = taskC_pred,
                                            taskC_prob = taskC_prob
                                        )



# explaining the project
@app.route('/project_explained')
def project_explained():
    # load the training data in and rename the columns and get EDA from function
    training_file = pd.read_csv("app/machine_learning/original_data/olid-training-v1.0.tsv", sep="\t")
    training_file = training_file.rename(columns={'subtask_a': 'taskA', 'subtask_b': 'taskB', "subtask_c" : "taskC"})
    total_entries, taskA_split, taskB_split, taskC_split, missing_values, total_words, off_dist, not_dist, tin_dist, unt_dist, ind_dist, grp_dist, oth_dist, stop_words = tc.eda(training_file)


    return render_template("project_explained.html",
                                                    total_entries = total_entries,
                                                    taskA_split = taskA_split.to_dict(), # convert to dictionary for ease of display
                                                    taskB_split = taskB_split.to_dict(), # convert to dictionary for ease of display
                                                    taskC_split = taskC_split.to_dict(), # convert to dictionary for ease of display
                                                    missing_values = missing_values.to_dict(), # convert to dictionary for ease of display
                                                    total_words = total_words,
                                                    off_dist = off_dist,
                                                    not_dist = not_dist,
                                                    tin_dist =  tin_dist,
                                                    unt_dist = unt_dist,
                                                    ind_dist = ind_dist,
                                                    grp_dist = grp_dist,
                                                    oth_dist = oth_dist,
                                                    stop_words = stop_words
                                                    )





# FUNCTION to save the tweet into a dataframe with the corresponding columns
def save_new_tweet_pandas(tweet):
    data = [[str(tweet), "", "", ""]]
    new_tweet = pd.DataFrame(data, columns=["tweet", "taskA", "taskB", "taskC"])
    return new_tweet


# FUNCTION to clean the tweet in the dataframe and return each stage of the cleaning process (uses the functions from functions.py)
def clean_text(new_tweet):
   # remove stop words
    new_tweet["clean_tweet"] = new_tweet["tweet"].apply(lambda x: tc.remove_stopwords(x))
    stopwords_tweet = new_tweet["clean_tweet"].item()

    # preprocess
    new_tweet["clean_tweet"] = new_tweet["clean_tweet"].apply(lambda x: tc.preprocess(x))
    preprocessed_tweet = new_tweet["clean_tweet"].item()

    # lemmatise
    new_tweet["clean_tweet"] = new_tweet["clean_tweet"].apply(lambda x: tc.lemmatiser(x))
    lemmatised_tweet = new_tweet["clean_tweet"].item()

    return stopwords_tweet, preprocessed_tweet, lemmatised_tweet, new_tweet


