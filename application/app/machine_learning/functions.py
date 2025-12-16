'''
    Twitter Classifier App:
    1. Load the dataset in and organize columns
        - get rid of redundant ones
        - combine the labels with the tweet in corresponding file
        - rename the column names

    2. Exploratory Data Analysis (EDA) on training dataset: 
        - check all entries
        - check the split of the classifiers
        - check missing data
        - check the number of words
        - check the distribution words per class 

    3. Clean the data up by pre-processing 
        - get rid of emojis
        - get rid of punctuation
        - lower case all text
        - remove stop words

    4. Use Lemmatizer on Tweets 

    5. Split the training and testing data accordingly 
    

    6. For each task vecorise the Tweets 
        - Remove "Nan" values
        - Build models
        - Run and Evaluate the models
    
    7. Save the training data and the most accurate models to file

    8. Integrate ML model into the Flask website
'''


''' PRE-STEP: IMPORT LIBRARIES REQUIRED '''
# import necessary libraries
import pickle
import numpy as np
import pandas as pd
import string
import re
import time


# import nltk libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')


# import sklearn libraries
from sklearn.dummy import DummyClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


''' STEP 1: LOAD THE DATASET IN AND ORGANIZE COLUMNS '''
# FUNCTION to load the datasets in and organise them
def load_dataset():
    # load the training dataset in
    training_file = pd.read_csv("original_data/olid-training-v1.0.tsv", sep="\t")
    # load in the test dataset in
    testA_file = pd.read_csv("original_data/testset-levela.tsv", sep="\t")
    testB_file = pd.read_csv("original_data/testset-levelb.tsv", sep="\t")
    testC_file = pd.read_csv("original_data/testset-levelc.tsv", sep="\t")
    # load in the test labels and assign column names to them
    testA_labels = pd.read_csv("original_data/labels-levela.csv", names= ["id", "class"])
    testB_labels = pd.read_csv("original_data/labels-levelb.csv", names= ["id", "class"])
    testC_labels = pd.read_csv("original_data/labels-levelc.csv", names= ["id", "class"])

    # get rid of redundant data (ID column) in the test and training datasets
    del training_file["id"]
    del testA_file["id"]
    del testB_file["id"]
    del testC_file["id"]

    # combine the labels with the test data
    testA_file["taskA"] = testA_labels["class"]
    testB_file["taskB"] = testB_labels["class"]
    testC_file["taskC"] = testC_labels["class"]

    # rename the columns
    training_file = training_file.rename(columns={'subtask_a': 'taskA', 'subtask_b': 'taskB', "subtask_c" : "taskC"})
    return training_file, testA_file, testB_file, testC_file



''' STEP 2 EXPLORATORY DATA ANALYSIS '''
# FUNCTION to print out EDA on training data
def eda(training_file):
    # total number of rows in training data
    total_entries = len(training_file)

    # split of taskA
    taskA_split = training_file['taskA'].value_counts()

    # split of taskB
    taskB_split = training_file['taskB'].value_counts()

    # split of taskC
    taskC_split = training_file['taskC'].value_counts()

    # number of missing values for each column
    missing_values = training_file.isna().sum()

    # make a new column for word distribution
    training_file['word_count'] = training_file['tweet'].apply(lambda x: len(str(x).split()))
    # total number of words in the tweets
    total_words = training_file['word_count'].sum()

    # check the distribution of words across each classifier
    # check for taskA's word distribution
    off_dist = training_file[training_file['taskA']=="OFF"]['word_count'].mean()
    not_dist = training_file[training_file['taskA']=="NOT"]['word_count'].mean()

    # check for taskB's word distribution
    tin_dist = training_file[training_file['taskB']=="TIN"]['word_count'].mean()
    unt_dist = training_file[training_file['taskB']=="UNT"]['word_count'].mean()

    # check for taskC's word distribution
    ind_dist = training_file[training_file['taskC']=="IND"]['word_count'].mean()
    grp_dist = training_file[training_file['taskC']=="GRP"]['word_count'].mean()
    oth_dist = training_file[training_file['taskC']=="OTH"]['word_count'].mean()

    # print out the stop words we will be removing
    stop_words = stopwords.words("english")
    # get rid of the column we dont need after we finished with it
    del training_file["word_count"]
    return total_entries, taskA_split, taskB_split, taskC_split, missing_values, total_words, off_dist, not_dist, tin_dist, unt_dist, ind_dist, grp_dist, oth_dist, stop_words




''' STEP 3 & 4: CLEAN UP THE DATA AND LEMMITISATION '''
# FUNCTION to clean the data
def preprocess(text):
    # # dont need to remove URLS as they are subbed in the data with "URL"
    # text = re.sub("http[s]?\://\S+","",text)
    # removes the "@user" from text
    text = re.sub(r"@\S+", "",text)
    # removes the "URL" word from the text
    text = text.replace("URL", "")
    # removes numbers from text
    text = re.sub(r"[0-9]", "",text)
    # replace the punctuation with a space
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    # get rid of emojis
    text = text.encode('ascii', 'ignore').decode('ascii')
    # remove any tabs or new lines
    text = re.sub('\s+', ' ', text)
    # make text lowercase and strip it of any whitespace
    text = text.lower()
    text=text.strip()
    return text

# FUNCTION to remove all stopwords in the line
def remove_stopwords(line):
    # check to see if the word in the split line is in the stop words, if it isnt, keep it
    word = [x for x in line.split() if x not in stopwords.words('english')]
    return ' '.join(word)

# FUNCTION to add the POS on the word
def get_pos(word_tag):
    if word_tag.startswith('J'):
        return wordnet.ADJ
    elif word_tag.startswith('V'):
        return wordnet.VERB
    elif word_tag.startswith('N'):
        return wordnet.NOUN
    elif word_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# FUNCTION to lemmatize the line
def lemmatiser(line):
    # Will be lemmitizing rather than stemming due to stemming being harsher and lemmitization will reduce the word down to its core element
    wl = WordNetLemmatizer()
    # get the position tag
    word_pos_tags = nltk.pos_tag(word_tokenize(line))
    # map the position tag and lemmatize the word
    x = [wl.lemmatize(tag[0], get_pos(tag[1])) for i, tag in enumerate(word_pos_tags)]
    return " ".join(x)

# FUNCTION to preprocess the line
def finalpreprocess(line):
    return lemmatiser(remove_stopwords(preprocess(line)))


# FUNCTION that makes preprocessing simpler on the tweets
def clean_tweets(training_file, testA_file, testB_file, testC_file):
    # clean, remove stopwords and lemmatise the tweets in each of the files training and test
    training_start = time.process_time()
    training_file['clean_tweet'] = training_file['tweet'].apply(lambda x: finalpreprocess(x))
    training_end = time.process_time()

    testA_start = time.process_time()
    testA_file['clean_tweet'] = testA_file['tweet'].apply(lambda x: finalpreprocess(x))
    testA_end = time.process_time()

    testB_start = time.process_time()
    testB_file['clean_tweet'] = testB_file['tweet'].apply(lambda x: finalpreprocess(x))
    testB_end = time.process_time()

    testC_start = time.process_time()
    testC_file['clean_tweet'] = testC_file['tweet'].apply(lambda x: finalpreprocess(x))
    testC_end = time.process_time()
    
    clean_training_time = round(training_end - training_start, 4)
    clean_taskA_time = round(testA_end - testA_start, 4)
    clean_taskB_time = round(testB_end - testB_start, 4)
    clean_taskC_time = round(testC_end - testC_start, 4)

    return training_file, testA_file, testB_file, testC_file, clean_training_time, clean_taskA_time, clean_taskB_time , clean_taskC_time






''' STEP 5 & 6 & 7 : FOR EACH TASK VECTORISE THE TWEETS, BUILD THE MODELS, REMOVE "Nan" VALUES, BUILD & RUN AND EVALUATE AND SAVE THE MODELS'''

''' 
NOTE:
    For all tasks we are using TFIDF vectoriser.

    TASKA:
        - Dont need to remove "nan" values
    TASKB:
        - Need to remove the "nan" values
    TASKC:
        - Need to remove the "nan" values

'''


''' TASKA '''
def taskA_dummy(training_file, testA_file):
    # Using TFIDF, TFIDF runs on non-tokenized sentences unlike word2vec
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskA"]
    # testing data
    x_test = testA_file["clean_tweet"]
    y_test = testA_file["taskA"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # Naives Bayes Multinominal
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    y_predict = dummy.predict(x_test)

    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKA Dummy Classifier:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))




def taskA_naive_bayes(training_file, testA_file):
    # Using TFIDF, TFIDF runs on non-tokenized sentences unlike word2vec
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskA"]
    # testing data
    x_test = testA_file["clean_tweet"]
    y_test = testA_file["taskA"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # Naives Bayes Multinominal
    naive = MultinomialNB()
    naive.fit(x_train, y_train)
    y_predict = naive.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKA naive bayes:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskA_SVC(training_file, testA_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskA"]
    # testing data
    x_test = testA_file["clean_tweet"]
    y_test = testA_file["taskA"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    clf = svm.SVC(kernel='linear')

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"kernel": ["linear", "poly", "rbf", "sigmoid"],
            #   "gamma": ["scale", "auto"],
            #   "probability": [True, False]}]
    # model = GridSearchCV(estimator=clf, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKA SVC:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskA_LR(training_file, testA_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskA"]
    # testing data
    x_test = testA_file["clean_tweet"]
    y_test = testA_file["taskA"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # Logistic Regression
    lr = LogisticRegression()

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
    #           "penalty": ["l1", "l2", "elasticnet"]}]
    # model = GridSearchCV(estimator=lr, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKA logistic regression:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskA_random_forest(training_file, testA_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskA"]
    # testing data
    x_test = testA_file["clean_tweet"]
    y_test = testA_file["taskA"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # Random Forest Classifier
    rfc=RandomForestClassifier(n_estimators=100)

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"criterion": ["gini", "entropy", "log_loss"],
    #           "n_estimators": [10, 100, 500]}]
    # model = GridSearchCV(estimator=rfc, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    rfc.fit(x_train,y_train)
    y_predict=rfc.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKA Random Forest:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))

    # save this algorithm as it is the best performing
    pickle.dump(rfc, open('ml_models/taskA_best.sav', 'wb'))
    pickle.dump(tfidf_vectorizer, open('ml_models/taskA_vector.sav', 'wb'))





def taskA_DT(training_file, testA_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskA"]
    # testing data
    x_test = testA_file["clean_tweet"]
    y_test = testA_file["taskA"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # build the model
    tree = DecisionTreeClassifier()

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"max_features": ["auto", "sqrt", "log2", "None"],
    #           "criterion": ["gini", "entropy", "log_loss"]}]
    # model = GridSearchCV(estimator=tree, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    tree.fit(x_train, y_train)
    y_predict = tree.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKA Decision Tree:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskA_gradientboosting(training_file, testA_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)    
    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskA"]
    # testing data
    x_test = testA_file["clean_tweet"]
    y_test = testA_file["taskA"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # build the model
    gradient = GradientBoostingClassifier()

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"loss": ["log_loss", "exponential"],
    #           "criterion": ["friedman_mse", "squared_error"],
    #           "n_estimators": [10, 100, 500]}]
    # model = GridSearchCV(estimator=gradient, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    gradient.fit(x_train, y_train)
    y_predict = gradient.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKA Gradient Boosting Algorithm:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskA_VP(training_file, testA_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)   

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskA"]
    # testing data
    x_test = testA_file["clean_tweet"]
    y_test = testA_file["taskA"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # build the model
    vp = Perceptron(random_state=42, max_iter=100, tol=0.001)

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"alpha": [0.1, 0.5, 1.0],
    #           "penalty": ["l1", "l2", "elasticnet"],
    #           "max_iter" : [100, 500, 1000]}]
    # model = GridSearchCV(estimator=vp, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    vp.fit(x_train, y_train)
    y_predict = vp.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKA Voted Perceptron:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskA_cnn(training_file, testA_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)   

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskA"]
    # testing data
    x_test = testA_file["clean_tweet"]
    y_test = testA_file["taskA"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # build the model
    cnn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    cnn.fit(x_train, y_train)
    y_predict = cnn.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKA CNN:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))












''' TASKB '''
def taskB_dummy(training_file, testB_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # drop the rows with "nan" values in them
    training_file = training_file.dropna(subset = ["taskB"])
    
    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskB"]
    # testing data
    x_test = testB_file["clean_tweet"]
    y_test = testB_file["taskB"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # Dummy Classifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    y_predict = dummy.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKB Dummy Classifier:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))






def taskB_naive_bayes(training_file, testB_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # drop the rows with "nan" values in them
    training_file = training_file.dropna(subset = ["taskB"])
    
    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskB"]
    # testing data
    x_test = testB_file["clean_tweet"]
    y_test = testB_file["taskB"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # Naives Bayes Multinominal
    naive = MultinomialNB()
    naive.fit(x_train, y_train)
    y_predict = naive.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKB naive bayes:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskB_SVC(training_file, testB_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # drop the rows with "nan" values in them
    training_file = training_file.dropna(subset = ["taskB"])

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskB"]
    # testing data
    x_test = testB_file["clean_tweet"]
    y_test = testB_file["taskB"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # SVC model
    clf = svm.SVC(kernel="linear")

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"kernel": ["linear", "poly", "rbf", "sigmoid"],
    #           "gamma": ["scale", "auto"],
    #           "probability": [True, False]}]
    # model = GridSearchCV(estimator=clf, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKB SVC:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskB_LR(training_file, testB_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # drop the rows with "nan" values in them
    training_file = training_file.dropna(subset = ["taskB"])
    
    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskB"]
    
    # testing data
    x_test = testB_file["clean_tweet"]
    y_test = testB_file["taskB"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # Logistic Regression
    lr = LogisticRegression()

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
    #           "penalty": ["l1", "l2", "elasticnet"]}]
    # model = GridSearchCV(estimator=lr, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)


    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKB logistic regression:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))




def taskB_random_forest(training_file, testB_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # drop the rows with "nan" values in them
    training_file = training_file.dropna(subset = ["taskB"])
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskB"]
    # testing data
    x_test = testB_file["clean_tweet"]
    y_test = testB_file["taskB"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # Random Classifier
    rfc=RandomForestClassifier(n_estimators=100)

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"criterion": ["gini", "entropy", "log_loss"],
    #           "n_estimators": [10, 100, 500]}]
    # model = GridSearchCV(estimator=rfc, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    rfc.fit(x_train,y_train)
    y_predict=rfc.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKB Random Forest:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))

    # save this algorithm as it is the best performing
    pickle.dump(rfc, open('ml_models/taskB_best.sav', 'wb'))
    pickle.dump(tfidf_vectorizer, open('ml_models/taskB_vector.sav', 'wb'))





def taskB_DT(training_file, testB_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)   

    training_file = training_file.dropna(subset = ["taskB"])
 
    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskB"]
    # testing data
    x_test = testB_file["clean_tweet"]
    y_test = testB_file["taskB"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # build the model
    tree = DecisionTreeClassifier(criterion="entropy", max_features="sqrt")

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"max_features": ["auto", "sqrt", "log2", "None"],
    #           "criterion": ["gini", "entropy", "log_loss"]}]
    # model = GridSearchCV(estimator=tree, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    tree.fit(x_train, y_train)
    y_predict = tree.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKB Decision Tree:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskB_gradientboosting(training_file, testB_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)   

    training_file = training_file.dropna(subset = ["taskB"])
 
    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskB"]
    # testing data
    x_test = testB_file["clean_tweet"]
    y_test = testB_file["taskB"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # build the model
    gradient = GradientBoostingClassifier()

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"loss": ["log_loss", "exponential"],
    #           "criterion": ["friedman_mse", "squared_error"],
    #           "n_estimators": [10, 100, 500]}]
    # model = GridSearchCV(estimator=gradient, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    gradient.fit(x_train, y_train)
    y_predict = gradient.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKB Gradient Boosting:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskB_VP(training_file, testB_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)   

    training_file = training_file.dropna(subset = ["taskB"])

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskB"]
    # testing data
    x_test = testB_file["clean_tweet"]
    y_test = testB_file["taskB"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # build the model
    vp = Perceptron(random_state=42, max_iter=100, tol=0.001)

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"alpha": [0.1, 0.5, 1.0],
    #           "penalty": ["l1", "l2", "elasticnet"],
    #           "max_iter" : [100, 500, 1000]}]
    # model = GridSearchCV(estimator=vp, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    vp.fit(x_train, y_train)
    y_predict = vp.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKB Voted Perceptron:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskB_cnn(training_file, testB_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)   

    training_file = training_file.dropna(subset = ["taskB"])

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskB"]
    # testing data
    x_test = testB_file["clean_tweet"]
    y_test = testB_file["taskB"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # build the model
    cnn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    cnn.fit(x_train, y_train)
    y_predict = cnn.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKB CNN:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))












''' TASKC '''
def taskC_dummy(training_file, testC_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # drop the rows with "nan" values in them
    training_file = training_file.dropna(subset = ["taskC"])
    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskC"]
    # testing data
    x_test = testC_file["clean_tweet"]
    y_test = testC_file["taskC"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # Dummy Classifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    y_predict = dummy.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKC Dummy Classifier:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskC_naive_bayes(training_file, testC_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # drop the rows with "nan" values in them
    training_file = training_file.dropna(subset = ["taskC"])
    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskC"]
    # testing data
    x_test = testC_file["clean_tweet"]
    y_test = testC_file["taskC"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # Naives Bayes Multinominal
    naive = MultinomialNB()
    naive.fit(x_train, y_train)
    y_predict = naive.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKC naive bayes:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskC_SVC(training_file, testC_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # drop the rows with "nan" values in them
    training_file = training_file.dropna(subset = ["taskC"])

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskC"]
    # testing data
    x_test = testC_file["clean_tweet"]
    y_test = testC_file["taskC"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # SVC model
    clf = svm.SVC(kernel="linear")

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"kernel": ["linear", "poly", "rbf", "sigmoid"],
    #           "gamma": ["scale", "auto"],
    #           "probability": [True, False]}]
    # model = GridSearchCV(estimator=clf, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKC SVC:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskC_LR(training_file, testC_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # drop the rows with "nan" values in them
    training_file = training_file.dropna(subset = ["taskC"])

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskC"]
    # testing data
    x_test = testC_file["clean_tweet"]
    y_test = testC_file["taskC"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # Logistic Regression
    lr = LogisticRegression()

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
    #           "penalty": ["l1", "l2", "elasticnet"]}]
    # model = GridSearchCV(estimator=lr, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKC logistic regression:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskC_random_forest(training_file, testC_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # drop the rows with "nan" values in them
    training_file = training_file.dropna(subset = ["taskC"])

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskC"]
    # testing data
    x_test = testC_file["clean_tweet"]
    y_test = testC_file["taskC"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # Random Forest Classifier
    rfc=RandomForestClassifier(n_estimators=50)

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"criterion": ["gini", "entropy", "log_loss"],
    #           "n_estimators": [10, 50, 100, 500]}]
    # model = GridSearchCV(estimator=rfc, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    rfc.fit(x_train,y_train)
    y_predict=rfc.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKC Random Forest:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))

    # save this algorithm as it is the best performing
    pickle.dump(rfc, open('ml_models/taskC_best.sav', 'wb'))
    pickle.dump(tfidf_vectorizer, open('ml_models/taskC_vector.sav', 'wb'))




def taskC_DT(training_file, testC_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)    
    training_file = training_file.dropna(subset = ["taskC"])

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskC"]
    # testing data
    x_test = testC_file["clean_tweet"]
    y_test = testC_file["taskC"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # build the model
    tree = DecisionTreeClassifier()

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"max_features": ["auto", "sqrt", "log2", "None"],
    #           "criterion": ["gini", "entropy", "log_loss"]}]
    # model = GridSearchCV(estimator=tree, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    tree.fit(x_train, y_train)
    y_predict = tree.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKC Decision Tree:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskC_gradientboosting(training_file, testC_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)   

    training_file = training_file.dropna(subset = ["taskC"])
 
    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskC"]
    # testing data
    x_test = testC_file["clean_tweet"]
    y_test = testC_file["taskC"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # build the model
    gradient = GradientBoostingClassifier()

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"loss": ["log_loss", "exponential"],
    #           "criterion": ["friedman_mse", "squared_error"],
    #           "n_estimators": [10, 100, 500]}]
    # model = GridSearchCV(estimator=gradient, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    gradient.fit(x_train, y_train)
    y_predict = gradient.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKC Gradient Bosting:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskC_VP(training_file, testC_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)   

    training_file = training_file.dropna(subset = ["taskC"])

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskC"]
    # testing data
    x_test = testC_file["clean_tweet"]
    y_test = testC_file["taskC"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # build the model
    vp = Perceptron(random_state=42, max_iter=100, tol=0.001)

    # # GRIDSEARCH FOR BEST PARAMETERS
    # param = [{"alpha": [0.1, 0.5, 1.0],
    #           "penalty": ["l1", "l2", "elasticnet"],
    #           "max_iter" : [100, 500, 1000]}]
    # model = GridSearchCV(estimator=vp, param_grid=param, scoring="accuracy", cv=2, verbose=8)
    # model.fit(x_train, y_train)
    # print(model.best_params_, model.best_score_)

    vp.fit(x_train, y_train)
    y_predict = vp.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKC Voted Perceptron:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))





def taskC_cnn(training_file, testC_file):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)   

    training_file = training_file.dropna(subset = ["taskC"])

    # training data
    x_train = training_file["clean_tweet"]
    y_train = training_file["taskC"]
    # testing data
    x_test = testC_file["clean_tweet"]
    y_test = testC_file["taskC"]

    start = time.process_time()
    # Don't fit TFIDF on test data or it will change the weights to fit the test data. Fit on training data and use same model on test data
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)

    # build the model
    cnn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    cnn.fit(x_train, y_train)
    y_predict = cnn.predict(x_test)
    end = time.process_time()

    # print out the evaluation of the model
    print("\n\tTASKC CNN:")
    print("Classification Report:\n", classification_report(y_test, y_predict))
    labels = np.unique([y_test, y_predict])
    cm = pd.DataFrame(confusion_matrix(y_test, y_predict, labels=labels), index = ['true:{:}'.format(x) for x in labels], columns=['pred:{:}'.format(x) for x in labels])
    print(cm)
    print("Accuracy score:", accuracy_score(y_test, y_predict))
    print("CPU Time Taken (s): ", round(end - start, 4))
