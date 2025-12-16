''''
The data is currently in various forms such as TSV and split over multiple files and we will be doing certain steps to ensure that it will be accepted by WEKA. It will be converted into the format suitable (.arff)

Clean up the data for the files ["olid-training-v1.0.tsv", "testset-levela.tsv", "testset-levelb.tsv", "testset-levelc.tsv"]:
    1. Convert all ["] to ['] 
    2. Remove all emojis
    3. Add ["] before and after each string
    4. Remove ID's

    5. For each corresponding testset append the label at the end of the file.
    6. Save them all as CSV's in the "weka_data" directory
    7. Convert csv files into ARFF format by adding relevant data
    8. Change the file name extension

'''

# import Required Libraries
import pandas as pd
import os
import csv
import shutil


# function to add in the data required to convert a csv to an arff format
def add_arff_data(data, file):
    fp = open(file, 'r').readlines()
    fp[0] = data
    file = open(file, 'w')
    for line in fp:
        file.write(line)
    file.close()


# load the files in
training_file = pd.read_table("original_data/olid-training-v1.0.tsv", sep="\t")
testsetA_file = pd.read_table("original_data/testset-levela.tsv", sep="\t")
testsetB_file = pd.read_table("original_data/testset-levelb.tsv", sep="\t")
testsetC_file = pd.read_table("original_data/testset-levelc.tsv", sep="\t")

# get rid of all the double quotation marks and replace them with single quotations
training_file['tweet'] = training_file["tweet"].str.replace("\"", "'")
testsetA_file['tweet'] = testsetA_file["tweet"].str.replace("\"", "'")
testsetB_file['tweet'] = testsetB_file["tweet"].str.replace("\"", "'")
testsetC_file['tweet'] = testsetC_file["tweet"].str.replace("\"", "'")

# get rid of all emojis using a lambda
training_file = training_file.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
testsetA_file = testsetA_file.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
testsetB_file = testsetB_file.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
testsetC_file = testsetC_file.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))

# add in double quotation marks at the start and end of the tweets
training_file["tweet"] = '"' + training_file["tweet"].astype(str) + '"'
testsetA_file["tweet"] = '"' + testsetA_file["tweet"].astype(str) + '"'
testsetB_file["tweet"] = '"' + testsetB_file["tweet"].astype(str) + '"'
testsetC_file["tweet"] = '"' + testsetC_file["tweet"].astype(str) + '"'

# delete the ID column from the test and training data as this is not required for the classification
del training_file["id"]
del testsetA_file["id"]
del testsetB_file["id"]
del testsetC_file["id"]

# for each test set we will open the label and append them to the end of the file
testsetA_label = pd.read_csv("original_data/labels-levela.csv", names= ["id", "class"])
testsetB_label = pd.read_csv("original_data/labels-levelb.csv", names= ["id", "class"])
testsetC_label = pd.read_csv("original_data/labels-levelc.csv", names= ["id", "class"])

# take the column that we require (class) and append it to the test data named "class"
testsetA_file["class"] = testsetA_label["class"]
testsetB_file["class"] = testsetB_label["class"]
testsetC_file["class"] = testsetC_label["class"]

# save the data as csv format
training_file.to_csv('weka_data/csv_format/training.csv', index=False, quoting=csv.QUOTE_NONE, quotechar=None, escapechar="\\")
testsetA_file.to_csv('weka_data/csv_format/testA.csv', index=False, quoting=csv.QUOTE_NONE, quotechar=None, escapechar="\\")
testsetB_file.to_csv('weka_data/csv_format/testB.csv', index=False, quoting=csv.QUOTE_NONE, quotechar=None, escapechar="\\")
testsetC_file.to_csv('weka_data/csv_format/testC.csv', index=False, quoting=csv.QUOTE_NONE, quotechar=None, escapechar="\\")


# copy over csv files to arff folder
source_dir = "weka_data/csv_format/"
target_dir = "weka_data/arff_format/"
files = os.listdir(source_dir)

for file in files:
    shutil.copy(source_dir + file, target_dir + file)

# convert the CSV files into ARFF format so it can be used in WEKA
training_line = """
@relation training-data
@attribute tweet string
@attribute taskA {OFF, NOT}
@attribute taskB {TIN,UNT,nan}
@attribute taskC {IND,GRP,OTH,nan}

@data
"""

testA_line = """
@relation training-data
@attribute tweet string
@attribute taskA {OFF, NOT}

@data
"""

testB_line = """
@relation taskB-data
@attribute tweet string
@attribute taskB {TIN,UNT}

@data
"""

testC_line = """
@relation taskC-data
@attribute tweet string
@attribute taskC {IND,GRP,OTH}

@data
"""

# delete the first line of each file and append the corresponding ones above
# function for this is shown at the end of the code
add_arff_data(training_line, "weka_data/arff_format/training.csv")
add_arff_data(testA_line, "weka_data/arff_format/testA.csv")
add_arff_data(testB_line, "weka_data/arff_format/testB.csv")
add_arff_data(testC_line, "weka_data/arff_format/testC.csv")

# convert the file extension from .csv to .arff
os.rename("weka_data/arff_format/training.csv", "weka_data/arff_format/training.arff")
os.rename("weka_data/arff_format/testA.csv", "weka_data/arff_format/testA.arff")
os.rename("weka_data/arff_format/testB.csv", "weka_data/arff_format/testB.arff")
os.rename("weka_data/arff_format/testC.csv", "weka_data/arff_format/testC.arff")
