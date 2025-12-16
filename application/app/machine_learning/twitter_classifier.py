import time
import functions as tc



# load the datasets in and clear up headings and match the labels
training_file, testA_file, testB_file, testC_file = tc.load_dataset()

# do some eda on the training file
total_entries, taskA_split, taskB_split, taskC_split, missing_values, total_words, off_dist, not_dist, tin_dist, unt_dist, ind_dist, grp_dist, oth_dist, stop_words = tc.eda(training_file)
print("\n\nTotal number of entries: ", total_entries)
print("\nTaskA split: \n", taskA_split)
print("\nTaskB split: \n", taskB_split)
print("\nTaskC split: \n",taskC_split)
print("\nMissing Values:\n",missing_values)
print("\nTotal number of words used:", total_words)
print("\nWord distribution for an OFFENSIVE tweet:", off_dist)
print("Word distribution for a NOT OFFENSIVE tweet:", not_dist)
print("\nWord distribution for an TARGETED INSULT tweet:", tin_dist)
print("Word distribution an UNTARGETED INSULT tweet:", unt_dist)
print("\nWord distribution for an INDIVIDUAL INSULT tweet:", unt_dist)
print("Word distribution for an GROUP INSULT tweet:", grp_dist)
print("Word distribution for an OTHER INSULT tweet:", oth_dist)
print("\nStopwords: \n",stop_words)


# clean the training files and the test files
training_file, testA_file, testB_file, testC_file, training_time, testA_time, testB_time, testC_time = tc.clean_tweets(training_file, testA_file, testB_file, testC_file)
print("\n CPU Time Taken for cleaning training file (s):", training_time)
print("\n CPU Time Taken for cleaning test A file (s):", testA_time)
print("\n CPU Time Taken for cleaning test B file (s):", testB_time)
print("\n CPU Time Taken for cleaning test C file (s):", testC_time)



# Run the algorithms on the following tasks
print("\n\t'''TASK A'''")
tc.taskA_dummy(training_file, testA_file)
tc.taskA_naive_bayes(training_file, testA_file)
tc.taskA_SVC(training_file, testA_file)
tc.taskA_LR(training_file, testA_file)
tc.taskA_random_forest(training_file, testA_file)
tc.taskA_DT(training_file, testA_file)
tc.taskA_gradientboosting(training_file, testA_file)
tc.taskA_VP(training_file, testA_file)
tc.taskA_cnn(training_file, testA_file)


print("\t'''TASK B'''")
tc.taskB_dummy(training_file, testB_file)
tc.taskB_naive_bayes(training_file, testB_file)
tc.taskB_SVC(training_file, testB_file)
tc.taskB_LR(training_file, testB_file)
tc.taskB_random_forest(training_file, testB_file)
tc.taskB_DT(training_file, testB_file)
tc.taskB_gradientboosting(training_file, testB_file)
tc.taskB_VP(training_file, testB_file)
tc.taskB_cnn(training_file, testB_file)



print("\t'''TASK C'''")
tc.taskC_dummy(training_file, testC_file)
tc.taskC_naive_bayes(training_file, testC_file)
tc.taskC_SVC(training_file, testC_file)
tc.taskC_LR(training_file, testC_file)
tc.taskC_random_forest(training_file, testC_file)
tc.taskC_DT(training_file, testC_file)
tc.taskC_gradientboosting(training_file, testC_file)
tc.taskC_VP(training_file, testC_file)
tc.taskC_cnn(training_file, testC_file)

