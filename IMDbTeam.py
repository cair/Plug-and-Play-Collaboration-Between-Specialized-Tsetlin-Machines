import numpy as np
import keras
from keras.datasets import imdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--imdb-num-words", default=5000, type=int)
parser.add_argument("--imdb-index-from", default=2, type=int)

train, test = keras.datasets.imdb.load_data(num_words=args.imdb_num_words, index_from=args.imdb_index_from)

Y_test = test_y.astype(np.uint32)

Y_test_scores_1 = np.loadtxt("class_sums/IMDBAnalyzer_1_10000_8000_2.00_1_2_32_1_word_0.75_5000.txt", delimiter=',')
Y_test_scores_2 = np.loadtxt("class_sums/IMDBAnalyzer_1_10000_8000_2.00_3_3_32_1_char_wb_0.75_3000.txt", delimiter=',')

votes = np.zeros(Y_test_scores_1.shape, dtype=np.float32)
for i in range(Y_test.shape[0]):
    votes[i] += 1.0*Y_test_scores_1[i]/(np.max(Y_test_scores_1) - np.min(Y_test_scores_1))
    votes[i] += 1.0*Y_test_scores_2[i]/(np.max(Y_test_scores_2) - np.min(Y_test_scores_2))

Y_test_predicted = votes.argmax(axis=1)

print("Team Accuracy: %.1f" % (100*(Y_test_predicted == Y_test).mean()))