# author: Mihir Khatri
# HW4 CS540 fall 2020
import json
import os
import collections
import math


def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # TODO: add your code here
    words_in_file = []
    none_value = 0
    file = open(filepath, encoding='utf-8')
    for line in file:
        words_in_file.append(line.strip('\n'))
    total_words = dict(collections.Counter(words_in_file))  # create a dictionary with the frequency
    bow = total_words

    for key, value in total_words.copy().items():  # add all the other key value pairs
        if key not in vocab:
            bow.pop(key)
            none_value += value
            bow[eval('None')] = none_value
    return bow


def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    dataset = []
    # TODO: add your code here
    paths = os.listdir(directory)  # open 1st directory
    for i in paths:
        if i.__contains__('2016'):
            file_2016 = os.listdir(os.path.join(directory, '2016'))  # create path to files inside that directory
            for file in file_2016:
                entry = {'2016': create_bow(vocab, os.path.join(directory, '2016', file))}  # final path to file
                dataset.append(entry)
        if i.__contains__('2020'):
            file_2020 = os.listdir(os.path.join(directory, '2020'))
            for file in file_2020:
                entry = {'2020': create_bow(vocab, os.path.join(directory, '2020', file))}
                dataset.append(entry)

    return dataset


def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """
    vocab = []
    words = []
    # TODO: add your code here
    root_dir = directory
    file_2016 = os.listdir(os.path.join(directory, '2016'))
    file_2020 = os.listdir(os.path.join(directory, '2020'))
    for i in file_2016:
        with open(os.path.join(directory, '2016', i),
                  encoding="utf-8") as f:  # open file, strip of anything unnecessary
            for line in f:
                words.append(line.strip())
    for i in file_2020:
        with open(os.path.join(directory, '2020', i), encoding="utf-8") as f:
            for line in f:
                words.append(line.strip())
    counter = collections.Counter(sorted(words))  # get the frequency
    for key, value in counter.items():
        if value >= cutoff:
            vocab.append(key)

    return sorted(vocab)


def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """
    smooth = 1  # smoothing factor
    logprob = {}
    # TODO: add your code here
    count_2016 = 0
    count_2020 = 0
    for i in training_data:  # find the count of the words for that year
        if i.keys().__contains__('2016'):
            count_2016 += 1
        if i.keys().__contains__('2020'):
            count_2020 += 1
    prob_2016 = (count_2016 + 1) / (len(training_data) + 2)  # probability of words for the year
    prob_2020 = (count_2020 + 1) / (len(training_data) + 2)
    logprob['2020'] = math.log(prob_2020)  # log probability dictionary entry
    logprob['2016'] = math.log(prob_2016)
    return logprob


def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1  # smoothing factor
    count = 0
    count_out = 0
    words_2016 = 0
    words_2020 = 0
    word_prob = {}
    count_none = 0
    word_freq = {}  # freq of words in vocab for that year
    for i in training_data:  # this will only find the frequency for the words in that year for vocab
        for key, value in i.items():
            if key == label:
                for key1, value1 in value.items():
                    if key1 in word_freq:
                        word_freq[key1] += value1
                    else:
                        count += value1
                        word_freq[key1] = count
                    count = 0
    for i in vocab:  # frequency of the words left out
        if i not in word_freq.keys():
            word_freq[i] = count_out
    for i in training_data:
        for key, value in i.items():
            if key == label:
                for key1, value1 in value.items():
                    if key1 is None:
                        count_none += value1
    word_freq[eval('None')] = count_none  # cannot iterate over TypeNone so done separately
    for i in training_data:  # total words for each year
        for j in i.keys():
            for k in i.values():
                if j == '2016':
                    words_2016 += sum(k.values())
                else:
                    words_2020 += sum(k.values())
    for key, value in word_freq.items():  # calculate probability and return in dictionary
        if label == '2016':
            word_prob[key] = math.log((value + 1) / (words_2016 + smooth * (len(vocab) + 1)))
        if label == '2020':
            word_prob[key] = math.log((value + 1) / (words_2020 + smooth * (len(vocab) + 1)))

    return word_prob


##################################################################################
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    # TODO: add your code here
    # calls of all variables needed and added to dictionary with the key names as needed
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    retval['vocabulary'] = vocab
    retval['log prior'] = prior(training_data, ['2020', '2016'])
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, training_data, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, '2020')

    return retval


def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>, 
             'log p(y=2016|x)': <log probability of 2016 label for the document>, 
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    total_2016 = 0
    total_2020 = 0
    bow = create_bow(model['vocabulary'], filepath)  # create bow needed
    for i in bow.keys():  # all additions to the total are made according the the formula given
        total_2016 += (bow[i] * model['log p(w|y=2016)'][i])  # how many times it occurs
        total_2020 += (bow[i] * model['log p(w|y=2020)'][i])
    total_2020 += model['log prior']['2020']  # log a + log b
    total_2016 += model['log prior']['2016']
    if total_2016 > total_2020:
        year = '2016'
    else:
        year = '2020'
    retval['log p(y=2020|x)'] = total_2020  # add in the format needed to be displayed
    retval['log p(y=2016|x)'] = total_2016
    retval['predicted y'] = year
    return retval

