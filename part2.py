from collections import defaultdict
from preprocess import *

# qn 1

def get_count_y(train):
    count_y = defaultdict(int)
    for line in train:
        for obs_label in line:
            count_y[obs_label[1]] += 1
    return count_y

def get_emission(train):
    count_y = get_count_y(train)
    emission = defaultdict(float)
    for line in train:
        for obs_label in line:
            emission[obs_label] += 1.0 / count_y[obs_label[1]]

    #for e in emission: # smoothing
    #    emission[e] += 1 / count_y[e[1]]

    return emission


# qn 2

def modified_data(train, k):
    count_obs = defaultdict(int)
    for line in train:
        for obs_label in line:
            count_obs[obs_label[0]] += 1

    return [[obs_label if count_obs[obs_label[0]] >= k else
            ('#UNK#', obs_label[1]) for obs_label in line]
            for line in train]

def get_emission2(train, k):
    count_y = defaultdict(int)
    count_obs = defaultdict(int)
    count_obs_label = defaultdict(int)
    emission = defaultdict(float)
    for line in train:
        for obs_label in line:
            count_y[obs_label[1]] += 1
            count_obs[obs_label[0]] += 1
            count_obs_label[obs_label] += 1

    for obs_label, count in count_obs_label.items():
        if count_obs[obs_label[0]] < k:
            emission[('#UNK#', obs_label[1])] += count
        else:
            emission[obs_label] += count

    for obs_label, count in emission.items():
        emission[obs_label] = count / count_y[obs_label[1]]

    return emission


# qn 3

def max_em_label(word, possible_labels, emissions, words):
    if word not in words:
        word = '#UNK#'
    return max((emissions.get((word, label), 0), label) for label in possible_labels)[1]

def predict_max_em(test, emissions):
    words, labels = tuple(set(i) for i in zip(*emissions.keys()))
    return [[(word, max_em_label(word, labels, emissions, words)) for word in line] for line in test]


if __name__ == '__main__':
    outfile = '/dev.p2.out'
    for lang in languages:
        # qn 1
        train = data_from_file(lang + '/train') # A list of list of tuples. Each list in train is a sentence, each tuple in a sentence is a (word, tag).
        emission = get_emission(train)

        # qn 2
        new_emission = get_emission2(train, 3)

        # qn 3
        labels = list(get_count_y(train))
        test = data_from_file(lang + '/dev.in') # A list of list of tuples of size 1. Each list in test is a sentence.
        test = [[word[0] for word in line] for line in test]
        prediction = predict_max_em(test, new_emission)
        write_predictions(prediction, lang, outfile)

        pred = get_entities(open(lang+outfile, encoding='utf-8'))
        gold = get_entities(open(lang+'/dev.out', encoding='utf-8'))
        print(lang)
        compare_result(gold, pred)
        print()
