# get_entities and compare_result functions were not written by us

import re
import string
from collections import defaultdict

languages = ('CN', 'EN', 'FR', 'SG')
languagesP4 = languages[1:3]

def data_from_file(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        obs_and_labels = []
        sentence = []
        for line in f:
            if line == '\n' or line == '\r\n':
                obs_and_labels.append(sentence)
                sentence = []
            else:
                obs_label = line.strip().split(' ')
                if len(obs_label) > 2:
                    obs_label = (' '.join(obs_label[:-1]), obs_label[-1])
                sentence.append(tuple(obs_label))
    return obs_and_labels

def data_from_file2(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        obs_and_labels = []
        sentence = []
        regex = re.compile(r'[^@]+@[^@]+\.[^@]+')
        for line in f:
            if line in string.whitespace:
            #if line == '\n' or line == '\r\n':
                obs_and_labels.append(sentence)
                sentence = []
            else:
                sentence.append(process_raw_line(line))
    return process_sentences(obs_and_labels)

def process_raw_line(line):
    obs_label = tuple(line.strip().split(' '))
    if len(obs_label) > 2:
        obs_label = (' '.join(obs_label[:-1]), obs_label[-1])
    return obs_label

def process_sentences(obs_and_labels):
    test = len(obs_and_labels[0][0]) == 1
    return [[(clean(obs_label[0]),) if test else (clean(obs_label[0]), obs_label[1])\
            for obs_label in sentence] for sentence in obs_and_labels]

def clean(word):
    result = word
    if word.isdigit():
        try:
            if 1800 <= int(word) <= 2100:
                result = '#YEAR#'
            else:
                result = '#DIGIT#'
        except:
            result = '#DIGIT#'
    elif all([char in string.punctuation or char == ' ' for char in word]):
        result = '#PUNCT#'
    elif 'http' in word or 'www' in word:
        result = '#WEBSITE#'
    elif re.search(r'[^@]+@[^@]+\.[^@]+', word):
        result = '#EMAIL#'
    elif '#' in word:
        result = '#HASH#'
    return result.lower() if result[0] + result[-1] != '##' else result

def write_predictions(prediction, folder, filename):
    with (open(folder + filename, 'w', encoding='UTF-8')) as f:
        for sentence in prediction:
            for word in sentence:
                result = word[0] + ' ' + word[1] + '\n'
                f.write(result)
            f.write('\n')

def prec_rec_F1(correct, predicted, gold):
    precision = correct / predicted
    recall = correct / gold
    F1 = 2 * precision * recall / (precision+recall)
    return precision, recall, F1

def print_prec_rec_F1(name, prec, rec, F1):
    print(name)
    print('Precision: ' + str(prec))
    print('Recall: ' + str(rec))
    print('F1: ' + str(F1))

def get_entities(observed, sep=' ', output_col=1):
    """Get entities from file."""
    example = 0
    word_index = 0
    entity = []
    last_ne = 'O'
    last_sent = ''
    last_entity = []

    observations = defaultdict(defaultdict)
    observations[example] = []

    for line in observed:
        line = line.strip()
        if line.startswith('##'):
            continue
        elif len(line) == 0:
            if entity:
                observations[example].append(list(entity))
                entity = []

            example += 1
            observations[example] = []
            word_index = 0
            last_ne = 'O'
            continue

        split_line = line.split(sep)
        # word = split_line[0]
        value = split_line[output_col]
        ne = value[0]
        sent = value[2:]

        last_entity = []

        # check if it is start of entity
        if ne == 'B' or (ne == 'I' and last_ne == 'O') or \
                (ne == 'I' and last_ne != 'O' and last_sent != sent):
            if entity:
                last_entity = entity
            entity = [sent]
            entity.append(word_index)
        elif ne == 'I':
            entity.append(word_index)
        elif ne == 'O':
            if last_ne == 'B' or last_ne == 'I':
                last_entity = entity
            entity = []

        if last_entity:
            observations[example].append(list(last_entity))
            last_entity = []

        last_ne = ne
        last_sent = sent
        word_index += 1

    if entity:
        observations[example].append(list(entity))

    return observations

def compare_result(observed, predicted):
    """Compare bewteen gold data and prediction data."""
    total_observed = 0
    total_predicted = 0
    correct = {'entities': 0, 'sentiment': 0}

    for example in observed:
        observed_instance = observed[example]
        predicted_instance = predicted[example]
        total_observed += len(observed_instance)
        total_predicted += len(predicted_instance)

        for span in predicted_instance:
            span_sent = span[0]
            span_ne = (span[1], len(span) - 1)

            for observed_span in observed_instance:
                sent = observed_span[0]
                ne = (observed_span[1], len(observed_span) - 1)

                if span_ne == ne:
                    correct['entities'] += 1
                    if span_sent == sent:
                        correct['sentiment'] += 1

    print('Entities in gold data : %d' % total_observed)
    print('Entities in prediction: %d' % total_predicted)

    for t in ('entities', 'sentiment'):
        print()
        prec = correct[t] / total_predicted
        recl = correct[t] / total_observed
        try:
            f = 2 * prec * recl / (prec + recl)
        except ZeroDivisionError:
            f = 0
        print(t)
        print(' - Correct   : %d' % correct[t])
        print(' - Precision : %.4f' % prec)
        print(' - Recall    : %.4f' % recl)
        print(' - F score   : %.4f' % f)


if __name__ == '__main__':
    en_train = data_from_file2(languages[1] + '/train')
    print(len(en_train[0]))
