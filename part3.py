import part2
from collections import defaultdict
from functools import lru_cache
from preprocess import *

# qn 1

# Returns the transition params

def q(train):
    result = defaultdict(float)
    count_yim1 = part2.get_count_y(train)
    count_yim1.update({'STARTNOW':len(train)})

    for line in train:
        result[('STARTNOW', line[0][1])] += 1.0
        result[(line[-1][1], 'STOPNOW')] += 1.0

        previous = line[0][1]   # Transition parameters for all other observations
        for obs_label in line[1:]:
            result[(previous, obs_label[1])] += 1.0
            previous = obs_label[1]

    for obs_label, count in result.items():
        result[obs_label] = count / count_yim1[obs_label[0]]

    return result


# qn 2

def viterbi(sentence, words, possible_labels, a, em):

    @lru_cache(maxsize=128) # Cache all results for the current sentence
    def pai(k, v):
        word = sentence[k-1]
        word = '#UNK#' if word not in words else word

        if k > 1:

            # Return the maximum of pai(k-1, u) * transition(u -> v) * emission(word, v) for all u in possible_labels
            # Multiply result by 100 to prevent underflow

            return max([pai(k-1, u) * a[(u, v)] * em.get((word, v), 0)\
                        for u in possible_labels]) * 100

        elif k == 1:
            return a[('STARTNOW', v)] * em.get((word, v), 0) * 100

        else:
            return 100

    def backtrack(k, next_label):
        probabilities = {label : pai(k, label) * a[(label, next_label)] for label in possible_labels}
        return max(probabilities, key=lambda key: probabilities[key])   # Returns label with the highest probability

    next_label = 'STOPNOW'
    seq = []

    for k in range(len(sentence), 0, -1):   # Start at the last word, loop backwards until k = 1
        label = backtrack(k, next_label)
        next_label = label  # Move to previous word
        seq.append(label)

    return seq[::-1]

def predict_viterbi_decode(test, trans, em):
    words, labels = tuple(set(i) for i in zip(*em.keys()))
    prediction = [list(zip(line, viterbi(line, words, labels, trans, em))) for line in test]
    return prediction


if __name__ == '__main__':
    outfile = '/dev.p3.out'
    for lang in languages:
        # qn 1
        train = data_from_file(lang + '/train')
        kew = q(train)

        # qn 2
        emission = part2.get_emission2(train, 3)
        test = data_from_file(lang + '/dev.in')

        mod_test = [[word[0] for word in line] for line in test]
        prediction = predict_viterbi_decode(mod_test, kew, emission)

        write_predictions(prediction, lang, outfile)

        pred = get_entities(open(lang+outfile, encoding='utf-8'))
        gold = get_entities(open(lang+'/dev.out', encoding='utf-8'))
        print(lang)
        compare_result(gold, pred)
        print()
