import part2, part3
import random
from functools import lru_cache
from preprocess import *

def max_marginal(sentence, words, possible_labels, a, em):

    n = len(sentence)

    @lru_cache(maxsize=128)
    def alpha(u, j):
        word = sentence[j-2]
        word = '#UNK#' if word not in words else word

        if j > 1:
            return sum(alpha(v, j-1) * a[(v, u)] * em.get((word, v), 0) for v in possible_labels)
        else:
            return a[('STARTNOW', u)]

    @lru_cache(maxsize=128)
    def beta(u, j):
        word = sentence[j-1]
        word = '#UNK#' if word not in words else word

        if j < n:
            return sum(a[(u, v)] * em.get((word, u), 0) * beta(v, j+1) for v in possible_labels)
        else:
            return a[(u, 'STOPNOW')] * em.get((word, u), 0)

    splitrandom = random.randint(1, n)
    Z = sum([alpha(v, splitrandom) * beta(v, splitrandom) for v in possible_labels])
    y_star = []

    for i in range(1, n+1):
        estimates = {u : alpha(u, i) * beta(u, i) / Z for u in possible_labels}
        y_star.append(max(estimates.keys(), key=lambda key: estimates[key]))

    return y_star

def predict_max_marginal_decode(test, trans, em):
    words, labels = tuple(set(i) for i in zip(*em.keys()))
    prediction = [list(zip(line, max_marginal(line, words, labels, trans, em))) for line in mod_test]
    return prediction


if __name__ == '__main__':
    outfile = '/dev.p4.out'
    for lang in languagesP4:
        train = data_from_file(lang + '/train')
        kew = part3.q(train)
        emission = part2.get_emission2(train, 3)    # Using train to avoid getting emissions for #UNK#

        test = data_from_file(lang + '/dev.in')
        mod_test = [[word[0] for word in line] for line in test]

        prediction = predict_max_marginal_decode(mod_test, kew, emission)
        #prediction = [list(zip(line, max_marginal(line, kew, possible_labels, emission))) for line in mod_test]

        write_predictions(prediction, lang, outfile)

        pred = get_entities(open(lang+outfile, encoding='utf-8'))
        gold = get_entities(open(lang+'/dev.out', encoding='utf-8'))
        print(lang)
        compare_result(gold, pred)
        print()
