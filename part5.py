import part2, part3, part4
from collections import defaultdict
from functools import lru_cache
from preprocess import *

def traink(k, train, test, kew, lang, outfile, func):
    emission = part2.get_emission2(train, k)
    mod_test = [[word[0] for word in line] for line in test]
    prediction = func(mod_test, kew, emission)

    write_predictions(prediction, lang, outfile)

    pred = get_entities(open(lang+outfile, encoding='utf-8'))
    gold = get_entities(open(lang+'/dev.out', encoding='utf-8'))

    return compare_result2(gold, pred)


if __name__ == '__main__':
    outfile = '/test.p5.out'
    for lang in languagesP4:
        train = data_from_file2(lang + '/train')
        kew = part3.q(train)

        test = data_from_file2(lang + '/dev.in')

        vit = part3.predict_viterbi_decode
        maxmarginal = part4.predict_max_marginal_decode

        highestE = 0
        highestS = 0

        for k in range(1, 10):
            E, S = traink(k, train, test, kew, lang, outfile, vit)
            if E > highestE and S > highestS:
                highestE = E
                highestS = S
                bestk = k
        print(lang)
        print(bestk)
        print('Viterbi best: %f\t%f' % (highestE, highestS))

        test = data_from_file2(lang + '/test.in')
        traink(bestk, train, test, kew, lang, outfile, vit)
        print()
