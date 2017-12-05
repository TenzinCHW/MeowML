import math
import random
from preprocess import *
from functools import lru_cache
from collections import defaultdict
from itertools import cycle


WINDOW_SIZE = 3

y = ('O', 'B-positive', 'B-negative', 'B-neutral', 'I-positive', 'I-negative', 'I-neutral')

def get_count_xy(train):
    count_x = defaultdict(int)
    count_y = defaultdict(int)
    for sentence in train:
        for obs_label in sentence:
            count_x[obs_label[0]] += 1
            count_y[obs_label[1]] += 1
    return count_x, count_y

def common_words(train, count_obs, min_freq, min_prop):
    '''Tuple of observations to be checked for unambiguous words.'''
    count_freq = defaultdict(float)
    for sentence in train:
        for obs_label in sentence:
            count_freq[obs_label] += 1

    for obs_label, freq in count_freq.items():
        count_freq[obs_label] = freq / count_obs[obs_label[0]]

    #count_all_x = sum(count_obs.values())

    most_common = tuple(obs_label for obs_label, val in count_freq.items()\
        if val * count_obs[obs_label[0]] >= min_freq and val >= min_prop)
    return most_common

def map_features(train, feat_funcs, window_sz):
    '''Outputs a list of dictionaries that map a features defined by
    functions in *feat_funcs to a real integer value for training set,
    as well as the max of each feature.'''
    #mean and standard deviation of each feature.'''
    feat_counters = [1 for _ in feat_funcs]
    featuremap = [{} for _ in feat_funcs]

    iterator = zip(cycle(range(len(feat_funcs))), iter_feat(train, feat_funcs, window_sz))

    for i, feat in iterator:
        if feat not in featuremap[i]:
            featuremap[i][feat] = feat_counters[i]
            feat_counters[i] += 1

    maximum = [max(feature.values()) for feature in featuremap]

    return featuremap, maximum #mean_feat, sd_feat

def iter_feat(data, feat_funcs, window_sz):
    '''Iterates over the features for each word in data.
    Yields one feature of the word at a time'''
    start = tuple('STARTNOW' + str(i) for i in range(window_sz))
    end = tuple('ENDNOW' + str(i) for i in range(window_sz))

    for sentence in data:
        full_sent = start + tuple(obs[0] for obs in sentence) + end
        full_tag = start + tuple(obs[1] for obs in sentence) + end
        for pos, obs_label in enumerate(sentence):
            for func in feat_funcs:
                yield func(pos, window_sz, full_sent, full_tag[pos:pos+window_sz])

def get_obs_feature(index, feat_funcs, window_sz, full_sent, tags, featuremap, maximum):
    '''Gets the features of the index-th word in full_sent.
    Assume full_sent has STARTNOW<#> and END<#> attached.'''
    assert(len(tags) == window_sz)  # Make sure number of previous tags == window_sz

    features = []
    norm_max = max(maximum)

    for i, feature in enumerate(featuremap):
        key = feat_funcs[i](index, window_sz, full_sent, tags)
        if key in feature:
            features.append(feature[key])
        else:
            feature[key] = max(feature.values()) + 1    # TODO change this to get value of a rare word
            features.append(feature[key])

    for i, feat_val in enumerate(features):
        features[i] = feat_val / maximum[i] * norm_max

    return tuple(features)

def iter_training_obs(train, feat_funcs, window_sz, featuremap, maximum):
    '''Iterates over training set, obtaining features and tags from each word.'''
    start = tuple('STARTNOW' + str(i) for i in range(window_sz))
    end = tuple('ENDNOW' + str(i) for i in range(window_sz))
    for sentence in train:
        sent = start + tuple(obs[0] for obs in sentence) + end
        tags = start + tuple(obs[1] for obs in sentence)# + end
        for i, obs in enumerate(sentence):
            position = y.index(tags[i+window_sz])
            why = tuple(1 if position == pos else 0 for pos in range(len(y)))
            yield get_obs_feature(i, feat_funcs, window_sz, sent, tags[i:i+window_sz], featuremap, maximum), why

def predict_test(network, test, feat_funcs, window_sz, featuremap, maximum, tag_dict):
    guesses = []
    start = tuple('STARTNOW' + str(i) for i in range(window_sz))
    end = tuple('ENDNOW' + str(i) for i in range(window_sz))
    for sentence in test:
        sent = start + tuple(obs[0] for obs in sentence) + end
        tags = start
        for i, obs in enumerate(sentence):
            if obs in tag_dict:
                guess = tag_dict[obs]
            else:
                inputs = get_obs_feature(i, feat_funcs, window_sz, sent, tags[i:i+window_sz], featuremap, maximum)
                guess = y[max(enumerate(network.predict(inputs)), key=lambda ls: ls[1])[0]]
            tags += tuple(guess)
        guesses.append(list(zip(sent[window_sz:-window_sz], tags[window_sz:])))
    return guesses

def get_feat_funcs(window_sz):
    '''Returns a tuple of functions that will each output a feature for every observation/label pair.
     pos/index (first arg) starts at the word's index - window_sz -> add window_sz to it'''

    words = tuple(lambda i, w_sz, f_s, t: f_s[i+j] for j in range(2*window_sz+1))
    tags = tuple(lambda i, w_sz, f_s, t: t[j] for j in range(window_sz))    # Will not add word's tag
    prefix = tuple(lambda i, w_sz, f_s, t: f_s[i+j][:3] for j in range(2*window_sz+1))
    suffix = tuple(lambda i, w_sz, f_s, t: f_s[i+j][-3:] for j in range(2*window_sz+1))

    feat_funcs = words + tags + prefix + suffix
    return feat_funcs

# Softmax function that prevents overflow
@lru_cache(maxsize=256)
def softmax(ind, inputs):
    max_in = max(inputs)
    e = [math.e ** (inputs[i] - max_in) for i in range(len(inputs))]
    temp = sum(e)
    return e[ind]/temp

class NLPHiddenLayer():
    __slots__ = ['weights', 'b', 'act', 'dact', 'x', 'del_y']
    def __init__(self, input_sz, num_nodes, act, dact):
        self.weights = [[random.random() for _ in range(input_sz)] for _ in range(num_nodes)]
        self.b = [1 for _ in range(num_nodes)]
        self.act = act
        self.dact = dact

    def forward(self, inputs):
        assert(len(inputs) == len(self.weights[0]))
        self.x = inputs
        return tuple(self.act(sum(self.weights[j][i] * inputs[i] + self.b[j] for i in range(len(self.weights[j])))) for j in range(len(self.weights)))

    def back_prop(self, next_layer, alpha=0.1):
        self.del_y = tuple(sum(next_layer.weights[i][j] * next_layer.del_y[i] for i in range(len(next_layer.weights))) * self.dact(next_layer.x[j]) for j in range(len(next_layer.x)))
        mean_del_y = sum(self.del_y) / len(self.del_y)
        for j in range(len(self.weights)):
            self.b[j] += alpha * mean_del_y
            for i in range(len(self.weights[0])):
                self.weights[j][i] += alpha * self.x[i] * self.del_y[j]

class NLPLog():
    __slots__ = ['weights', 'b', 'act', 'x', 'del_y']
    def __init__(self, input_sz, num_nodes):
        self.weights = [[0 for _ in range(input_sz)] for _ in range(num_nodes)]
        self.b = [0 for _ in range(num_nodes)]
        self.act = softmax

    def predict(self, inputs):
        temp = tuple(sum(self.weights[j][i] * inputs[i] + self.b[j] for i in range(len(self.weights[0]))) for j in range(len(self.weights)))
        return tuple(self.act(i, temp) for i, node in enumerate(self.weights))

    def forward(self, inputs):
        self.x = inputs
        return self.predict(inputs)

    def neg_log_like(self, inputs, y):
        activation = self.predict(inputs)
        x_entropy_tot = [y[i] * math.log(activation[i]) + (1-y[i]) * math.log(1-activation[i]) for i in range(len(activation))]
        return - sum(x_entropy_tot) / len(x_entropy_tot)

    def back_prop(self, inputs, y, alpha=0.1):
        '''Returns the del value for this layer and updates the weights in this layer.'''
        activation = self.predict(self.x)
        self.del_y = [y[i] - activation[i] for i in range(len(y))]
        mean_del_y = sum(self.del_y) / len(self.del_y)
        for j in range(len(self.weights)):
            self.b[j] += alpha * mean_del_y
            for i in range(len(self.weights[0])):
                self.weights[j][i] += alpha * self.x[i] * self.del_y[j]

class NLPNN():
    __slots__ = ['layers', 'output']
    def __init__(self, input_sz, output_sz, layer_dim=(100,), funcs=(lambda z: z*(z>0),), dfuncs=(lambda z: 1*(z>0),)):
        self.layers = []
        num_input = input_sz
        for i, num_nodes in enumerate(layer_dim):
            self.layers.append(NLPHiddenLayer(num_input, num_nodes, funcs[i], dfuncs[i]))
            num_input = num_nodes

        self.output = NLPLog(num_input, output_sz)

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return self.output.forward(inputs)

    def train(self, inputs_y, num_epoch=10):
        for _ in range(num_epoch):
            for i, inp in enumerate(inputs_y):  # inp has 2 values, first is the set of features, second is the tag
                guess = self.predict(inp[0])
                self.output.back_prop(inp[0], inp[1])
                next_layer = self.output
                for layer in reversed(self.layers):
                    layer.back_prop(next_layer)
                    next_layer = layer
            random.shuffle(inputs_y)


if __name__ == '__main__':
    outfile = '/dev.p5.out'
    for lang in languagesP4:
        train = data_from_file2(lang + '/train')
        x, y = get_count_xy(train)
        tag_dict = common_words(train, x, 20, 0.9)
        #meow, woof = create_obs_lab_map(x, y)
        #print(meow)
        #print(type(woof))

        feat_funcs = get_feat_funcs(WINDOW_SIZE)
        featuremap, maximum = map_features(train, feat_funcs, WINDOW_SIZE)
        layer_dim = (100,100)
        hidden_funcs = tuple(lambda z: z*(z>0) for _ in range(len(layer_dim)))
        dhidden_funcs = tuple(lambda z: 1*(z>0) for _ in range(len(layer_dim)))
        network = NLPNN(len(feat_funcs), len(y), layer_dim=layer_dim, funcs=hidden_funcs, dfuncs=dhidden_funcs)
        training_data = [item for item in iter_training_obs(train, feat_funcs, WINDOW_SIZE, featuremap, maximum)]
        network.train(training_data, num_epoch=100)

        test = data_from_file2(lang + '/dev.in')
        prediction = predict_test(network, test, feat_funcs, WINDOW_SIZE, featuremap, maximum, tag_dict)
        write_predictions(prediction, lang, outfile)

        pred = get_entities(open(lang+outfile, encoding='utf-8'))
        gold = get_entities(open(lang+'/dev.out', encoding='utf-8'))
        print(lang)
        compare_result(gold, pred)
        print()
