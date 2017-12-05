import part2
import part3

if __name__ == '__main__':
    obs = [['b','c','a','b'],['a','b','a'],['b','c','a','b','d'],['c','b','a'],
            ['c','a'],['d'],['d','b']]
    tags = [['X','X','Z','X'],['X','Z','Y'],['Z','Y','X','Z','Y'],['Z','Z','Y'],
            ['X','X'],['Z'],['Z','Z']]

    data = [tuple(zip(obs[i],tags[i])) for i in range(len(obs))]

    emissions = part2.get_emission(data)
    print('Emissions:')
    for k, v in emissions.items():
        print(k, v)

    transitions = part3.q(data)
    print('Transitions:')
    for k, v in transitions.items():
        print(k, v)
