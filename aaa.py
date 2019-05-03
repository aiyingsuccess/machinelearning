from id3_c45 import DecisionTree
import random

if __name__ == '__main__':
    # Toy data

    def rand_pick(seq, probabilities):
        x = random.uniform(0, 1)
        cumprob = 0.0
        for item, item_pro in zip(seq, probabilities):
            cumprob += item_pro
            if x < cumprob:
                break
        return item


    def data_once(k):
        value_list = [0, 1]
        probabilities = [0.5, 0.5]
        arrayx = []
        arrayy = []

        x1 = rand_pick(value_list, probabilities)
        xchange = x1
        arrayx.append(x1)

        sum1 = 0
        for num in range(1, k):
            sum1 = sum1 + 0.9 ** (i + 1)

        sum2 = 0
        for num in range(1, k):
            value_list = [xchange, 1 - xchange]
            probabilities = [0.75, 0.25]
            x2 = rand_pick(value_list, probabilities)
            arrayx.append(x2)
            sum2 += 0.9 ** (i + 1) / sum1 * x2
            xchange = x2
        print(sum2)
        if sum2 >= 0.5:
            y1 = x1
        else:
            y1 = 1-x1
        arrayy.append(y1)
#        print(arrayy)
#        print(arrayx)
        return arrayx + arrayy


    X = []
    y = []
    for i in range(30):
        DATA = data_once(4)
        print(DATA)
        X.append(DATA[0:4])
        if DATA[4] == 1:
            y.append('YES')
        else:
            y.append('NO')
    print(X)
    print(y)
#    data(30, 4)
    
    """
    X = [[1, 2, 0, 1, 0],
           [0, 1, 1, 0, 1],
           [1, 0, 0, 0, 1],
           [2, 1, 1, 0, 1],
           [1, 1, 0, 1, 1]]
    y = ['yes','yes','no','no','no']

    """
    clf = DecisionTree(mode='ID3')
    clf.fit(X, y)
    clf.show()
    print(clf.predict(X))  # ['yes' 'yes' 'no' 'no' 'no']

    clf_ = DecisionTree(mode='C4.5')
    clf_.fit(X, y).show()
    print(clf_.predict(X))  # ['yes' 'yes' 'no' 'no' 'no']

    