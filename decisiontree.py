import pandas as pd
import math
import numpy as np
import operator
import json

def Entropy(data):
    P, N = len(data[data['Buys'] == 'Yes']), len(data[data['Buys'] == 'No'])
    if P == 0 or N == 0:
        return 0
    return -P/(P+N)*math.log2(P/(P+N))-N/(P+N)*math.log2(N/(P+N))

dataframe = pd.read_csv('dec.csv')
dataframe = dataframe.drop('ID', axis=1)
Class = dataframe.columns.values[len(dataframe.columns.values)-1]
Columns = list(dataframe.columns.values[:len(dataframe.columns.values)-1])

def Construct(data, datasize, label, Class):
    # check if the reduced dataset contains only one class as output
    for C in np.unique(data[Class]):
        if len(data) == np.count_nonzero(data[Class] == C):
            print('The data has only',C)
            print(data)
            return C
    # else filter out using entropy
    Entr = Entropy(data)
    print('Entropy for label', label+":", Entr)
    Gain = {}
    # iterate over all the attributes
    for x in data.columns.values[:len(data.columns.values)-1]:
        uniques, ActualEntropy = np.unique(data[x]), 0
        # for each unique labels for an attribute, calculate I(x,n) and overall entropy
        for y in uniques:
            Value = Entropy(data[data[x] == y])
            ActualEntropy += Value*np.count_nonzero(data[x] == y) / datasize
        # store the information gain in dictionary
        Gain[x] = Entr-ActualEntropy
    # obtain the attribute with max info gain from the dictionary
    Attribute = max(Gain.items(), key=operator.itemgetter(1))[0]
    print('Max Info gain: {}, {}'.format(Attribute, Gain[Attribute]))
    # print(data)
    # recur over every unique labels of the attribute having max entropy
    return {Attribute: {y: Construct(data[data[Attribute] == y]
                                            .drop(Attribute, axis=1), datasize, Attribute, Class) for y in np.unique(data[Attribute])}}

def Predict(row, Tree):
    if type(Tree) is str:
        return Tree
    for x in Columns:
        if Tree.get(x) is not None:
            Dictionary = Tree[x][row[Columns.index(x)]]
            return Predict(row, Dictionary)

if __name__ == "__main__":
    Tree = Construct(dataframe, len(dataframe), Class, Class)
    ans = Predict(["<21", "Low", "Female", "Married"], Tree)
    print(json.dumps(Tree, indent=4))
    print(ans)