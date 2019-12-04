import operator
import json
import math
import pandas as pd
import numpy as np

def Entropy(data, Cls):
    DataLength, Entropy = len(data), 0
    for x in np.unique(data[Cls]):
        Temp = len(x)/DataLength
        Entropy -= (Temp)*math.log2(Temp)
    return Entropy

dataframe = pd.read_csv('dec.csv')
dataframe = dataframe.drop('ID', axis=1)
ClassIndex = len(dataframe.columns.values)-1
Class = dataframe.columns.values[ClassIndex]
Columns = list(dataframe.columns.values[:len(dataframe.columns.values)-1])
AllOutputLabels = np.unique(dataframe[Class])

def Construct(data, datasize, label, Cls):
    # check if the reduced dataset contains only one class as output
    for C in np.unique(data[Cls]):
        if len(data) == np.count_nonzero(data[Cls] == C):
            print('The data has only',C)
            print(data)
            return C
    # else filter out using entropy
    Entr = Entropy(data, Cls)
    print('Entropy for label', label+":", Entr)
    Gain = {}
    # iterate over all the attributes
    for x in data.columns.values[:len(data.columns.values)-1]:
        uniques, ActualEntropy = np.unique(data[x]), 0
        # for each unique labels for an attribute, calculate I(x,n) and overall entropy
        for y in uniques:
            Value = Entropy(data[data[x] == y], Cls)
            ActualEntropy += Value*np.count_nonzero(data[x] == y) / datasize
        # store the information gain in dictionary
        Gain[x] = Entr-ActualEntropy
    # obtain the attribute with max info gain from the dictionary
    Attribute = max(Gain.items(), key=operator.itemgetter(1))[0]
    print('Max Info gain: {}, {}'.format(Attribute, Gain[Attribute]))
    # print(data)
    # recur over every unique labels of the attribute having max entropy
    return {Attribute: {y: Construct(data[data[Attribute] == y]
                                            .drop(Attribute, axis=1), datasize, Attribute, Cls) for y in np.unique(data[Attribute])}}

def Predict(row, tree):
    if type(tree) is str:
        return tree
    for x in Columns:
        if tree.get(x) is not None:
            Dictionary = tree[x][row[Columns.index(x)]]
            return Predict(row, Dictionary)

if __name__ == "__main__":
    Tree = Construct(dataframe, len(dataframe), Class, Class)
    ans = Predict(["<21", "Low", "Female", "Married"], Tree)
    print('#################################################')
    print(json.dumps(Tree, indent=4))
    print('#################################################')
    print(ans)
