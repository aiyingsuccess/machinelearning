from __future__ import print_function
from pandas import read_csv

import random 
import math
import numpy as np
import pandas as pd

from random import randint
import matplotlib.pyplot as plt
dataset = read_csv('/home/aiying/Machinelearning/dataorigin.csv')

headers = list(dataset)
ds=dataset.values.tolist()              

limit=int(144/5)
validrow=[]
rowsum=list(dataset.isnull().sum(axis=1))
for i in range(len(rowsum)):
    if rowsum[i]<limit:
        validrow.append(i)    

validds=[]
for i in validrow:
    validds.append(ds[i])    


order=[]
for i in range(41):
    order=order+random.sample(range(0,144), 144)    

    n=0
for row in validds[0:len(validds)]:
    while math.isnan(row[order[n]]):
        order[n]=(order[n]+1)%144
    row[order[n]]=np.NaN  
    validds.append(row)
    n=n+1

validdataset=pd.DataFrame(validds)

headers = list(validdataset)
ds=validdataset.values.tolist()              
 
modset=[]
modframe=[]
modframelen=[]
testcolumnameset=[]
testcolumnset=[]
j=-1
for i in headers:
     j=j+1
     indexNames = validdataset[i].index[validdataset[i].apply(np.isnan)]
     if len(indexNames)==0:
         continue
     newds=validdataset.drop(indexNames)
     modframe.append(newds)
     lnewds=newds.values.tolist()
     modset.append(lnewds)
     modframelen.append(len(lnewds))
     testcolumnameset.append(i)
     testcolumnset.append(j)
print('sum of columns have missing data', len(modset))
print('shortest column',min(modframelen))

testcolumn=0
colrefine=[0,1,29,30,37,38,41,42,58,59,60,61,62,63,64,65,66,67,71,72,91,92,99,100,101,102,120,121,123,124,125,126,127,128,130,131]

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""  
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[testcolumn]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

class Question:
    """A Question is used to partition a dataset.
    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in thequestion
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):    
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return 2

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            headers[self.column], condition, str(self.value))

def partition(rows, question):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows, N_rows= [],[],[]
    for row in rows:
        if question.match(row)==True:
            true_rows.append(row)
        elif question.match(row)==2:
            N_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows, N_rows

def entropy(rows):
    #Entropy
    counts = class_counts(rows)
    origin = 0.0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        prob_of_rbl=1.0- prob_of_lbl
        if prob_of_lbl==0.0 or prob_of_lbl==1.0:
            return 0.0
        origin -= (prob_of_lbl*np.log2(prob_of_lbl)+prob_of_rbl*np.log2(prob_of_rbl))
    return origin


def info_gain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)

def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature
        if col==testcolumn:
            continue
        values=list(pd.unique(modframe[testcolumnset.index(testcolumn)][headers[testcolumn]]))  # unique values in the column
        for val in values:  # for each value
            if math.isnan(val):
                continue

            question = Question(col, val)

            true_rows, false_rows, N_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch,
                 N_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.N_branch=N_branch


def build_tree(rows):
   
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows, N_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    if len(N_rows)>0:
        N_branch = build_tree(N_rows)
    else:
        if len(true_rows)>=len(false_rows):
            N_branch=true_branch
        else:
            N_branch=false_branch          
    return Decision_Node(question, true_branch, false_branch, N_branch)

def build_treelimitsize(rows,s):
   
    if len(rows)<=s:
        return Leaf(rows)

    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows,N_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_treelimitsize(true_rows,s)

    # Recursively build the false branch.
    false_branch = build_treelimitsize(false_rows,s)

    if len(N_rows)>0:
        N_branch = build_treelimitsize(N_rows,s)
    else:
        if len(true_rows)>=len(false_rows):
            N_branch=true_branch
        else:
            N_branch=false_branch          

    return Decision_Node(question, true_branch, false_branch,N_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

    print (spacing + '--> Unkown:')
    print_tree(node.N_branch, spacing + "  ")


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row)==True:
        return classify(row, node.true_branch)
    elif node.question.match(row)==2:
        return classify(row, node.N_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

def accuracyoftreeall(percent):
    Accurate=[]
    M=[]
    for i in colrefine:
        m=int(len(modset[i])*0.8*percent)
        n=int(m/4)
        M.append(m)
        training_data=modset[i][0:m]
        global testcolumn
        testcolumn=i
        my_tree = build_tree(training_data)
        # print_tree(my_tree)
        print('testclumn',testcolumn)
        print('len of training data',m)
        print('len of testing data',n)
        # Evaluate
        testing_data = modset[i][m:m+n]
        accurate=0
        for row in testing_data:
            print ("Actual: %s. Predicted: %s" %
                (row[testcolumn], print_leaf(classify(row, my_tree))))
            if list(classify(row,my_tree).keys())[0]==row[testcolumn] and len(list(classify(row,my_tree).keys()))==1:
                accurate=accurate+1
        Accurate.append(accurate/len(testing_data))    
        print('accurate rate is',accurate/len(testing_data))
    print(Accurate)
    with open('refine'+'txt', 'w') as f:
        for item in Accurate:
            f.write("%s\t" % headers[i])
            f.write("%s\t" % i)
            f.write("%s\n" % item)
    with open('refinedifferent_m_of_column'+'txt', 'w') as f:
        for item in M:
            f.write("%s\n" % item)            

def accuracyoftree(per):
        m=int(len(modset[0])*5/6*per)
        n=int(m/5)
        training_data=modset[0][0:m]
        global testcolumn
        testcolumn=testcolumnset[0]
        my_tree = build_tree(training_data)
        # print_tree(my_tree)
        print('testclumn',testcolumn)
        print('len of training data',m)
        print('len of testing data',n)
        # Evaluate
        testing_data = modset[0][m:m+n]
        accurate=0
        for row in testing_data:
            print ("Actual: %s. Predicted: %s" %
                (row[testcolumn], print_leaf(classify(row, my_tree))))
            if list(classify(row,my_tree).keys())[0]==row[testcolumn] and len(list(classify(row,my_tree).keys()))==1:
                accurate=accurate+1
        print('accurate rate is',accurate/len(testing_data))

def testm():
    Result=[]
    M=[]
    for m in range(30,200,10):
        M.append(m)
        average=0.0
        for repeat in range(50):                         #need to be set correctly to get the average value
            training_data=ds[0:50]
            my_tree = build_tree(training_data)

            # Evaluate
            testing_data = ds[51:71]
            accurate=0
            for row in testing_data:
                if list(classify(row,my_tree).keys())[0]==row[testcolumn]:
                    accurate=accurate+1
            average=average+accurate/len(testing_data)
            # print(accurate/len(testing_data))
        # print("average error rate",average/(repeat+1))
        Result.append(1-average/(repeat+1))
    print(Result)
    print(M)
    plt.plot(M, Result)
    plt.show()

accuracyoftreeall(1)




