# For Python 2 / 3 compatability
from __future__ import print_function
import random
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import chi2_contingency
from scipy.stats import chi2

k=21

def generatedata(m,k):
    data=[]
    for i in range(m):
        row=[]
        seed=random.random()
        if seed<0.5:
            x0=0
        else:
            x0=1
        row.append(x0)
        for j in range(14):
            if(random.random()<0.75):
                row.append(x0)
            else:
                row.append(1-x0)
        for l in range(6):
            seed=random.random()
            if seed<0.5:
                row.append(0)
            else:
                row.append(1)             
        if x0==0:
            row.append(Counter(row[2:8]).most_common()[0][0])
        else:
            row.append(Counter(row[9:15]).most_common()[0][0])
        data.append(row)
    return data
                

header = ["x"+i for i in list(map(str,list(range(0,k))))]
header.append("Y")

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""

    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def Endsplit(counts):
    a=list(counts.values())
    b=list(counts.keys())
    result=b[a.index(max(a))]
    return result

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

class Question:
    """A Question is used to partition a dataset.
    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
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
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

def partition(rows, question):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

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

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            true_rows, false_rows = partition(rows, question)

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
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows,noise):
   
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    if question.column>=16:
        noise[0]=noise[0]+1

    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows,noise)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows,noise)

    return Decision_Node(question, true_branch, false_branch)


def build_treelimitdepth(rows,noise,depth,d):
   
    gain, question = find_best_split(rows)

    if gain == 0 or depth[0]==d:
        return Leaf(rows)

    if question.column>=16:
        noise[0]=noise[0]+1

    true_rows, false_rows = partition(rows, question)
    depth[0]=depth[0]+1
    leftdepth=depth
    rightdepth=depth

    # Recursively build the true branch.
    true_branch = build_treelimitdepth(true_rows,noise,leftdepth,d)

    # Recursively build the false branch.
    false_branch = build_treelimitdepth(false_rows,noise,rightdepth,d)

    return Decision_Node(question, true_branch, false_branch)


def build_treelimitsize(rows,noise,s):
   
    gain, question = find_best_split(rows)

    if gain == 0 or len(rows)<=s:
        return Leaf(rows)

    if question.column>=16:
        # print("hhhhhhhhhhhhh find you noise")
        noise[0]=noise[0]+1

    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_treelimitsize(true_rows,noise,s)

    # Recursively build the false branch.
    false_branch = build_treelimitsize(false_rows,noise,s)

    return Decision_Node(question, true_branch, false_branch)

def kaitest(rows):             #x=1,0;y=1,0
    Prob=[]
    n=0
    for column in range(len(rows[0])-1):
        L=[[0,0],[0,0]]
        for row in rows:
            if row[column]==1 and row[-1]==1:
                L[0][0]=L[0][0]+1
            if row[column]==0 and row[-1]==1:
                L[0][1]=L[0][1]+1
            if row[column]==1 and row[-1]==0:
                L[1][0]=L[1][0]+1
            if row[column]==0 and row[-1]==0:
                L[1][1]=L[1][1]+1
        stat,p,dof,expected=chi2_contingency(L)
        # print(L)
        Prob.append(p)
    # critical = chi2.ppf(0.95, dof)
    # print(critical)
    return Prob


def build_treelimitsignificance(rows,noise,Prob,P):
    gain, question = find_best_split(rows)

    if gain == 0 or Prob[question.column]>=1-P:
        return Leaf(rows)

    if question.column>=16:
        noise[0]=noise[0]+1

    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_treelimitsignificance(true_rows,noise,Prob,P)

    # Recursively build the false branch.
    false_branch = build_treelimitsignificance(false_rows,noise,Prob,P)

    return Decision_Node(question, true_branch, false_branch)


def print_tree(node,spacing=""):
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


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return Endsplit(node.predictions)

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

m=10000
training_data = generatedata(int(0.8*m),k)
testing_data = generatedata(int(0.2*m),k)

Prob=kaitest(training_data)
def Deeptest(): 
    Error=[]
    Noise=[]                          
    Deepthreshold=list(range(5,80,1))
    Depth=[]
    for i in Deepthreshold:
        n=[0]
        depth=[0]
        my_tree = build_treelimitdepth(training_data,n,depth,i)
        Depth.append(depth)
        Noise.append(n[0])
        accurate=0
        for row in testing_data:
            # print ("Actual: %s. Predicted: %s" %
            #     (row[-1], classify(row, my_tree)))
            classify(row, my_tree)
            if classify(row,my_tree)==row[-1]:
                accurate=accurate+1
        Error.append(1-accurate/len(testing_data))
    print("Error",Error)
    print("Actual depth",Depth)
    print("Noise",Noise)
    plt.plot(Deepthreshold, Error)
    plt.xticks(list(range(5,max(Deepthreshold)+1,5)),[str(i) for i in range(5,max(Deepthreshold)+1,5)])
    
    print(len(Error))
    print(len(Depth))
    plt.grid()
    plt.show()
    # plt.savefig('Deep'+'.png')
        
def Sizetest():
    Error=[]
    Noise=[]
    Sthreshold=list(range(0,800,10))
    for i in Sthreshold:
        n=[0]
        my_tree = build_treelimitsize(training_data,n,i)
        Noise.append(n[0])
        accurate=0
        for row in testing_data:
            # print ("Actual: %s. Predicted: %s" %
            #     (row[-1], classify(row, my_tree)))
            classify(row, my_tree)
            if classify(row,my_tree)==row[-1]:
                accurate=accurate+1
        Error.append(1-accurate/len(testing_data))
    print("Error",Error)
    print("Noise",Noise)
    plt.plot(Sthreshold, Error)
    plt.xticks(list(range(100,max(Sthreshold)+1,10)),[str(i) for i in range(100,max(Sthreshold)+1,10)])
    plt.grid()
    plt.show()
    # plt.savefig('Size'+'.png')
        
def Sigtest():
    Error=[]
    Noise=[]
    Pthreshold=list(np.arange(0.7,1.0,0.05)) 
    for i in Pthreshold:    
        n=[0]                              #important using list to change the n[0] value  
        my_tree = build_treelimitsignificance(training_data,n,Prob,i)
        Noise.append(n[0])
        accurate=0
        for row in testing_data:
            # print ("Actual: %s. Predicted: %s" %
            #     (row[-1], classify(row, my_tree)))
            classify(row, my_tree)
            if classify(row,my_tree)==row[-1]:
                accurate=accurate+1
        Error.append(1-accurate/len(testing_data))
    print("Error",Error)
    print("Noise",Noise)
    plt.plot(Pthreshold, Error)
    plt.xticks(list(np.arange(0.7,1.0,0.05)),[str(i) for i in np.arange(0.7,1.0,0.05)])
    plt.grid()
    plt.show()
    # plt.savefig('significance'+'.png')
    
def testm():
    Result=[]
    M=[]
    Noise=[]
    for m in range(1000,50000,100): 
        M.append(m)
        average=0.0
        sumnoise=0.0
        for repeat in range(20):                         #need to be set correctly to get the average value
            training=generatedata(m,k)
            n=[0]
            my_tree = build_tree(training,n)
            
            # Evaluate
            testing = generatedata(m,k)
            accurate=0
            for row in testing:
                if classify(row,my_tree)==row[-1]:
                # if list(classify(row,my_tree).keys())[0]==row[-1]:
                    accurate=accurate+1
            average=average+accurate/len(testing_data)
            sumnoise+=n[0]
            # print(accurate/len(testing_data))
        # print("average error rate",average/(repeat+1))
        Result.append(1-average/(repeat+1))
        Noise.append(sumnoise/(repeat+1))
    print(Result)
    print(M)
    plt.plot(M, Noise)
    plt.grid()
    plt.savefig('testm'+'.png')

def compare1():
    Result=[]
    M=[]
    Noise=[]
    Noise2=[]
    for m in range(1000,10000,300): 
        if m%100==0:
            print("success",m)                        #6000,7000,100
        M.append(m)
        average=0.0
        average2=0.0
        sumnoise=0
        sumnoise2=0
        for repeat in range(10):                         #need to be set correctly to get the average value
            training=generatedata(m,k)
            n=[0]
            n2=[0]
            my_tree = build_tree(training,n)
            depth=[0]
            my_treedeep=build_treelimitdepth(training,n2,depth,45)
            testing = generatedata(m,k)
            accurate=0
            accurate2=0
            for row in testing:
                if classify(row,my_tree)==row[-1]:
                # if list(classify(row,my_tree).keys())[0]==row[-1]:
                    accurate=accurate+1
                if classify(row,my_treedeep)==row[-1]:
                    accurate2=accurate2+1
            average=average+accurate/len(testing)
            average2=average2+accurate2/len(testing)

            sumnoise+=n[0]
            sumnoise+=n2[0]

            # print(accurate/len(testing_data))
        # print("average error rate",average/(repeat+1))
        Result.append(1-average/(repeat+1))
        Noise.append(sumnoise/(repeat+1))
        Noise2.append(sumnoise2/(repeat+1))

    fig,ax = plt.subplots()

    for Y in [Noise, Noise2]:
        ax.plot(M, Y)
    plt.grid()
    plt.savefig('compare1'+'.png')

def compare2():
    Result=[]
    M=[]
    Noise=[]
    Noise2=[]
    for m in range(1000,10000,300): 
        if m%100==0:
            print("success",m)                        #6000,7000,100
        M.append(m)
        average=0.0
        average2=0.0
        sumnoise=0
        sumnoise2=0
        data=generatedata(m,k)
        training = data[0:int(0.8*m)]
        testing = data[int(0.8*m)+1:m]

        for repeat in range(10):                         #need to be set correctly to get the average value
            n=[0]
            n2=[0]
            my_tree = build_tree(training,n)
            my_treesize=build_treelimitsize(training,n2,50)
            
            # accurate=0
            # accurate2=0
            # for row in testing:
            #     if classify(row,my_tree)==row[-1]:
            #     # if list(classify(row,my_tree).keys())[0]==row[-1]:
            #         accurate=accurate+1
            #     if classify(row,my_treesize)==row[-1]:
            #         accurate2=accurate2+1
            # average=average+accurate/len(testing)
            # average2=average2+accurate2/len(testing)

            sumnoise+=n[0]
            sumnoise+=n2[0]
        # Result.append(1-average/(repeat+1))
        Noise.append(sumnoise/(repeat+1))
        Noise2.append(sumnoise2/(repeat+1))
    
    fig,ax = plt.subplots()
    for Y in [Noise, Noise2]:
        ax.plot(M, Y)
    # plt.plot(M,Noise)
    plt.grid()
    plt.xlabel('M')
    plt.ylabel('Noise')
    plt.savefig('compare2'+'.png')

def compare3():
    Result=[]
    M=[]
    Noise=[]
    Noise2=[]
    for m in range(1000,10000,300): 
        if m%100==0:
            print("success",m)                        #6000,7000,100
        M.append(m)
        average=0.0
        average2=0.0
        sumnoise=0
        sumnoise2=0
        for repeat in range(50):                         #need to be set correctly to get the average value
            training=generatedata(m,k)
            n=[0]
            n2=[0]
            my_tree = build_tree(training,n)
            depth=[0]
            my_treesignificance=build_treelimitsignificance(training,n2,Prob,0.95)
            testing = generatedata(m,k)
            accurate=0
            accurate2=0
            for row in testing:
                if classify(row,my_tree)==row[-1]:
                # if list(classify(row,my_tree).keys())[0]==row[-1]:
                    accurate=accurate+1
                if classify(row,my_treesignificance)==row[-1]:
                    accurate2=accurate2+1
            average=average+accurate/len(testing)
            average2=average2+accurate2/len(testing)

            sumnoise+=n[0]
            sumnoise+=n2[0]

            # print(accurate/len(testing_data))
        # print("average error rate",average/(repeat+1))
        Result.append(1-average/(repeat+1))
        Noise.append(sumnoise/(repeat+1))
        Noise2.append(sumnoise2/(repeat+1))
    
    fig,ax = plt.subplots()

    for Y in [Noise, Noise2]:
        ax.plot(M, Y)
    plt.grid()
    plt.savefig('compare3'+'.png')

# Deeptest()
# Sizetest()
# Sigtest()
# testm()
# compare1()
compare2()
# compare3()

