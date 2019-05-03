# For Python 2 / 3 compatability
from __future__ import print_function
import random
import matplotlib.pyplot as plt

k=10
weight=[]
weight.append(0)
s=0
for i in range(2,k+1):
    s+=0.9**i
for j in range(2,k+1):
    weight.append(0.9**j/s)
print(weight)

def computey(k,row):
    return sum([m*n for m in weight for n in row])

def generatedata(m,k):
    data=[]
    for i in range(m):
        row=[]
        seed=random.random()
        if(seed<0.5):
            x1=0
        else:
            x1=1
        row.append(x1)
        for j in range(k-1):
            if(random.random()<0.75):
                row.append(x1)
            else:
                row.append(1-x1)
        if(computey(k,row)>=0.5):
            row.append(x1)
        else:
            row.append(1-x1)
        data.append(row)
    return data
                
header = ["x"+i for i in list(map(str,list(range(1,k+1))))]
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


def build_tree(rows,col):
    question = Question(col, 1)

    true_rows, false_rows = partition(rows, question)

    if len(true_rows) == 0 or len(false_rows) == 0:
        return Leaf(rows)
    # Recursively build the true branch.
    true_branch = build_tree(true_rows,col+1)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows,col+1)

    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return
  
    if isinstance(node, int):
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
        return node.predictions

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

if __name__ == '__main__':
    global col
    col=0
    Result=[]
    M=[]
    for m in range(30,200,10):
        M.append(m)
        average=0.0
        for repeat in range(50):
            training_data=generatedata(m,k)
            my_tree = build_tree(training_data,col)
            #print_tree(my_tree)

            # Evaluate
            testing_data = generatedata(m,k)
            accurate=0
            for row in testing_data:
                #print ("Actual: %s. Predicted: %s" %(row[-1], print_leaf(classify(row, my_tree))))
                if list(classify(row,my_tree).keys())[0]==row[-1]:
                    accurate=accurate+1
            average=average+accurate/len(testing_data)
        # print("average error rate",average/(repeat+1))
        Result.append(1-average/(repeat+1))
    print(Result)
    print(M)
    plt.plot(M, Result)
    plt.show()