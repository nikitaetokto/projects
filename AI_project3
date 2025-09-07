'''
                            INTRODUCTION TO ARTIFICIAL INTELLIGENCE - LAB ASSIGNMENT 3
                                            ZIBOROV NIKITA, FER UNIZG

used literature sources:
1. Lecture 10 (IAI 2025) - Intranet FER UNIZG
2. https://www.geeksforgeeks.org/sklearn-iterative-dichotomiser-3-id3-algorithms/
3. https://docs.python.org/3/library/csv.html
4. https://www.geeksforgeeks.org/counters-in-python-set-2-accessing-counters/
5. https://www.geeksforgeeks.org/self-in-python-class/
6. https://www.youtube.com/watch?v=HQNiSfb795A (Python Lambda Functions Explained)
7. https://gist.github.com/psambit9791/dc0eead27e39e5abfed70677b914891b
8. https://docs.python.org/3/library/collections.html
9. https://medium.com/@prithivrk7/decoding-the-id3-algorithm-in-5-minutes-f807e721ad0e
10. https://docs.python.org/3/tutorial/classes.html
11. https://discourse.opengenus.org/t/using-id3-algorithm-to-build-a-decision-tree-to-predict-the-weather/3343
12. https://www.geeksforgeeks.org/confusion-matrix-machine-learning/

NOTE: I tried to implement algorithm in the way i understood it, it can be "specific" from smb's angle.
(because of studying different literature sources). It can be "strange" making so many comments from me,
but it is just about implementing and explaining my way of thinking. If anybody reads this, thanks for understanding! ;)
                                                                                                                    '''
import math
import argparse
import csv

def readfile(filename): #function to read data from csv file
    f = open(filename, 'r')
    reader = csv.reader(f) #method reader for csv files - used from source 3
    data = list(reader)  #rearranging data - can not work with reader return (iterator)
    names = data[0]
    values = data[1:] #first line is header, should be separated from data (obvious)
    return names, values

class ID3: #from lab task: "...learning algorithm should be implemented as a separate class..." (source 10)
    #initializer - from lab task: "1. A constructor which accepts and stores the algorithm hyperparameters"
    def __init__(self, depth):
        self.depth = depth #depth is so-called hyperparameter for ID3-limited part (from lecturer)
        self.tree = None #self as basis of OOP - necessary to working with methods and variables that are "inside" the class
        self.names = None

    def entropy(self, values): #function to calculate entropy - first step for ID3
        entropy = 0
        for i in set(values):
            p = values.count(i) / len(values)
            entropy -= p * math.log(p, 2) #that's where math lib is used
        return entropy

    def IG(self, values, x, y): #function for information gain - entropy is helpful RIGHT HERE
        labels = [row[y] for row in values] #so-called list comprehension - different from "standard" for-cycle...
        features = [row[x] for row in values]
        entropy2 = 0 #from lecture 10 - "expected entropy after splitting the dataset by values of the feature"
        for value in set(features): #set is used for operating without duplicate features
            subset = [row for row in values if row[x] == value]
            entropy2 += (len(subset) / len(values)) * self.entropy([row[y] for row in subset])
        return self.entropy(labels) - entropy2

    def algorithm(self, values, parent_values, features, y, depth=0): #initially our depth is 0 because of recursion
        # using pseudocode from lecture and source 7 - implementation of ID3 using function of IG:
        labels = [row[y] for row in values]
        countlabels = {}
        for label in labels:
            if label in countlabels: countlabels[label] += 1
            else: countlabels[label] = 1
        countlabels = list(countlabels.items())
        countlabels.sort(key=lambda x: (-x[1], x[0]))
        #using source 6: sorts by max value (minus is for doing sort desc by value, then asc alphabetically)
        #sorting like this was smth new for me, source - geeks2geeks web
        if not values and countlabels: return countlabels[0][0] #from pseudocode: if D is empty
        #from pseudocode: if X is empty or D=D(y=v) (all labels are the same)
        if (not features or len(set(labels)) == 1) and countlabels: return countlabels[0][0]
        #if tree is "finished" by depth (idk how to translate this better):
        if self.depth is not None and depth >= self.depth and countlabels: return countlabels[0][0]

        #finding best option to continue algorithm, including if can not be found:
        #basically, creating "dictionary-styled" tuples
        best_options = [(self.IG(values, f, y), self.names[f], f) for f in features]
        best_options.sort(key=lambda x: (-x[0])) #sorting by IG desc
        if best_options: best_option = best_options[0][2]
        else: best_option = None
        if best_option is None and countlabels:
            return countlabels[0][0]

        result = {best_option: {}} #result tree should (obviously) begin with best option to begin, then another "leaves"
                                   #will be added to tree
        valuess = set(row[best_option] for row in values)
        new_features = [feature for feature in features]
        for value in valuess:
            subset = [row for row in values if row[best_option] == value]
            #source 9: "...recursively grow the tree for each partition until fully
            #grown or stopping criteria are met." - sources 2, 7: making function recursive
            result[best_option][value] = self.algorithm(subset, values, new_features, y, depth + 1)
        return result

    def fit(self, names, values):
    #from lab task: "The fit method, which obtains a dataset as an argument and performs model learning [1]"
    #basically it is just recursive function for "training" model with dataset
        self.names = names
        features=[]
        for i in range(len(names) - 1): features.append(i)
        self.tree = self.algorithm(values, values, features, (len(names) - 1)) #recursive implementation of performing[1]

    def fit_out(self, tree, prefix='', depth=1): #extra function to print branches from fit function
                                                 #(i am not smart enough to implement it in one function with fit)
        if type(tree) != dict: #checking if tree is actually leaf or subtree (sorry for game of words)
            print(prefix, tree)
            return
        x = list(tree.keys())[0] #first "key" value (retyping to list, dict is not comparable)
        feature_name = self.names[x]
        print("[BRANCHES]:")
        for value, subtree in tree[x].items():
            new_prefix = "{} {}:{}={}".format(prefix, depth, feature_name, value) #format of necessary output
            self.fit_out(subtree, new_prefix, depth + 1) #recursive calling of function

    def predict(self, values): #from lab task: "The predict method, which obtains a dataset as an argument
                               #and performs prediction of the class label based on a trained model"
        result = []
        for row in values:
            tree = self.tree
            while type(tree) == dict:
                #comparing to dictionary - seems to be strange, HOWEVER (!) this data type stores data in pairs
                #like (key, value) - so comparing with dict type is like validation, source - python documentation, geeks2geeks webpage
                node = list(tree.keys())[0] #first "key" value (retyping to list, dict is not comparable)
                riadok = tree[node] #set of "leaves" (вітки дерева) - basically, branches of decision tree
                tree = riadok.get(row[node]) #getting leave at specific row
                if tree is None:
                    leaves = [value for value in riadok.values() if type(tree)!=dict]
                    #from source 6: lambda expression, to sort set(best for sorting), we need most "popular" leaf
                    result.append(sorted(set(leaves), key=lambda x: (-leaves.count(x)))[0])
                    break
            else: result.append(tree)
        print("[PREDICTIONS]:", " ".join(result))
        return result

#after implementing ID3 class, it is necessary to implement another functions:

def accuracy(true_labels, predictions): #function of implementing accuracy
    x = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predictions[i]: x += 1
    accuracy = x / len(true_labels) #basically, just finding quantity (that's how we call it?), then accuracy formula
    #print (round(accuracy, 5)) - does NOT match with output format of autograder, sadly
    print("[ACCURACY]:", "{:.5f}".format(accuracy)) #just via format it works fine...

def confusion(true_labels, predictions): #function to create confusion matrix
    labels = []
    for label in true_labels + predictions: #from literature sources - 2 cycles are ineffective, union is not fine with lists...
        if label not in labels: labels.append(label) #adding unique labels from BOTH so-called "true labels" and predictions
    labels.sort() #sorting labels (with no sorting does not do nice job)
    spisok = {}
    for i in range(len(labels)): spisok[labels[i]] = i #creating "spisok" - list-dictionary of unique labels indexes
    matrix = [[0] * len(labels) for i in range(len(labels))] #creating matrix of necessary size
    for i in range(len(true_labels)):
        a = spisok[true_labels[i]]
        p = spisok[predictions[i]]
        matrix[a][p] += 1 #just filling matrix with counts increasing by 1 in cycle (source 12)

    print("[CONFUSION_MATRIX]:")
    for row in matrix:
        for x in row:
            print(x, end=' ')  #printing every element of matrix in necessary format
        print()
#part of parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("filename1")
parser.add_argument("filename2")
parser.add_argument("depth", nargs='?', type=int) #optional argument (isdigit, so type is int)
args = parser.parse_args()
if args.depth: depth = args.depth
else: depth = None

#finally - CALL THIS FUNCTIONS HERE RIGHT NOW!!! :)
names, values1 = readfile(args.filename1)
names1, values2 = readfile(args.filename2)
algorithm = ID3(depth)
algorithm.fit(names, values1)
algorithm.fit_out(algorithm.tree)
result=algorithm.predict(values2)
values3 = [row[-1] for row in values2]
accuracy(values3, result)
confusion(values3, result)
