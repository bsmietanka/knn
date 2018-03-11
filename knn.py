import argparse
import csv
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import measure

class kNN:

    '''class for classifying objects using kNN algorithm'''

    def __init__(self, _k, _metric, _voting):
        '''constructor'''
        self.k = _k
        self.metric = _metric
        self.voting = _voting

    @measure.performance
    def classifyObject(self, trainingSet, testPoint):
        '''classify single element basing on training set and return computed class'''
        dist = { 'distance' : self.metric(trainingSet['features'] - testPoint), 'class' : trainingSet['class']}
        idx = np.argpartition(dist['distance'], self.k)
        neighbours = [(dist['distance'][i], dist['class'][i]) for i in idx[:self.k]]
        return self.voting(neighbours)

    @measure.performance
    def classify(self, trainingSet, testSet, dst):
        '''clasify all elements from test set basing on training set and return results in a list'''
        cls = []
        total = len(testSet)
        for count, point in enumerate(testSet, start = 1):
            cls.append(self.classifyObject(trainingSet, point))
            if count % 100 == 0:
                print("Progress: {} / {}".format(count, total), end='\r')
        print("\nClassification complete")
        return cls

    @measure.performance
    def euclideanDist(distMat):
        '''euclidean metric'''
        distMat = np.square(distMat)
        # could be done without square root for performance gain,
        # but it may influence weighted voting
        return np.power(np.sum(distMat, axis = 1), 2)

    @measure.performance
    def taxicabDist(distMat):
        '''taxicab metric'''
        return np.sum(np.absolute(distMat), axis = 1)

    @measure.performance
    def chebyshevDist(distMat):
        '''Chebyshev metric'''
        return np.amax(distMat, axis = 1)

    # calling this metric requires slight change in approach
    # as it requires additional argument
    @measure.performance
    def minkowskiDist(distMat, p):
        '''Minkowski metric'''
        distMat = np.power(distMat, p)
        return np.power(np.sum(distMat, axis = 1), float(1/p))


    @measure.performance
    def simpleVoting(neighbours):
        '''simple voting'''
        votes = {}
        for dist, category in neighbours:
            if votes.get(category) == None:
                votes[category] = 1
            else:
                votes[category] += 1
        sorted_votes = sorted(votes.items(), key=lambda tup: tup[1])
        if len(sorted_votes) > 1 and sorted_votes[-1] == sorted_votes[-2]:
            return simpleVoting(sorted(neighbours, key=lambda tup: tup[0])[1:])
        return sorted_votes[-1][0]
        # return max(votes.items(), key=lambda tup: tup[1])[0]

    @measure.performance
    def weightedVoting(neighbours):
        '''weighted variant of voting - weights are reciprocal of distance'''
        votes = {}
        for (dist, category) in neighbours:
            if votes.get(category) == None:
                votes[category] = 1/dist
            else:
                votes[category] += 1/dist
        sorted_votes = sorted(votes.items(), key=lambda tup: tup[1])
        if len(sorted_votes) > 1 and sorted_votes[-1] == sorted_votes[-2]:
            return simpleVoting(sorted(neighbours, key=lambda tup: tup[0])[1:])
        return sorted_votes[-1][0]
        # return max(votes.items(), key=lambda tup: tup[1])[0]

    # static dictionaries - map strings to function pointer
    metrics = {
        'euclidean' : euclideanDist,
        'taxicab' : taxicabDist,
        'chebyshev' : chebyshevDist#,
        #'minkowski' : minkowskiDist
        }

    votings = {
        'simple' : simpleVoting,
        'weighted' : weightedVoting
        }

    def calculateAccuracy(self, results, reference):
        '''calculate classification accuracy'''
        return np.sum(results == reference) / len(results)

    def __str__(self):
        '''print knn fields'''
        return str({'k' : self.k, 'metric' : self.metric.__name__, 'voting' : self.voting.__name__})


# NOTE: consider checking consistency of data sets
def readDataSetFromCsv(filename):
    """
    reads csv file row by row (skips header) and returns a dictionary of lists where
    'features' contains tuples of floats describing the objects (all elements in a row but the last one) and
    'class' contains objects classes (last element in row)
    """
    with open(filename, newline='') as csvfile:
        features = []
        classes = []
        csvreader = csv.reader(csvfile)
        next(csvreader, None) #skip header
        for row in csvreader:
            features.append([float(elem) for elem in row[:-1]])
            classes.append(int(row[-1]))
    return {'features' : np.array(features), 'class' : np.array(classes)}

def main():
    #create args parser and parse args
    parser = argparse.ArgumentParser(description='k Nearest Neighbours algorithm')
    parser.add_argument('trainingSetPath', metavar='training_set_path', help='path to a training set in .csv')
    parser.add_argument('testSetPath', metavar='test_set_path', help='path to a test set in .csv')
    parser.add_argument('-k', type=int, default=5, help='number of neighbours')
    parser.add_argument('-m', '--metric', default='euclidean', choices=kNN.metrics.keys(), help='metric used to calculate distance')
    parser.add_argument('-v', '--voting', default='simple', choices=kNN.votings.keys(), help='voting scheme used in the kNN algorithm')
    parser.add_argument('-d', '--dst', help='output destination')
    args = parser.parse_args()

    #parse csv files for training and test set
    trainingSet = readDataSetFromCsv(args.trainingSetPath)
    testSet = readDataSetFromCsv(args.testSetPath)

    #knn algorithm
    knn = kNN(args.k, kNN.metrics[args.metric], kNN.votings[args.voting])
    cls = np.array(knn.classify(trainingSet, testSet['features'], args.dst))

    acc = knn.calculateAccuracy(cls, testSet['class'])
    print ("Classification accuracy: {}".format(acc))

    #print performance information
    measure.dump()

    cmap_bold  = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])

    #transpose [samples][features] to get [features][samples]
    X = trainingSet['features'].T
    y = trainingSet['class']

    XX = testSet['features'].T
    yy = cls

    plt.scatter(X[0], X[1], s=2, c=y, cmap=cmap_bold)
    plt.scatter(XX[0], XX[1], s=2, c=yy, cmap=cmap_light)
    plt.title("kNN classification (ACC={})".format(acc))
    plt.ylabel("Y")
    plt.xlabel("X")
    fig = plt.gcf()
    fig.canvas.set_window_title('kNN')
    plt.show()

    if args.dst is not None:
        with open(args.dst, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['x', 'y', 'cls'])
            for features, c in zip(testSet['features'], cls):
                csvwriter.writerow([features[0], features[1], c])



#data three gauss: data.three_gauss.train.10000.csv data.three_gauss.test.10000.csv
#data simple: data.simple.train.10000.csv data.simple.test.10000.csv
if __name__ == "__main__":
    main()
