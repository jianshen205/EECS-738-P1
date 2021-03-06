from sklearn import datasets
import numpy

#read data from local file
def load_data():

    return

#determine the number of clusters in a dataset using silhouette method
def number_of_clusters(dataset):
    int best_k = INT_MIN;
    ##for each k value, calculate silhouette value, choose the minimum silhouette and respective k
    for i in range(1,10):
        centroids, label = kmeans(dataset, i)
        temp_silhouette_value = sklearn.metrics.silhouette_score(dataset, label, metric=’euclidean’, sample_size=None, random_state=None)
        if temp_silhouette_value > best_k
            best_k = temp_silhouette_value
    print "best_k is :"
    print (best_k)
    return best_k;


#k means clustering
def kmeans(dataset, k):
    # geneterate k random centroids within the range of dataset
    def __rand_centroids__(dataset, k):
        n = len(dataset[0])# retrive number of features
        centroids = numpy.zeros((k, n)) #initial n rows k columns
        #randomlize the value for each feature for each centroid
        for i in range(n):
            min_value = numpy.min(dataset[:,i])
            max_value = numpy.max(dataset[:,i])

            centroids[:,i] = min_value + (max_value - min_value)* numpy.random.rand(k)
        return centroids


    #step1:initial k points randomly
    centroids = __rand_centroids__(dataset, k)
    converged = False #indicate if cluster converged, does it have to be equal? or less than 0.00001?

    num_of_entries = dataset.shape[0] #number of data point
    label = numpy.zeros(num_of_entries) #use to store the nearest centroid for each entry data, set all to zero

    #step 5: while any of centroids changes, repeat
    while not converged:
        prev_centroids = numpy.copy(centroids)
        for i in range(num_of_entries):
            min_dist, index = numpy.Inf, -1 # set min distance to positive infinity, set index of k to -1
            for j in range(k):

                #step 2:calculate eculidean distance
                dist = numpy.linalg.norm(dataset[i]-centroids[j])

                #step 3:categorize each point to each nearest mean
                if dist < min_dist:
                    min_dist = dist
                    index = j
                    label[i] = j # the nearest mean of data i is j
        #step 4:update mean, shift center
        for l in range(k):
            centroids[l] = numpy.mean(dataset[label == l])
        #check if centroids change
        converged = True if (prev_centroids == centroids).all() else False
        print(centroids)

    return centroids, label

iris = datasets.load_iris()
dataset = iris.data
number_of_clusters(dataset)
"""
todo list:
algorithm to find value of k:3 options
iris data problem
read data from
choose another data
formulate ideas, document process and results
"""
