"""
file@ kmean.py
author@ Brian Q, Jian Shen
date@ Feb 17, 2019
"""
from sklearn import metrics
from sklearn import datasets
import numpy
from random import choice
from matplotlib import pyplot

#read mushroom data from local
def load_mushroom(file_name):
    file = open(file_name)
    rows = file.read().splitlines() #split the lines at line boundaries returns a list of lines
    file.close()
    dataset = []

    #for each data row, create a new list to store 23 features
    for i in range(1,len(rows)): # skip first row(name of features)
        col = rows[i].split(',') # create a list of strings after breaking the given string by ','
        item_features = [] #one list for each item
        # for each column
        for j in range(0, len(col)):
            val = float(ord(col[j])); #convert values to float, make sure types are not flexible
            item_features.append(val); #add feature value to item list
        dataset.append(item_features)

    dataset = numpy.array(dataset) #conversion from 2d list to 2d array
    return dataset;


#read iris data from local
def load_iris(file_name):
    file = open(file_name)
    rows = file.read().splitlines() # split the lines at line boundaries. returns a list of lines
    file.close()
    dataset = []

    #for each data row, create a new list to store four features
    for i in range(1,len(rows)): # skip first row(name of features)
        col = rows[i].split(',') # create a list of strings after breaking the given string by ','
        item_features = [] #one list for each item
        # for each column
        for j in range(1, len(col) - 1): # skip first column(id) and last column(name of specie)
            val = float(col[j]); #convert values to float, make sure type are not flexible
            item_features.append(val); #add feature value to item list
        dataset.append(item_features);


    dataset = numpy.array(dataset) # conversion from 2d list to 2d array

    return dataset;




#determine the number of clusters in a dataset using silhouette method
def number_of_clusters(dataset):
    max_silhouette_value = -1; #the minimum value for silhouette coefficient( range from -1 to 1, higher the value, better the cluster)
    best_k = 1;
    #for each k value, calculate silhouette value, choose the maximum silhouette and respective k
    for i in range(2, 11):# test cases where k from 2 to 10
        centroids, label = kmeans(dataset, i) #retrieve centroids location and label info
        temp_silhouette_value = metrics.silhouette_score(dataset, label) #calculate value here
        print("silhouette coefficient for k = ", i,  " is ", temp_silhouette_value)
        if temp_silhouette_value > max_silhouette_value: #update value if find a larger one
            best_k = i
            max_silhouette_value = temp_silhouette_value

    print ("best_k is : ", best_k)
    return best_k;




#k means clustering
def kmeans(dataset, k):

    # geneterate k random centroids within the range of dataset
    def __rand_centroids__(dataset, k):
        n = len(dataset[0])# retrive number of features
        centroids = numpy.zeros((k, n)) #initial n rows k columns

        #randomlize the value for each feature of centroids
        for i in range(n):
            min_value = numpy.min(dataset[:, i]) #find minimum value for each column
            max_value = numpy.max(dataset[:, i]) #find maximum value for each column
            centroids[:,i] = min_value + (max_value - min_value) * numpy.random.rand(k) #make sure random value is within range

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
                dist = numpy.linalg.norm(dataset[i] - centroids[j])

                #step 3:categorize each point to each nearest mean
                if dist < min_dist:
                    min_dist = dist
                    index = j
                    label[i] = j # set the nearest mean of data i is j

        #step 4:update mean, shift center
        for l in range(k):
            # print("dataset label ==" , l," is ", "\n", dataset[label == l], "\n\n")
            if len(dataset[label == l]) != 0: #dont update centroid if label l cluster is null,
                centroids[l] = numpy.mean(dataset[label == l])

        converged = True if (prev_centroids == centroids).all() else False  #check if centroids change
        # print("centroids for k = ", k, "\n", centroids, "\n\n\n")

    return centroids, label

##create 2d image to plot data
def PlotClusters(dataset, label, centroids):
    fig, (ax1,ax2) = pyplot.subplots(1, 2, figsize = (12,5))#use ax1 to plot original data, ax2 to indicate clustering
    ax1.scatter(dataset[:, 1],dataset[:, 2],c = 'c',marker = 'o')
    ax2.scatter(dataset[label == 0][:, 1], dataset[label == 0][:, 2], c = 'r') #first column and second column data as parameters,plot first cluster
    ax2.scatter(dataset[label == 1][:, 1], dataset[label == 1][:, 2], c = 'c')# plot second cluster
    ax2.scatter(centroids[:, 1], centroids[:, 2], c = 'b', s = 120, marker = 'o')# plot centroids
    pyplot.show()



#test case
print("--------------------Iris--------------------\n\n")
iris = load_iris("Iris.csv") #load iris data from local file
centroids, label = kmeans(iris, number_of_clusters(iris)) #retrieve attributes from dataset under best k condition
PlotClusters(iris, label, centroids); #plot data

#test case
print("------------------Mushroom------------------\n\n")
mushroom = load_mushroom("mushrooms.csv") #load mushroom data from local file
centroids, label = kmeans(mushroom, number_of_clusters(mushroom)) #retrieve attributes from dataset under best k condition
PlotClusters(mushroom, label, centroids); #plot data
