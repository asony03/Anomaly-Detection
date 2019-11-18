# Implementation for paper 4: NetSimile: A Scalable Approach to Size-Independent Network Similarity

import networkx as nx
import numpy as np
import os
import sys
import scipy
from scipy.stats import kurtosis, skew
import scipy.spatial.distance
from os.path import join
from pylab import *

def netSimile(listOfGraphs):
    # Generate feature matrix for each graph    
    featureMatrices = getFeatures(listOfGraphs)
    # Convert feature matrix to signature vector
    signatureVectors = aggregator(featureMatrices)
    # Calculate distance between signature vectors
    distances = compare(signatureVectors)
    # Calculate the upper threshold which will be used to determine the anomalies
    threshold = calculateUpperThreshold(distances)
    # Find anomalous graphs on the basis of threshold
    anomalies = findAnomalies(distances, threshold)
    # Generate the time series text file
    generateTimeSeriesFile(distances)
    # Plot the time series indicating the threshold
    plotTimeSeries(distances, threshold, anomalies)

def getFeatures(listOfGraphs):
    # Input: A list of graphs
    # Output: A list of feature matrix for each of the graphs. Seven features are calculated.
    featureMatrices = []
    for graph in listOfGraphs:
        featureMatrix = []
        for node in graph.nodes():
            # Degree of the node
            deg = graph.degree(node)

            # Clustering coefficient of the node
            clusteringCoeff = nx.clustering(graph, node)

            twoHopNeighbors = 0
            clusteringCoeffNeighbors = 0
            for neighbor in graph.neighbors(node):
                twoHopNeighbors = twoHopNeighbors + graph.degree(neighbor)
                clusteringCoeffNeighbors = clusteringCoeffNeighbors + nx.clustering(graph, neighbor)

            # Average number of node's two-hop away neighbors
            avgTwoHopNeighbors = float(twoHopNeighbors) / float(deg)

            # Average Clustering coefficient of the neighbors of the node
            avgClusteringCoeffNeighbors = float(clusteringCoeffNeighbors) / float(deg)
            
            # Number of edges in node's egonet
            egonet = nx.ego_graph(graph, node)
            edgesInEgonet = len(egonet.edges())

            
            outEdgesEgonet = 0
            neighborsOfEgonet = 0

            edges = set()
            neighbors = set()

            for vertex in egonet:
                edges = edges.union(graph.edges(vertex))
                neighbors = neighbors.union(graph.neighbors(vertex))

            edegs = edges - set(egonet.edges())
            neighbors = neighbors - set(egonet.nodes())

            # Number of outgoing edges from node's egonet
            outEdgesEgonet = len(list(edges))

            # Number of neighbors of egonet
            neighborsOfEgonet = len(list(neighbors))

            featureMatrix.append([deg, clusteringCoeff, avgTwoHopNeighbors, avgClusteringCoeffNeighbors, edgesInEgonet, outEdgesEgonet, neighborsOfEgonet])
        featureMatrices.append(featureMatrix)
    return featureMatrices

def aggregator(featureMatrices):
    # Input: A list of feature matrices
    # Output: A list of signature vectors, one for every feature matrix.
    signatureVectors = list()
    # Aggregating the values over all the nodes, for each feature.
    for featureMatrix in featureMatrices:
        signatureVector = []
        for i in range(7):
            featureColumn = [node[i] for node in featureMatrix]
            aggregatedFeature = [np.median(featureColumn), np.mean(featureColumn), np.std(featureColumn),
                          skew(featureColumn), kurtosis(featureColumn, fisher=False)]
            signatureVector = signatureVector + aggregatedFeature
        signatureVectors.append(signatureVector)
    return (signatureVectors)

def compare(signatureVectors):
    # Calculate canberra distance between i and i+1 graphs 
    n = len(signatureVectors)
    distance = [scipy.spatial.distance.canberra(signatureVectors[i], signatureVectors[i+1]) for i in range(0, n-1)]
    return distance

def calculateUpperThreshold(distances):
    # Input: A list of distances
    # Output: The upper threshold value for detecting anomalies
    diff = 0
    for i in range(2, len(distances)):
        diff = diff + abs(distances[i] - distances[i - 1])
    avgDiff = diff / (len(distances) - 1)
    median = np.median(distances)
    threshold = median + 3 * avgDiff
    return threshold


def findAnomalies(distances, threshold):
    # Input: Canberra distances between graphs and the upper threshold for finding anomalies
    # Output: The indexes of anomalous graphs
    anomalousGraphs = []
    n = len(distances)
    for i in range(0, n-1):
        # Graph is anomalous if there are two consecutive anomalous time points in the output
        if distances[i] > threshold and distances[i + 1] > threshold:
            anomalousGraphs.append(i+1)
    return anomalousGraphs


def generateTimeSeriesFile(distances):
    #Input: List of canberra distances between graphs
    #Output: Creates a text file with similarity values of the time series
    timeSeriesFilePath = os.getcwd() + '/output/' + os.path.basename(sys.argv[1]) + "_time_series.txt"
    if not os.path.exists(os.path.dirname(timeSeriesFilePath)):
        os.makedirs(os.path.dirname(timeSeriesFilePath))
    f = open(timeSeriesFilePath, 'w+')
    for distance in distances:
        f.write(str(distances) + '\n')
    f.close()


def plotTimeSeries(dists, threshold, anomalies):
    #Input: List of canberra distances between graphs and the upper threshold value for detecting anomalies
    #Output: Generates the time series plot for detecting anomalies indicating the threshold
    figure(figsize=(12, 6))
    plt.plot(dists, "-o")
    axhline(y=threshold, c='red', lw=2)
    plt.title("Anomaly Detection for " + os.path.basename(sys.argv[1]) + " dataset")
    plt.xlabel("Time Series")
    plt.ylabel("Canberra Distance")
    plotFilePath = os.getcwd() + '/output/' + os.path.basename(sys.argv[1]) + "_time_series.png"
    if not os.path.exists(os.path.dirname(plotFilePath)):
        os.makedirs(os.path.dirname(plotFilePath))
    savefig(plotFilePath, bbox_inches='tight')


if __name__ == "__main__":
    # Read path of the folder containing the graphs.
    graphFolderPath = sys.argv[1]
    # Read all files in the directory and create a networkx graph for each file
    listOfGraphs = list()
    for file in os.listdir(graphFolderPath):
        file = graphFolderPath + '/' + file
        f = open(file, 'r')
        # Skipping the first row as it contains the number of nodes and edges
        next(f)
        g = nx.Graph()
        for line in f:
            line = line.split()
            g.add_edge(int(line[0]), int(line[1]))
        # Append the graph into the list
        listOfGraphs.append(g)
    # Algoritm 1: NetSimile
    netSimile(listOfGraphs)
