#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <chrono>

using namespace std;

// Function to calculate the squared distance between two points
double calculateDistance(double x1, double y1, double x2, double y2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

// Function to read input data from `input10K.txt`
void readInputData(vector<double>& x, vector<double>& y) {
    ifstream inputFile("input10K.txt");
    if (!inputFile.is_open()) {
        cerr << "Error opening input10K.txt file!" << endl;
        exit(EXIT_FAILURE);
    }

    double xi, yi;
    while (inputFile >> xi >> yi) {
        x.push_back(xi);
        y.push_back(yi);
    }
    inputFile.close();
}

int main() {
    // Input data variables
    int numPoints = 0;
    int numClusters = 0;
    vector<double> x, y; // Vectors to store all points

    // Read input data
    readInputData(x, y);
    numPoints = x.size();

    auto start = std::chrono::high_resolution_clock::now();
    // Get the number of clusters from the user
    cout << "Enter the number of clusters: ";
    cin >> numClusters;

    if (numClusters <= 0 || numClusters > numPoints) {
        cerr << "Invalid number of clusters." << endl;
        return EXIT_FAILURE;
    }

    // Initialize centroids
    vector<double> centroidsX(numClusters);
    vector<double> centroidsY(numClusters);
    for (int i = 0; i < numClusters; ++i) {
        centroidsX[i] = x[i];
        centroidsY[i] = y[i];
    }

    // Run K-Means for a fixed number of iterations
    int maxIterations = 100;
    for (int iter = 0; iter < maxIterations; ++iter) {
        vector<int> clusterCounts(numClusters, 0);
        vector<double> clusterSumsX(numClusters, 0.0);
        vector<double> clusterSumsY(numClusters, 0.0);

        // Assign points to the nearest centroid
        for (int i = 0; i < numPoints; ++i) {
            double minDist = calculateDistance(x[i], y[i], centroidsX[0], centroidsY[0]);
            int bestCluster = 0;

            for (int j = 1; j < numClusters; ++j) {
                double dist = calculateDistance(x[i], y[i], centroidsX[j], centroidsY[j]);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }

            clusterSumsX[bestCluster] += x[i];
            clusterSumsY[bestCluster] += y[i];
            clusterCounts[bestCluster]++;
        }

        // Update centroids
        for (int i = 0; i < numClusters; ++i) {
            if (clusterCounts[i] > 0) {
                centroidsX[i] = clusterSumsX[i] / clusterCounts[i];
                centroidsY[i] = clusterSumsY[i] / clusterCounts[i];
            }
        }
    }

    // Output final centroids
    cout << "Final Centroids after " << maxIterations << " iterations:" << endl;
    for (int i = 0; i < numClusters; ++i) {
        cout << "Cluster " << i << ": (" << centroidsX[i] << ", " << centroidsY[i] << ")" << endl;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Serial Run (without MPI and CUDA) took " << duration.count() << " milliseconds." << std::endl;
    
    return 0;
}
