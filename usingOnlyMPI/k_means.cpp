#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <chrono>

using namespace std;

// Function to calculate the squared distance between two points
double calculateDistance(double x1, double y1, double x2, double y2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

// Function to read input data from `input.txt`
void readInputData(double* x, double* y, int& numPoints) {
    ifstream inputFile("input10K.txt");
    if (!inputFile.is_open()) {
        cerr << "Error opening input.txt file!" << endl;
        exit(EXIT_FAILURE);
    }

    numPoints = 0;
    while (inputFile >> x[numPoints] >> y[numPoints]) {
        numPoints++;
    }
    inputFile.close();
}

int main(int argc, char** argv) {
    int rank, numProcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Input data variables
    int numPoints = 0;
    int numClusters = 0;
    double allX[1000], allY[1000]; // Array to store all points
    
    auto start = std::chrono::high_resolution_clock::now();
    // Rank 0 reads the input and determines the number of clusters
    if (rank == 0) {
        cout << "Enter the number of clusters: ";
        cin >> numClusters;
        readInputData(allX, allY, numPoints);
    }

    // Broadcast the number of clusters and total points
    MPI_Bcast(&numClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the data among processes
    int pointsPerProc = numPoints / numProcs;
    int extraPoints = numPoints % numProcs;
    int* recvCounts = new int[numProcs];
    int* displacements = new int[numProcs];
    for (int i = 0; i < numProcs; ++i) {
        recvCounts[i] = pointsPerProc + (i < extraPoints ? 1 : 0);
        displacements[i] = (i == 0) ? 0 : (displacements[i - 1] + recvCounts[i - 1]);
    }

    double* localX = new double[recvCounts[rank]];
    double* localY = new double[recvCounts[rank]];

    MPI_Scatterv(allX, recvCounts, displacements, MPI_DOUBLE, localX, recvCounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(allY, recvCounts, displacements, MPI_DOUBLE, localY, recvCounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Initialize centroids
    double* centroidsX = new double[numClusters];
    double* centroidsY = new double[numClusters];
    int* clusterCounts = new int[numClusters];
    double* clusterSumsX = new double[numClusters];
    double* clusterSumsY = new double[numClusters];

    if (rank == 0) {
        for (int i = 0; i < numClusters; ++i) {
            centroidsX[i] = allX[i];
            centroidsY[i] = allY[i];
        }
    }

    // Broadcast initial centroids to all processes
    MPI_Bcast(centroidsX, numClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(centroidsY, numClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Run K-Means for a fixed number of iterations
    int maxIterations = 100;
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Reset local cluster sums and counts
        int* localClusterCounts = new int[numClusters]();
        double* localClusterSumsX = new double[numClusters]();
        double* localClusterSumsY = new double[numClusters]();

        // Assign points to the nearest centroid
        for (int i = 0; i < recvCounts[rank]; ++i) {
            double minDist = (localX[i] - centroidsX[0]) * (localX[i] - centroidsX[0]) + 
                 (localY[i] - centroidsY[0]) * (localY[i] - centroidsY[0]);
            int bestCluster = 0;

            for (int j = 1; j < numClusters; ++j) {
                double dist = calculateDistance(localX[i], localY[i], centroidsX[j], centroidsY[j]);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }

            localClusterSumsX[bestCluster] += localX[i];
            localClusterSumsY[bestCluster] += localY[i];
            localClusterCounts[bestCluster]++;
        }

        // Reduce all cluster sums and counts to rank 0
        MPI_Reduce(localClusterSumsX, clusterSumsX, numClusters, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(localClusterSumsY, clusterSumsY, numClusters, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(localClusterCounts, clusterCounts, numClusters, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // Update centroids on rank 0
        if (rank == 0) {
            for (int i = 0; i < numClusters; ++i) {
                if (clusterCounts[i] > 0) {
                    centroidsX[i] = clusterSumsX[i] / clusterCounts[i];
                    centroidsY[i] = clusterSumsY[i] / clusterCounts[i];
                }
            }
        }

        // Broadcast updated centroids to all processes
        MPI_Bcast(centroidsX, numClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(centroidsY, numClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Clean up local cluster data
        delete[] localClusterCounts;
        delete[] localClusterSumsX;
        delete[] localClusterSumsY;
    }

    // Output final centroids on rank 0
    if (rank == 0) {
        cout << "Final Centroids after " << maxIterations << " iterations:" << endl;
        for (int i = 0; i < numClusters; ++i) {
            cout << "Cluster " << i << ": (" << centroidsX[i] << ", " << centroidsY[i] << ")" << endl;
        }
    }

    // Clean up dynamically allocated memory
    delete[] localX;
    delete[] localY;
    delete[] centroidsX;
    delete[] centroidsY;
    delete[] clusterCounts;
    delete[] clusterSumsX;
    delete[] clusterSumsY;
    delete[] recvCounts;
    delete[] displacements;

    MPI_Finalize();
    
    if (rank == 0) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Running with MPI took " << duration.count() << " milliseconds." << std::endl;
    }
    
    return 0;
}
