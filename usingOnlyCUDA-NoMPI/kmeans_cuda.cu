#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

// CUDA kernel to assign points to the nearest cluster
__global__ void assignPointsToClusters(const double* pointsX, const double* pointsY, int* clusterAssignments,
                                        const double* centroidsX, const double* centroidsY, int numPoints, int numClusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    double minDist = INFINITY;
    int bestCluster = -1;

    for (int j = 0; j < numClusters; ++j) {
        double dist = (pointsX[idx] - centroidsX[j]) * (pointsX[idx] - centroidsX[j]) +
                      (pointsY[idx] - centroidsY[j]) * (pointsY[idx] - centroidsY[j]);
        if (dist < minDist) {
            minDist = dist;
            bestCluster = j;
        }
    }

    clusterAssignments[idx] = bestCluster;
}

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
    vector<double> x, y;
    readInputData(x, y);
    int numPoints = x.size();

    auto start = std::chrono::high_resolution_clock::now();
    int numClusters;
    cout << "Enter the number of clusters: ";
    cin >> numClusters;
    if (numClusters <= 0 || numClusters > numPoints) {
        cerr << "Invalid number of clusters." << endl;
        return EXIT_FAILURE;
    }

    // Allocate memory for points and centroids
    double *d_pointsX, *d_pointsY;
    int *d_clusterAssignments;

    cudaMalloc(&d_pointsX, numPoints * sizeof(double));
    cudaMalloc(&d_pointsY, numPoints * sizeof(double));
    cudaMalloc(&d_clusterAssignments, numPoints * sizeof(int));

    cudaMemcpy(d_pointsX, x.data(), numPoints * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointsY, y.data(), numPoints * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize centroids
    vector<double> centroidsX(numClusters), centroidsY(numClusters);
    for (int i = 0; i < numClusters; ++i) {
        centroidsX[i] = x[i];
        centroidsY[i] = y[i];
    }

    double *d_centroidsX, *d_centroidsY;
    cudaMalloc(&d_centroidsX, numClusters * sizeof(double));
    cudaMalloc(&d_centroidsY, numClusters * sizeof(double));

    cudaMemcpy(d_centroidsX, centroidsX.data(), numClusters * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroidsY, centroidsY.data(), numClusters * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    int maxIterations = 100;
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Assign points to clusters
        assignPointsToClusters<<<gridSize, blockSize>>>(d_pointsX, d_pointsY, d_clusterAssignments,
                                                        d_centroidsX, d_centroidsY, numPoints, numClusters);

        // Copy cluster assignments back to host
        vector<int> clusterAssignments(numPoints);
        cudaMemcpy(clusterAssignments.data(), d_clusterAssignments, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

        // Compute new centroids on host
        vector<double> clusterSumsX(numClusters, 0.0), clusterSumsY(numClusters, 0.0);
        vector<int> clusterCounts(numClusters, 0);

        for (int i = 0; i < numPoints; ++i) {
            int cluster = clusterAssignments[i];
            clusterSumsX[cluster] += x[i];
            clusterSumsY[cluster] += y[i];
            clusterCounts[cluster]++;
        }

        for (int i = 0; i < numClusters; ++i) {
            if (clusterCounts[i] > 0) {
                centroidsX[i] = clusterSumsX[i] / clusterCounts[i];
                centroidsY[i] = clusterSumsY[i] / clusterCounts[i];
            }
        }

        // Copy updated centroids to device
        cudaMemcpy(d_centroidsX, centroidsX.data(), numClusters * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_centroidsY, centroidsY.data(), numClusters * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Output final centroids
    cout << "Final Centroids after " << maxIterations << " iterations:" << endl;
    for (int i = 0; i < numClusters; ++i) {
        cout << "Cluster " << i << ": (" << centroidsX[i] << ", " << centroidsY[i] << ")" << endl;
    }

    // Free device memory
    cudaFree(d_pointsX);
    cudaFree(d_pointsY);
    cudaFree(d_clusterAssignments);
    cudaFree(d_centroidsX);
    cudaFree(d_centroidsY);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Serial Run (without MPI and CUDA) took " << duration.count() << " milliseconds." << std::endl;

    return 0;
}
