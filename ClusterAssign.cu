#include <cuda_runtime.h>
#include <cstdio>
#include "ClusterAssign.h"

__global__ void calculateDistancesKernel(
    const double* pointsX, const double* pointsY, int numPoints,
    const double* centroidsX, const double* centroidsY, int numClusters,
    int* clusterAssignments) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints) {
        double x = pointsX[idx];
        double y = pointsY[idx];
        double minDist = (x - centroidsX[0]) * (x - centroidsX[0]) +
                         (y - centroidsY[0]) * (y - centroidsY[0]);
        int bestCluster = 0;

        for (int j = 1; j < numClusters; ++j) {
            double dist = (x - centroidsX[j]) * (x - centroidsX[j]) +
                          (y - centroidsY[j]) * (y - centroidsY[j]);
            if (dist < minDist) {
                minDist = dist;
                bestCluster = j;
            }
        }
        clusterAssignments[idx] = bestCluster;
        printf("Thread %d: Point (%f, %f), assigned to cluster %d\n", idx, pointsX[idx], pointsY[idx], bestCluster);
        
    }
    
    if (idx == 0) {
    printf("Kernel running for %d points and %d clusters\n", numPoints, numClusters);
    }

}

extern "C" void calculateDistancesCUDA(
    const double* hostPointsX, const double* hostPointsY, int numPoints,
    const double* hostCentroidsX, const double* hostCentroidsY, int numClusters,
    int* hostClusterAssignments) {
    
    // Device memory allocation
    double *d_pointsX, *d_pointsY, *d_centroidsX, *d_centroidsY;
    int* d_clusterAssignments;

    cudaError_t err;

    err = cudaMalloc(&d_pointsX, numPoints * sizeof(double));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for pointsX: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc(&d_pointsY, numPoints * sizeof(double));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for pointsY: %s\n", cudaGetErrorString(err));
        cudaFree(d_pointsX);
        return;
    }

    err = cudaMalloc(&d_centroidsX, numClusters * sizeof(double));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for centroidsX: %s\n", cudaGetErrorString(err));
        cudaFree(d_pointsX);
        cudaFree(d_pointsY);
        return;
    }

    err = cudaMalloc(&d_centroidsY, numClusters * sizeof(double));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for centroidsY: %s\n", cudaGetErrorString(err));
        cudaFree(d_pointsX);
        cudaFree(d_pointsY);
        cudaFree(d_centroidsX);
        return;
    }

    err = cudaMalloc(&d_clusterAssignments, numPoints * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for clusterAssignments: %s\n", cudaGetErrorString(err));
        cudaFree(d_pointsX);
        cudaFree(d_pointsY);
        cudaFree(d_centroidsX);
        cudaFree(d_centroidsY);
        return;
    }

    // Copy data to device
    cudaMemcpy(d_pointsX, hostPointsX, numPoints * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointsY, hostPointsY, numPoints * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroidsX, hostCentroidsX, numClusters * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroidsY, hostCentroidsY, numClusters * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel launch
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;
    //printf("Launching kernel with %d blocks and %d threads per block\n", numBlocks, blockSize);

    calculateDistancesKernel<<<numBlocks, blockSize>>>(
        d_pointsX, d_pointsY, numPoints, d_centroidsX, d_centroidsY, numClusters, d_clusterAssignments);

    // Check for errors during kernel launch
cudaError_t err2 = cudaGetLastError();
if (err2 != cudaSuccess) {
    printf("Error launching kernel: %s\n", cudaGetErrorString(err2));
}
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA kernel execution failed: %s\n", cudaGetErrorString(err));
    } else {
    printf("Kernel executed successfully.\n");
}

    // Copy results back to host
    cudaMemcpy(hostClusterAssignments, d_clusterAssignments, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pointsX);
    cudaFree(d_pointsY);
    cudaFree(d_centroidsX);
    cudaFree(d_centroidsY);
    cudaFree(d_clusterAssignments);
}

