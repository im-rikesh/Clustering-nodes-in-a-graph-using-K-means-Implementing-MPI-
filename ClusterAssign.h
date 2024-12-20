#ifndef CLUSTER_ASSIGN_H
#define CLUSTER_ASSIGN_H

#ifdef __cplusplus
extern "C" {
#endif

// Declare the CUDA function
void calculateDistancesCUDA(
    const double* pointsX,        // X coordinates of the points
    const double* pointsY,        // Y coordinates of the points
    int numPoints,                // Number of points
    const double* centroidsX,     // X coordinates of the centroids
    const double* centroidsY,     // Y coordinates of the centroids
    int numClusters,              // Number of clusters
    int* clusterAssignments       // Output cluster assignments
);

#ifdef __cplusplus
}
#endif

#endif // CLUSTER_ASSIGN_H

