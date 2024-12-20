#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define MAX_NODES 10
#define INF 1e9

using namespace std;

int adjacencyMatrix[MAX_NODES][MAX_NODES]; // Graph representation
float weights[MAX_NODES][MAX_NODES];      // Weights of edges between nodes
int clusterAssignments[MAX_NODES];        // Final cluster assignment of each node

void initializeGraph() {
    // Sample graph with weights representing the strength of the relationship
    srand(time(0));
    for (int i = 0; i < MAX_NODES; ++i) {
        for (int j = 0; j < MAX_NODES; ++j) {
            if (i != j && rand() % 2) { // Randomly decide whether there's an edge
                adjacencyMatrix[i][j] = 1;
                weights[i][j] = static_cast<float>(rand() % 100) / 100.0f; // Weight in range [0, 1]
            } else {
                adjacencyMatrix[i][j] = 0;
                weights[i][j] = 0.0f;
            }
        }
    }
}

float computeStrength(int node1, int node2) {
    return weights[node1][node2];
}

void printGraph() {
    cout << "Adjacency Matrix with Weights:\n";
    for (int i = 0; i < MAX_NODES; ++i) {
        for (int j = 0; j < MAX_NODES; ++j) {
            cout << weights[i][j] << " ";
        }
        cout << endl;
    }
}

void assignClusters(int start, int end, int k, float centroids[][MAX_NODES], int rank) {
    for (int i = start; i < end; ++i) {
        float minDist = INF;
        int bestCluster = 0;
        for (int c = 0; c < k; ++c) {
            float dist = 0.0;
            for (int j = 0; j < MAX_NODES; ++j) {
                dist += fabs(weights[i][j] - centroids[c][j]);
            }
            if (dist < minDist) {
                minDist = dist;
                bestCluster = c;
            }
        }
        clusterAssignments[i] = bestCluster;
        if (rank == 0) {
            cout << "Node " << i << " assigned to cluster " << bestCluster << endl;
        }
    }
}

void updateCentroids(int k, float centroids[][MAX_NODES], int rank) {
    float tempCentroids[k][MAX_NODES] = {0};
    int clusterSizes[k] = {0};

    for (int i = 0; i < MAX_NODES; ++i) {
        int cluster = clusterAssignments[i];
        for (int j = 0; j < MAX_NODES; ++j) {
            tempCentroids[cluster][j] += weights[i][j];
        }
        clusterSizes[cluster]++;
    }

    for (int c = 0; c < k; ++c) {
        for (int j = 0; j < MAX_NODES; ++j) {
            if (clusterSizes[c] > 0) {
                centroids[c][j] = tempCentroids[c][j] / clusterSizes[c];
            }
        }
    }

    if (rank == 0) {
        cout << "Updated centroids:\n";
        for (int c = 0; c < k; ++c) {
            cout << "Cluster " << c << ": ";
            for (int j = 0; j < MAX_NODES; ++j) {
                cout << centroids[c][j] << " ";
            }
            cout << endl;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int k = 3; // Number of clusters

    if (rank == 0) {
        initializeGraph();
        printGraph();
    }

    MPI_Bcast(adjacencyMatrix, MAX_NODES * MAX_NODES, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights, MAX_NODES * MAX_NODES, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float centroids[k][MAX_NODES];
    if (rank == 0) {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < MAX_NODES; ++j) {
                centroids[i][j] = weights[i][j];
            }
        }
    }

    MPI_Bcast(centroids, k * MAX_NODES, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int nodesPerProc = MAX_NODES / numProcs;
    int start = rank * nodesPerProc;
    int end = (rank == numProcs - 1) ? MAX_NODES : start + nodesPerProc;

    for (int iter = 0; iter < 10; ++iter) {
        assignClusters(start, end, k, centroids, rank);

        MPI_Allgather(MPI_IN_PLACE, nodesPerProc, MPI_INT, clusterAssignments, nodesPerProc, MPI_INT, MPI_COMM_WORLD);

        if (rank == 0) {
            updateCentroids(k, centroids, rank);
        }

        MPI_Bcast(centroids, k * MAX_NODES, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        cout << "Final cluster assignments:\n";
        for (int i = 0; i < MAX_NODES; ++i) {
            cout << "Node " << i << " -> Cluster " << clusterAssignments[i] << endl;
        }
    }

    MPI_Finalize();
    return 0;
}

