/*
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <assert.h>
#include <iostream>
// Needed for memalign
#include <malloc.h>
#include <mpi.h>

using namespace std;

void printMat(const char mesg[], double *E, int m, int n);
void printMatNaiveHelper(const char mesg[], double *E, int m, int n);

bool isRankMaster(int rank) {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process
  if (world_rank == rank)
    return true;
  return false;
}

//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init(double *E, double *E_prev, double *R, int m, int n) {
  if (isRankMaster(0)) {
    // MPI_Init(NULL, NULL);

    int i;

    for (i = 0; i < (m + 2) * (n + 2); i++) {
      R[i] = 0;
      E_prev[i] = 0;
    }

    for (i = (n + 2); i < (m + 1) * (n + 2); i++) {
      int colIndex =
          i %
          (n + 2); // gives the base index (first row's) of the current index

      // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
      if (colIndex == 0 || colIndex == (n + 1) || colIndex < ((n + 1) / 2 + 1))
        continue;

      E_prev[i] = 1.0;
    }

    for (i = 0; i < (m + 2) * (n + 2); i++) {
      int rowIndex =
          i /
          (n + 2); // gives the current row number in 2D array representation
      int colIndex =
          i %
          (n + 2); // gives the base index (first row's) of the current index

      // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
      if (colIndex == 0 || colIndex == (n + 1) || rowIndex < ((m + 1) / 2 + 1))
        continue;

      R[i] = 1.0;
    }
    // We only print the meshes if they are small enough
#if 1
    // printMatNaiveHelper("E_prev_initial", E_prev, m + 2, n + 2);
    // printMat("R",R,m,n);
#endif
  }
}

double *alloc1D(int m, int n) {
  if (isRankMaster(0)) {
    int nx = n, ny = m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E = (double *)memalign(16, sizeof(double) * nx * ny));
    return (E);
  }
  return NULL;
}

void printMat(const char mesg[], double *E, int m, int n) {
  int i;
#if 0
    if (m>8)
      return;
#else
  if (m > 34)
    return;
#endif
  printf("%s\n", mesg);
  for (i = 0; i < (m + 2) * (n + 2); i++) {
    int rowIndex = i / (n + 2);
    int colIndex = i % (n + 2);
    if ((colIndex > 0) && (colIndex < n + 1))
      if ((rowIndex > 0) && (rowIndex < m + 1))
        printf("%6.3f ", E[i]);
    if (colIndex == n + 1)
      printf("\n");
  }
}

void printMatNaiveHelper(const char mesg[], double *E, int m, int n) {
  int i;

  printf("\n%s\n", mesg);
  for (i = 0; i < m * n; i++) {
    int colIndex = i % n;
    printf("%6.3f ", E[i]);
    if (colIndex == n - 1)
      printf("\n");
  }
}
