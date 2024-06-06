/*
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 *
 * Modified and  restructured by Scott B. Baden, UCSD
 *
 */

#include "Plotting.h"
#include "apf.h"
#include "cblock.h"
#include "time.h"
#include <assert.h>
#include <emmintrin.h>
#include <iomanip>
#include <iostream>
#include <malloc.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string>
using namespace std;

#define TOP 0
#define RIGHT 1
#define BOTTOM 2
#define LEFT 3

int world_rank;
void repNorms(double l2norm, double mx, double dt, int m, int n, int niter,
              int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
void printMatNaive(const char mesg[], double *E, int m, int n);
void printMatRank(const char mesg[], int rank, double *E, int m, int n);
double *alloc1DAll(int size);
void padPhysicalBoundaryCells(double *E, int rankX, int rankY, int m, int n);
void buildSendGhostBuffer(double *E, double *buf, int dir, int m, int n);
void fillGhostCells(double *E, double *buf, int dir, int m, int n, int valid);
void exchangeGhostCells(double *E, double *sendBuf, double *recvBuf, int bufLen,
                        int rankX, int rankY, int m, int n,
                        MPI_Request *recvReq);
void calcComputeSpace(int rankX, int rankY, int m, int n, int &startIdx,
                      int &endIdx, int &strideComp);

void reshapePackedArray(double *E_prevPacked, double *E_prev,
                        int *sourceOffsets, int world_size, int smallStrideX,
                        int smallStrideY, int bigStrideX, int bigStrideY,
                        int numSmallRanksX, int numSmallRanksY, int n);

void solveAlievPanfilov(double *E_rank, double *E_prev_rank, double *R_rank,
                        int innerBlockRowStartIndex, int innerBlockRowEndIndex,
                        double alpha, double dt, int stride_rankX,
                        int stride_rankY, int strideComp);

void solveAlievPanfilovTimeStep(double *E_rank, double *E_prev_rank,
                                double *R_rank, int i, double alpha, double dt,
                                int stride_rankX, int stride_rankY);

void solveAlievPanfilovEdge(double *E_rank, double *E_prev_rank, double *R_rank,
                            double *ghostCellRecvBuf, int ghostBufLen,
                            double alpha, double dt, int stride_rankY,
                            int stride_rankX, int innerBlockRowStartIndex,
                            int innerBlockRowEndIndex, int strideComp,
                            int idxCompute, int valid);
extern control_block cb;

// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq) {
  double l2norm = sumSq / (double)((cb.m) * (cb.n));
  l2norm = sqrt(l2norm);
  return l2norm;
}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt,
           Plotter *plotter, double &L2, double &Linf) {

  // Simulated time is different from the integer timestep number
  double t = 0.0;
  double *E = *_E, *E_prev = *_E_prev;
  double mx, sumSq;
  int niter;
  int m = cb.m, n = cb.n;
  int innerBlockRowStartIndex;
  int innerBlockRowEndIndex;
  int world_size;
  extern int world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process
  int rankX = world_rank % cb.px;
  int rankY = world_rank / cb.px;
  int smallStrideX = (n + 2) / cb.px;
  int smallStrideY = (m + 2) / cb.py;
  int bigStrideX = smallStrideX + 1;
  int bigStrideY = smallStrideY + 1;
  int numBigRanksX = (n + 2) % cb.px;
  int numBigRanksY = (m + 2) % cb.py;
  int numSmallRanksX = cb.px - numBigRanksX;
  int numSmallRanksY = cb.py - numBigRanksY;
  int stride_rankX =
      (world_rank % cb.px < numSmallRanksX) ? smallStrideX : bigStrideX;
  int stride_rankY =
      (world_rank / cb.px < numSmallRanksY) ? smallStrideY : bigStrideY;
  int strideComp;
  int i, j;
  int *scatterCounts = new int[world_size];
  int *sourceOffsets = new int[world_size];
  int *packedOffsets = new int[world_size];
  double *E_prevPacked = NULL;
  double *RPacked = NULL;
  int ghostBufLen = MAX(stride_rankX, stride_rankY);
  double *ghostCellSendBuf = new double[4 * ghostBufLen];
  double *ghostCellRecvBuf = new double[4 * ghostBufLen];
  // register double tempE = 0;
  MPI_Request recvReq[4];

  if (world_rank == 0) {
    int *sourceOffsetsX = new int[cb.px];
    int *sourceOffsetsY = new int[cb.py];
    E_prevPacked = (double *)malloc(sizeof(double) * world_size *
                                    (bigStrideX * bigStrideY + bigStrideY * 2));
    RPacked = (double *)malloc(sizeof(double) * world_size *
                               (bigStrideX * bigStrideY + bigStrideY * 2));
    sourceOffsetsX[0] = 0;
    sourceOffsetsY[0] = 0;
    for (i = 1; i < cb.px; i++) {
      sourceOffsetsX[i] = sourceOffsetsX[i - 1] +
                          ((i <= numSmallRanksX) ? smallStrideX : bigStrideX);
    }
    for (i = 1; i < cb.py; i++) {
      sourceOffsetsY[i] = sourceOffsetsY[i - 1] +
                          ((i <= numSmallRanksY) ? smallStrideY : bigStrideY);
    }

    for (i = 0; i < cb.py; i++) {
      for (j = 0; j < cb.px; j++) {
        sourceOffsets[j + i * cb.px] =
            sourceOffsetsX[j] + sourceOffsetsY[i] * (n + 2);
        scatterCounts[j + i * cb.px] =
            (((j < numSmallRanksX) ? smallStrideX : bigStrideX) + 2) *
            ((i < numSmallRanksY) ? smallStrideY : bigStrideY);
      }
    }

    free(sourceOffsetsX);
    free(sourceOffsetsY);

    for (int rankIter = 0; rankIter < world_size; rankIter++) {
      int strideX =
          (rankIter % cb.px < numSmallRanksX) ? smallStrideX : bigStrideX;
      int strideY =
          (rankIter / cb.px < numSmallRanksY) ? smallStrideY : bigStrideY;
      int offset = sourceOffsets[rankIter];
      for (i = 0; i < strideY; i++) {
        for (j = 1; j <= strideX; j++) {
          E_prevPacked[rankIter * (bigStrideX * bigStrideY + 2 * bigStrideY) +
                       i * (strideX + 2) + j] =
              E_prev[offset + i * (n + 2) + (j - 1)];
          RPacked[rankIter * (bigStrideX * bigStrideY + 2 * bigStrideY) +
                  i * (strideX + 2) + j] = R[offset + i * (n + 2) + (j - 1)];
        }

        E_prevPacked[rankIter * (bigStrideX * bigStrideY + 2 * bigStrideY) +
                     i * (strideX + 2)] = 0;
        E_prevPacked[rankIter * (bigStrideX * bigStrideY + 2 * bigStrideY) +
                     i * (strideX + 2) + strideX + 1] = 0;
        RPacked[rankIter * (bigStrideX * bigStrideY + 2 * bigStrideY) +
                i * (strideX + 2)] = 0;
        RPacked[rankIter * (bigStrideX * bigStrideY + 2 * bigStrideY) +
                i * (strideX + 2) + strideX + 1] = 0;
      }
      packedOffsets[rankIter] =
          rankIter * (bigStrideX * bigStrideY + 2 * bigStrideY);
    }
  }

  double *E_rank = new double[(stride_rankY + 2) * (stride_rankX + 2)];
  double *E_prev_rank = new double[(stride_rankY + 2) * (stride_rankX + 2)];
  double *R_rank = new double[(stride_rankY + 2) * (stride_rankX + 2)];

  MPI_Scatterv(E_prevPacked, scatterCounts, packedOffsets, MPI_DOUBLE,
               E_prev_rank + (stride_rankX + 2),
               stride_rankY * (stride_rankX + 2), MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  // MPI_Barrier(MPI_COMM_WORLD);

  MPI_Scatterv(RPacked, scatterCounts, packedOffsets, MPI_DOUBLE,
               R_rank + (stride_rankX + 2), stride_rankY * (stride_rankX + 2),
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // MPI_Barrier(MPI_COMM_WORLD);

  for (niter = 0; niter < cb.niters; niter++) {

    if (cb.debug && (niter == 0) && world_rank == 0) {
      stats(E_prev, m, n, &mx, &sumSq);
      double l2norm = L2Norm(sumSq);
      repNorms(l2norm, mx, dt, m, n, -1, cb.stats_freq);
      if (cb.plot_freq && world_rank == 0)
        plotter->updatePlot(E_prev, -1, m + 1, n + 1);
    }

    // Update physical borders
    padPhysicalBoundaryCells(E_prev_rank, rankX, rankY, stride_rankY,
                             stride_rankX);

    // Ghost cell exchange
    exchangeGhostCells(E_prev_rank, ghostCellSendBuf, ghostCellRecvBuf,
                       ghostBufLen, rankX, rankY, stride_rankY, stride_rankX,
                       recvReq);

    // Perform computation
    innerBlockRowStartIndex = 0;
    innerBlockRowEndIndex = (stride_rankX + 2) * (stride_rankY + 2) - 1;

    calcComputeSpace(rankX, rankY, stride_rankY, stride_rankX,
                     innerBlockRowStartIndex, innerBlockRowEndIndex,
                     strideComp);

    // Inner Cell computation
    solveAlievPanfilov(E_rank, E_prev_rank, R_rank, innerBlockRowStartIndex,
                       innerBlockRowEndIndex, alpha, dt, stride_rankX,
                       stride_rankY, strideComp);

    int idxCompute;
    int startIdx;
    int endIdx;
    int stride;
    int recvCount = 0;
    int valid = 0;

    for (i = 0; i < 4; i++)
      if (recvReq[i] != MPI_REQUEST_NULL) {
        recvCount++;
      } else {
        idxCompute = i;
        valid = 0;
        solveAlievPanfilovEdge(E_rank, E_prev_rank, R_rank, ghostCellRecvBuf,
                               ghostBufLen, alpha, dt, stride_rankY,
                               stride_rankX, innerBlockRowStartIndex,
                               innerBlockRowEndIndex, strideComp, i, valid);
      }

    while (recvCount--) {
      MPI_Waitany(4, recvReq, &idxCompute, MPI_STATUS_IGNORE);

      valid = 1;
      solveAlievPanfilovEdge(E_rank, E_prev_rank, R_rank, ghostCellRecvBuf,
                             ghostBufLen, alpha, dt, stride_rankY, stride_rankX,
                             innerBlockRowStartIndex, innerBlockRowEndIndex,
                             strideComp, idxCompute, valid);
    }
    // // Top left corner
    int idx = innerBlockRowStartIndex - 1 - (stride_rankX + 2);
    solveAlievPanfilovTimeStep(E_rank, E_prev_rank, R_rank, idx, alpha, dt,
                               stride_rankX, stride_rankY);

    // // Top right corner
    idx = innerBlockRowStartIndex - (stride_rankX + 2) + strideComp;
    solveAlievPanfilovTimeStep(E_rank, E_prev_rank, R_rank, idx, alpha, dt,
                               stride_rankX, stride_rankY);

    // // Bottom right corner
    idx = innerBlockRowEndIndex + (stride_rankX + 2) + 1;
    solveAlievPanfilovTimeStep(E_rank, E_prev_rank, R_rank, idx, alpha, dt,
                               stride_rankX, stride_rankY);

    // // Bottom left corner
    idx = innerBlockRowEndIndex + (stride_rankX + 2) - strideComp;
    solveAlievPanfilovTimeStep(E_rank, E_prev_rank, R_rank, idx, alpha, dt,
                               stride_rankX, stride_rankY);
    // MPI_Barrier(MPI_COMM_WORLD);

    if (cb.stats_freq) {
      if (!(niter % cb.stats_freq)) {
        stats(E, m, n, &mx, &sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm, mx, dt, m, n, niter, cb.stats_freq);
      }
    }

    if (cb.plot_freq && !(niter % cb.plot_freq)) {
      MPI_Gatherv(E_rank + (stride_rankX + 2),
                  stride_rankY * (stride_rankX + 2), MPI_DOUBLE, E_prevPacked,
                  scatterCounts, packedOffsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      if (world_rank == 0) {

        reshapePackedArray(E_prevPacked, E_prev, sourceOffsets, world_size,
                           smallStrideX, smallStrideY, bigStrideX, bigStrideY,
                           numSmallRanksX, numSmallRanksY, n);
        plotter->updatePlot(E_prev, niter, m, n);
      }
    }

    // Swap current and previous meshes
    double *tmp = E_rank;
    E_rank = E_prev_rank;
    E_prev_rank = tmp;

  } // end of 'niter' loop at the beginning

  // Gather results back to rank 0 packed arrays
  MPI_Gatherv(E_prev_rank + (stride_rankX + 2),
              stride_rankY * (stride_rankX + 2), MPI_DOUBLE, E_prevPacked,
              scatterCounts, packedOffsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the
  // L2 and infinity norms via in-out parameters

  // MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 0) {
    ////////////////////////////////////////////////
    // Reshape packed array into original dimensions
    ////////////////////////////////////////////////

    reshapePackedArray(E_prevPacked, E_prev, sourceOffsets, world_size,
                       smallStrideX, smallStrideY, bigStrideX, bigStrideY,
                       numSmallRanksX, numSmallRanksY, n);
    stats(E_prev, m, n, &Linf, &sumSq);
    L2 = L2Norm(sumSq);

    // Swap pointers so we can re-use the arrays
    *_E = E;
    *_E_prev = E_prev;
  }
  delete[] scatterCounts;
  delete[] sourceOffsets;
  delete[] packedOffsets;
  delete[] ghostCellSendBuf;
  delete[] ghostCellRecvBuf;
  delete[] E_rank;
  delete[] E_prev_rank;
  delete[] R_rank;

  if (world_rank == 0) {
    free(E_prevPacked);
    free(RPacked);
  }
}

void printMat2(const char mesg[], double *E, int m, int n) {
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

void printMatNaive(const char mesg[], double *E, int m, int n) {
  int i;
  int j;
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process
  printf("%s\n", mesg);
  for (i = 0; i < m; i++) {
    printf("Rank%d row%d\t", world_rank, i);
    for (j = 0; j < n; j++) {
      printf("%6.3f ", E[i * n + j]);
    }
    printf("\n");
  }
}

void printMatRank(const char mesg[], int rank, double *E, int m, int n) {
  int i;
  int j;
  int world_size;
  int world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process
  for (int r = 0; r < world_size; r++) {
    if (world_rank == r && world_rank == rank) {
      printf("\n%s\n", mesg);
      for (i = 0; i < m; i++) {
        printf("Rank%2d row%2d\t", world_rank, i);
        for (j = 0; j < n; j++) {
          printf("%3.1f ", E[i * n + j]);
          // cout << E[i * n + j] << " ";
        }
        printf("\n");
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

double *alloc1DAll(int size) {
  double *E;
  // Ensures that allocatdd memory is aligned on a 16 byte boundary
  assert(E = (double *)memalign(16, sizeof(double) * size));
  return (E);
}

void padPhysicalBoundaryCells(double *E, int rankX, int rankY, int m, int n) {
  int i;
  // Top row
  if (rankY == 0) {
    // Top boundary cells
    for (i = (n + 2); i < 2 * (n + 2); i++) {
      E[i] = E[i + (n + 2) * 2];
    }

    // Top left corner
    if (rankX == 0) {
      for (i = (n + 2) + 1; i <= (m + 2) * (n + 2); i += (n + 2)) {
        E[i] = E[i + 2];
      }
    }
    // Top right corner
    else if (rankX == cb.px - 1) {
      for (i = 2 * (n + 2) - 2; i <= (m + 2) * (n + 2); i += (n + 2)) {
        E[i] = E[i - 2];
      }
    }
  }

  // Bottom row
  else if (rankY == cb.py - 1) {
    for (i = (m * (n + 2)); i < (m + 1) * (n + 2); i++) {
      E[i] = E[i - (n + 2) * 2];
    }

    // Bottom left corner
    if (rankX == 0) {
      for (i = (n + 2) + 1; i <= (m + 2) * (n + 2); i += (n + 2)) {
        E[i] = E[i + 2];
      }
    } else if (rankX == cb.px - 1) {
      // Right boundary cells
      for (i = 2 * (n + 2) - 2; i <= (m + 2) * (n + 2); i += (n + 2)) {
        E[i] = E[i - 2];
      }
    }
  }
  // Left edge
  else if (rankX == 0) {
    // Left boundary cells
    for (i = (n + 2) + 1; i <= (m + 2) * (n + 2); i += (n + 2)) {
      E[i] = E[i + 2];
    }
  }
  // Right edge
  else if (rankX == cb.px - 1) {
    // Right boundary cells
    for (i = 2 * (n + 2) - 2; i <= (m + 2) * (n + 2); i += (n + 2)) {
      E[i] = E[i - 2];
    }
  }
}

void buildSendGhostBuffer(double *E, double *buf, int dir, int m, int n) {
  if (!cb.noComm) {
    double *E_tmp = E + (n + 2) + 1; // move down 1 row and forward 1 column

    switch (dir) {
    case TOP:
      for (int i = 0; i < n; i++)
        buf[i] = E_tmp[i];
      break;
    case RIGHT:
      for (int i = 0; i < m; i++)
        buf[i] = E_tmp[(n - 1) + i * (n + 2)];
      break;
    case BOTTOM:
      for (int i = 0; i < n; i++)
        buf[i] = E_tmp[i + (n + 2) * (m - 1)];
      break;
    case LEFT:
      for (int i = 0; i < m; i++)
        buf[i] = E_tmp[i * (n + 2)];
      break;
    }
  }
}

void fillGhostCells(double *E, double *buf, int dir, int m, int n, int valid) {
  if (!cb.noComm || valid) {
    double *E_tmp;
    switch (dir) {
    case TOP:
      E_tmp = E + 1;
      for (int i = 0; i < n; i++)
        E_tmp[i] = buf[i];
      break;
    case RIGHT:
      E_tmp = E + (n + 2) + (n + 1);
      for (int i = 0; i < m; i++)
        E_tmp[i * (n + 2)] = buf[i];
      break;
    case BOTTOM:
      E_tmp = E + 1 + (n + 2) * (m + 1);
      for (int i = 0; i < n; i++)
        E_tmp[i] = buf[i];
      break;
    case LEFT:
      E_tmp = E + (n + 2);
      for (int i = 0; i < m; i++)
        E_tmp[i * (n + 2)] = buf[i];
      break;
    }
  }
}

void exchangeGhostCells(double *E, double *sendBuf, double *recvBuf, int bufLen,
                        int rankX, int rankY, int m, int n,
                        MPI_Request *recvReq) {

  recvReq[0] = MPI_REQUEST_NULL;
  recvReq[1] = MPI_REQUEST_NULL;
  recvReq[2] = MPI_REQUEST_NULL;
  recvReq[3] = MPI_REQUEST_NULL;

  if (!cb.noComm) {

    MPI_Request sendReq;

    // SEND Ghost Cells using non-blocking MPI calls
    // Send and receive top cells
    if ((rankY - 1) >= 0 && !cb.noComm) {
      double *bufSendTop = sendBuf;
      buildSendGhostBuffer(E, bufSendTop, TOP, m, n);
      MPI_Isend(bufSendTop, n, MPI_DOUBLE, (rankX + (rankY - 1) * cb.px), TOP,
                MPI_COMM_WORLD, &sendReq);
    }
    // Send and receive right cells
    if ((rankX + 1) < cb.px && !cb.noComm) {
      double *bufSendRight = sendBuf + bufLen;
      buildSendGhostBuffer(E, bufSendRight, RIGHT, m, n);
      MPI_Isend(bufSendRight, m, MPI_DOUBLE, (rankX + 1 + rankY * cb.px), RIGHT,
                MPI_COMM_WORLD, &sendReq);
    }
    // Send and receive bottom cells
    if ((rankY + 1) < cb.py && !cb.noComm) {
      double *bufSendBottom = sendBuf + 2 * bufLen;
      buildSendGhostBuffer(E, bufSendBottom, BOTTOM, m, n);
      MPI_Isend(bufSendBottom, n, MPI_DOUBLE, (rankX + (rankY + 1) * cb.px),
                BOTTOM, MPI_COMM_WORLD, &sendReq);
    }
    // Send and receive left cells
    if ((rankX - 1) >= 0 && !cb.noComm) {
      double *bufSendLeft = sendBuf + 3 * bufLen;
      buildSendGhostBuffer(E, bufSendLeft, LEFT, m, n);
      MPI_Isend(bufSendLeft, m, MPI_DOUBLE, (rankX - 1 + rankY * cb.px), LEFT,
                MPI_COMM_WORLD, &sendReq);
    }
    // RECEIVE Ghost Cells

    // Receive top cells
    if ((rankY - 1) >= 0) {
      double *bufRecvTop = recvBuf;
      MPI_Irecv(bufRecvTop, n, MPI_DOUBLE, (rankX + (rankY - 1) * cb.px),
                BOTTOM, MPI_COMM_WORLD, &recvReq[0]);
    }
    // Send and receive right cells
    if ((rankX + 1) < cb.px) {
      double *bufRecvRight = recvBuf + bufLen;
      MPI_Irecv(bufRecvRight, m, MPI_DOUBLE, (rankX + 1 + rankY * cb.px), LEFT,
                MPI_COMM_WORLD, &recvReq[1]);
    }
    // Send and receive bottom cells
    if ((rankY + 1) < cb.py) {
      double *bufRecvBottom = recvBuf + 2 * bufLen;
      MPI_Irecv(bufRecvBottom, n, MPI_DOUBLE, (rankX + (rankY + 1) * cb.px),
                TOP, MPI_COMM_WORLD, &recvReq[2]);
    }
    // Send and receive left cells
    if ((rankX - 1) >= 0) {
      double *bufRecvLeft = recvBuf + 3 * bufLen;
      MPI_Irecv(bufRecvLeft, m, MPI_DOUBLE, (rankX - 1 + rankY * cb.px), RIGHT,
                MPI_COMM_WORLD, &recvReq[3]);
    }
  }
}

void calcComputeSpace(int rankX, int rankY, int m, int n, int &startIdx,
                      int &endIdx, int &strideComp) {
  int i;
  int startX;
  int startY;
  int endX;
  int endY;
  strideComp = n - 2;

  // Calculate new stride for computation
  if (rankX == 0)
    strideComp--;
  if (rankX == cb.px - 1)
    strideComp--;

  // Calculate start positions for calculation
  // X component of start position
  if (rankX == 0)
    startX = 3;
  else
    startX = 2;

  // Y component of start position
  if (rankY == 0)
    startY = 3;
  else
    startY = 2;

  startIdx += startX + startY * (n + 2);

  // Calculate end positions for calculation
  // X component of end position
  if (rankX == cb.px - 1)
    endX = 3;
  else
    endX = 2;

  if (rankY == cb.py - 1)
    endY = 3;
  else
    endY = 2;

  endIdx -= (endX + endY * (n + 2));
}

void reshapePackedArray(double *E_prevPacked, double *E_prev,
                        int *sourceOffsets, int world_size, int smallStrideX,
                        int smallStrideY, int bigStrideX, int bigStrideY,
                        int numSmallRanksX, int numSmallRanksY, int n) {
  for (int rankIter = 0; rankIter < world_size; rankIter++) {
    int strideX =
        (rankIter % cb.px < numSmallRanksX) ? smallStrideX : bigStrideX;
    int strideY =
        (rankIter / cb.px < numSmallRanksY) ? smallStrideY : bigStrideY;
    int offset = sourceOffsets[rankIter];
    for (int i = 0; i < strideY; i++) {
      for (int j = 1; j <= strideX; j++) {
        E_prev[offset + i * (n + 2) + (j - 1)] =
            E_prevPacked[rankIter * (bigStrideX * bigStrideY + 2 * bigStrideY) +
                         i * (strideX + 2) + j];
      }
    }
  }
}
void solveAlievPanfilovTimeStep(double *E_rank, double *E_prev_rank,
                                double *R_rank, int i, double alpha, double dt,
                                int stride_rankX, int stride_rankY) {
  double *E_tmp, *E_prev_tmp, *R_tmp;
  E_tmp = E_rank;
  E_prev_tmp = E_prev_rank;
  R_tmp = R_rank;

  E_tmp[i] = E_prev_tmp[i] +
             alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] -
                      4 * E_prev_tmp[i] + E_prev_tmp[i + (stride_rankX + 2)] +
                      E_prev_tmp[i - (stride_rankX + 2)]);
  E_tmp[i] +=
      -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) +
             E_prev_tmp[i] * R_tmp[i]);
  R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) *
              (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
}

void solveAlievPanfilov(double *E_rank, double *E_prev_rank, double *R_rank,
                        int innerBlockRowStartIndex, int innerBlockRowEndIndex,
                        double alpha, double dt, int stride_rankX,
                        int stride_rankY, int strideComp) {
  double *E_tmp, *E_prev_tmp, *R_tmp;
  register double tempE = 0;
  register double tempE_prev = 0;
  register double tempR = 0;
  int i, j;

#ifdef SSE_VEC
  __m128d vec_alpha = _mm_set1_pd(alpha);
  __m128d vec_4f = _mm_set1_pd(-4.0f);
  __m128d vec_1f = _mm_set1_pd(1.0f);
  __m128d vec_neg1f = _mm_set1_pd(-1.0f);
  __m128d vec_a = _mm_set1_pd(a);
  __m128d vec_b = _mm_set1_pd(b);
  __m128d vec_dt = _mm_set1_pd(dt);
  __m128d vec_kk = _mm_set1_pd(kk);
  __m128d vec_eps = _mm_set1_pd(epsilon);
  __m128d vec_M1 = _mm_set1_pd(M1);
  __m128d vec_M2 = _mm_set1_pd(M2);

  // Solve for the excitation, a PDE
  for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex;
       j += (stride_rankX + 2)) {
    E_tmp = E_rank + j;
    E_prev_tmp = E_prev_rank + j;
    R_tmp = R_rank + j;
    for (i = 0; i < strideComp - 2; i += 2) {

      __m128d vec_center = _mm_loadu_pd(&E_prev_tmp[i]);
      __m128d vec_north = _mm_loadu_pd(&E_prev_tmp[i - (stride_rankX + 2)]);
      __m128d vec_east = _mm_loadu_pd(&E_prev_tmp[i + 1]);
      __m128d vec_south = _mm_loadu_pd(&E_prev_tmp[i + (stride_rankX + 2)]);
      __m128d vec_west = _mm_loadu_pd(&E_prev_tmp[i - 1]);
      __m128d vec_R = _mm_loadu_pd(&R_tmp[i]);

      __m128d vec_4xcenter = _mm_mul_pd(vec_center, vec_4f);

      __m128d vec_res = _mm_add_pd(vec_4xcenter, vec_north);
      vec_res = _mm_add_pd(vec_res, vec_east);
      vec_res = _mm_add_pd(vec_res, vec_south);
      vec_res = _mm_add_pd(vec_res, vec_west);
      vec_res = _mm_mul_pd(vec_res, vec_alpha);
      vec_res = _mm_add_pd(vec_res, vec_center);

      __m128d vec_center_less_1 = _mm_sub_pd(vec_center, vec_1f);
      __m128d vec_center_less_a = _mm_sub_pd(vec_center, vec_a);

      __m128d vec_mula = _mm_mul_pd(vec_kk, vec_center);
      vec_mula = _mm_mul_pd(vec_mula, vec_center_less_a);
      vec_mula = _mm_mul_pd(vec_mula, vec_center_less_1);

      vec_mula = _mm_add_pd(vec_mula, _mm_mul_pd(vec_center, vec_R));
      vec_mula = _mm_mul_pd(vec_mula, vec_dt);
      vec_res = _mm_sub_pd(vec_res, vec_mula);

      _mm_storeu_pd(&E_tmp[i], vec_res);

      vec_mula = _mm_sub_pd(vec_center_less_1, vec_b);
      vec_mula = _mm_mul_pd(vec_mula, vec_center);
      vec_mula = _mm_mul_pd(vec_mula, vec_kk);
      vec_mula = _mm_sub_pd(_mm_mul_pd(vec_neg1f, vec_R), vec_mula);

      __m128d vec_num = _mm_mul_pd(vec_M1, vec_R);
      __m128d vec_denom = _mm_add_pd(vec_center, vec_M2);
      __m128d vec_mulb = _mm_div_pd(vec_num, vec_denom);
      vec_mulb = _mm_add_pd(vec_eps, vec_mulb);
      vec_mulb = _mm_mul_pd(vec_mulb, vec_mula);
      vec_mulb = _mm_mul_pd(vec_dt, vec_mulb);
      // vec_res = _mm_mul_pd(vec_dt, vec_mulb);
      vec_res = _mm_add_pd(vec_mulb, vec_R);

      _mm_storeu_pd(&R_tmp[i], vec_res);
    }

    for (; i < strideComp; i++) {
      E_tmp[i] =
          E_prev_tmp[i] +
          alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] +
                   E_prev_tmp[i + (stride_rankX + 2)] +
                   E_prev_tmp[i - (stride_rankX + 2)]);
      E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) *
                             (E_prev_tmp[i] - 1) +
                         E_prev_tmp[i] * R_tmp[i]);
      R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) *
                  (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
    }
  }

#else
  // Solve for the excitation, a PDE
  for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex;
       j += (stride_rankX + 2)) {
    E_tmp = E_rank + j;
    E_prev_tmp = E_prev_rank + j;
    R_tmp = R_rank + j;
    for (i = 0; i < strideComp; i++) {
      tempE_prev = E_prev_tmp[i];
      tempR = R_tmp[i];
      tempE =
          E_prev_tmp[i] +
          alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * tempE_prev +
                   E_prev_tmp[i + (stride_rankX + 2)] +
                   E_prev_tmp[i - (stride_rankX + 2)]);
      tempE += -dt * (kk * tempE_prev * (tempE_prev - a) *
                          (tempE_prev - 1) +
                      tempE_prev * tempR);
      R_tmp[i] += dt * (epsilon + M1 * tempR / (tempE_prev + M2)) *
                  (-tempR - kk * tempE_prev * (tempE_prev - b - 1));
      E_tmp[i] = tempE;
    }
  }
#endif
}

void solveAlievPanfilovEdge(double *E_rank, double *E_prev_rank, double *R_rank,
                            double *ghostCellRecvBuf, int ghostBufLen,
                            double alpha, double dt, int stride_rankY,
                            int stride_rankX, int innerBlockRowStartIndex,
                            int innerBlockRowEndIndex, int strideComp,
                            int idxCompute, int valid) {
  int startIdx, endIdx, stride;

  if (idxCompute == 0) {
    // Top edge computation
    fillGhostCells(E_prev_rank, ghostCellRecvBuf, TOP, stride_rankY,
                   stride_rankX, valid);

    startIdx = innerBlockRowStartIndex - (stride_rankX + 2);
    endIdx = innerBlockRowStartIndex - (stride_rankX + 2) + strideComp - 1;
    stride = strideComp;

    solveAlievPanfilov(E_rank, E_prev_rank, R_rank, startIdx, endIdx, alpha, dt,
                       stride_rankX, stride_rankY, stride);
  }

  else if (idxCompute == 3) {
    // Left edge computation
    //
    fillGhostCells(E_prev_rank, ghostCellRecvBuf + 3 * ghostBufLen, LEFT,
                   stride_rankY, stride_rankX, valid);
    startIdx = innerBlockRowStartIndex - 1;
    endIdx = innerBlockRowEndIndex - strideComp;
    stride = 1;
    solveAlievPanfilov(E_rank, E_prev_rank, R_rank, startIdx, endIdx, alpha, dt,
                       stride_rankX, stride_rankY, stride);
  }

  else if (idxCompute == 1) {
    //  Right edge computation

    fillGhostCells(E_prev_rank, ghostCellRecvBuf + ghostBufLen, RIGHT,
                   stride_rankY, stride_rankX, valid);
    startIdx = innerBlockRowStartIndex + strideComp;
    endIdx = innerBlockRowEndIndex + 1;
    stride = 1;
    solveAlievPanfilov(E_rank, E_prev_rank, R_rank, startIdx, endIdx, alpha, dt,
                       stride_rankX, stride_rankY, stride);
  }

  // //  Bottom edge computation
  else if (idxCompute == 2) {
    // printf("rank=%d, idxCompute: %d\n", world_rank, idxCompute);
    fillGhostCells(E_prev_rank, ghostCellRecvBuf + 2 * ghostBufLen, BOTTOM,
                   stride_rankY, stride_rankX, valid);

    startIdx = innerBlockRowEndIndex + (stride_rankX + 2) - strideComp + 1;
    endIdx = innerBlockRowEndIndex + 1 + stride_rankX + 2 - 1;
    stride = strideComp;
    solveAlievPanfilov(E_rank, E_prev_rank, R_rank, startIdx, endIdx, alpha, dt,
                       stride_rankX, stride_rankY, stride);
  }
}
