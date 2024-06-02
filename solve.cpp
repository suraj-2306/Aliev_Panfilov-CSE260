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
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string>
#include <malloc.h>
using namespace std;

#define TOP 0
#define RIGHT 1
#define BOTTOM 2
#define LEFT 3

void repNorms(double l2norm, double mx, double dt, int m, int n, int niter,
              int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
void printMatNaive(const char mesg[], double *E, int m, int n);
void printMatRank(const char mesg[], int rank, double *E, int m, int n);
double *alloc1DAll(int size);
double *createSendGhostBuffer(double *E, int dir, int m, int n);
void fillGhostCells(double *E, double *buf, int dir, int m, int n);
void exchangeGhostCells(double *E, int rankX, int rankY, int m, int n);

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
double L2Norm(double sumSq)
{
    double l2norm = sumSq / (double)((cb.m) * (cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt,
           Plotter *plotter, double &L2, double &Linf)
{

    // Simulated time is different from the integer timestep number
    double t = 0.0;

    double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = R;
    double *E_tmp = *_E;
    double *E_prev_tmp = *_E_prev;
    double mx, sumSq;
    int niter;
    int m = cb.m, n = cb.n;
    int innerBlockRowStartIndex;
    int innerBlockRowEndIndex;
    // We continue to sweep over the mesh until the simulation has reached
    // the desired number of iterations

    int world_size;
    int world_rank;
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
    int stride_rankX = (world_rank % cb.px < numSmallRanksX) ? smallStrideX : bigStrideX;
    int stride_rankY = (world_rank / cb.px < numSmallRanksY) ? smallStrideY : bigStrideY;
    int i, j;
    // if (world_rank == 5)
    //     printf("small stride  %d %d\n big stride %d %d\n numBigranks %d %d\n "
    //            "numSmallRanks %d %d\n striderank %d %d \n",
    //            smallStrideX, smallStrideY, bigStrideX, bigStrideY, numBigRanksX,
    //            numBigRanksY, numSmallRanksX, numSmallRanksY, stride_rankX,
    //            stride_rankY);

    int *scatterCounts = (int *)malloc(sizeof(int) * world_size);
    int *sourceOffsets = (int *)malloc(sizeof(int) * world_size);
    int *packedOffsets = (int *)malloc(sizeof(int) * world_size);
    double *E_prevPacked = (double *)malloc(sizeof(double) * cb.px * bigStrideX * cb.py * bigStrideY + bigStrideY * 2); // alloc1DAll(cb.px * bigStrideX * cb.py * bigStrideY + bigStrideY * 2);
    double *RPacked = (double *)malloc(sizeof(double) * cb.px * bigStrideX * cb.py * bigStrideY + bigStrideY * 2);      // alloc1DAll(cb.px * bigStrideX * cb.py * bigStrideY + bigStrideY * 2);

    if (world_rank == 0)
    {
        int *sourceOffsetsX = (int *)malloc(sizeof(int) * cb.px);
        int *sourceOffsetsY = (int *)malloc(sizeof(int) * cb.py);
        sourceOffsetsX[i] = 0;
        sourceOffsetsY[i] = 0;
        for (i = 1; i < cb.px; i++)
        {
            sourceOffsetsX[i] = sourceOffsetsX[i - 1] + ((i < numSmallRanksX) ? smallStrideX : bigStrideX);
        }
        for (i = 1; i < cb.py; i++)
        {
            sourceOffsetsY[i] = sourceOffsetsY[i - 1] + ((i < numSmallRanksY) ? smallStrideY : bigStrideY) * (n + 2);
        }

        for (i = 0; i < cb.py; i++)
        {
            for (j = 0; j < cb.px; j++)
            {
                sourceOffsets[j + i * cb.px] = sourceOffsetsX[j] + sourceOffsetsY[i];
                scatterCounts[j + i * cb.px] = (((j < numSmallRanksX) ? smallStrideX : bigStrideX) + 2) * ((i < numSmallRanksY) ? smallStrideY : bigStrideY);
            }
        }

        for (i = 0; i < cb.py; i++)
        {
            for (j = 0; j < cb.px; j++)
            {
                printf("%d:%d\t", sourceOffsets[j + i * cb.px], scatterCounts[j + i * cb.px]);
            }
            printf("\n");
        }

        free(sourceOffsetsX);
        free(sourceOffsetsY);

        for (int rankIter = 0; rankIter < world_size; rankIter++)
        {
            int strideX = (rankIter % cb.px < numSmallRanksX) ? smallStrideX : bigStrideX;
            int strideY = (rankIter / cb.px < numSmallRanksY) ? smallStrideY : bigStrideY;
            int offset = sourceOffsets[rankIter];
            for (i = 0; i < strideY; i++)
            {
                for (j = 1; j <= strideX; j++)
                {
                    E_prevPacked[rankIter * (bigStrideX * bigStrideY + 2 * bigStrideY) + i * (strideX + 2) + j] = E_prev[offset + i * (n + 2) + (j - 1)];
                    RPacked[rankIter * (bigStrideX * bigStrideY + 2 * bigStrideY) + i * (strideX + 2) + j] = R[offset + i * (n + 2) + (j - 1)];
                }
            }
            packedOffsets[rankIter] = rankIter * (bigStrideX * bigStrideY + 2 * bigStrideY);
        }

        // printMatNaive("RPacked", RPacked, world_size, bigStrideX * bigStrideY + 2 * bigStrideY);
    }

    double *E_rank = (double *)malloc(sizeof(double) * (stride_rankY + 2) * (stride_rankX + 2));
    double *E_prev_rank = (double *)malloc(sizeof(double) * (stride_rankY + 2) * (stride_rankX + 2));
    double *R_rank = (double *)malloc(sizeof(double) * (stride_rankY + 2) * (stride_rankX + 2));
    // // // double *topRow_rank = (double *)malloc(sizeof(double) * (n + 2));
    // // // double *bottomRow_rank = (double *)malloc(sizeof(double) * (n + 2));

    MPI_Scatterv(E_prevPacked, scatterCounts, packedOffsets, MPI_DOUBLE,
                 E_prev_rank + (stride_rankX + 2), stride_rankY * (stride_rankX + 2), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatterv(E_prevPacked, scatterCounts, packedOffsets, MPI_DOUBLE,
                 R_rank + (stride_rankX + 2), stride_rankY * (stride_rankX + 2), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // printMatRank("E_prev_rank", 5, E_prev_rank, stride_rankY + 2, stride_rankX + 2);
    // printMatRank("R_rank", 1, R_rank, stride_rankY + 2, stride_rankX + 2);

    for (niter = 0; niter < cb.niters; niter++)
    {

        if (cb.debug && (niter == 0))
        {
            stats(E_prev, m, n, &mx, &sumSq);
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm, mx, dt, m, n, -1, cb.stats_freq);
            if (cb.plot_freq && world_rank == 0)
                plotter->updatePlot(E_prev, -1, m + 1, n + 1);
        }

        // Update physical borders
        // CHECK FOR SMALL STEP SIZES

        // Top row
        if (rankY == 0)
        {
            // Top boundary cells
            for (i = (stride_rankX + 2); i < 2 * (stride_rankX + 2); i++)
            {
                E_prev_rank[i] = E_prev_rank[i + (stride_rankX + 2) * 2];
            }

            // Top left corner
            if (rankX == 0)
            {
                for (i = (stride_rankX + 2) + 1; i <= (stride_rankY + 2) * (stride_rankX + 2); i += (stride_rankX + 2))
                {
                    E_prev_rank[i] = E_prev_rank[i + 2];
                }
            }
            // Top right corner
            else if (rankX == cb.px - 1)
            {
                for (i = 2 * (stride_rankX + 2) - 2; i <= (stride_rankY + 2) * (stride_rankX + 2); i += (stride_rankX + 2))
                {
                    E_prev_rank[i] = E_prev_rank[i - 2];
                }
            }
        }

        // Bottom row
        else if (rankY == cb.py - 1)
        {
            for (i = (stride_rankY * (stride_rankX + 2)); i < (stride_rankY + 1) * (stride_rankX + 2); i++)
            {
                E_prev_rank[i] = E_prev_rank[i - (stride_rankX + 2) * 2];
            }

            // Bottom left corner
            if (rankX == 0)
            {
                for (i = (stride_rankX + 2) + 1; i <= (stride_rankY + 2) * (stride_rankX + 2); i += (stride_rankX + 2))
                {
                    E_prev_rank[i] = E_prev_rank[i + 2];
                }
            }
            else if (rankX == cb.px - 1)
            {
                // Right boundary cells
                for (i = 2 * (stride_rankX + 2) - 2; i <= (stride_rankY + 2) * (stride_rankX + 2); i += (stride_rankX + 2))
                {
                    E_prev_rank[i] = E_prev_rank[i - 2];
                }
            }
        }
        // Left edge
        else if (rankX == 0)
        {
            // Left boundary cells
            for (i = (stride_rankX + 2) + 1; i <= (stride_rankY + 2) * (stride_rankX + 2); i += (stride_rankX + 2))
            {
                E_prev_rank[i] = E_prev_rank[i + 2];
            }
        }
        // Right edge
        else if (rankX == cb.px - 1)
        {
            // Right boundary cells
            for (i = 2 * (stride_rankX + 2) - 2; i <= (stride_rankY + 2) * (stride_rankX + 2); i += (stride_rankX + 2))
            {
                E_prev_rank[i] = E_prev_rank[i - 2];
            }
        }

        printMatRank("E_prev_rank_padded0", 0, E_prev_rank, stride_rankY + 2, stride_rankX + 2);
        MPI_Barrier(MPI_COMM_WORLD);
        printMatRank("E_prev_rank_padded2", 2, E_prev_rank, stride_rankY + 2, stride_rankX + 2);
        MPI_Barrier(MPI_COMM_WORLD);

        // Ghost cell exchange

        exchangeGhostCells(E_prev_rank, rankX, rankY, stride_rankY, stride_rankX);

        MPI_Barrier(MPI_COMM_WORLD);
        printMatRank("E_prev_rank_with_Ghost_cells", 0, E_prev_rank, stride_rankY + 2, stride_rankX + 2);

        //     // Perform computation

        //     innerBlockRowStartIndex =
        //         ((world_rank == 0) ? 2 : 1) * (n + 2) +
        //         1; // Ignore physical boundary padding at the top of first chunk of
        //         rows
        //     innerBlockRowEndIndex =
        //         (stride_rank - ((world_rank == world_size - 1) ? 1 : 0)) * (n + 2)
        //         + 1; // Ignore physical boundary padding at the bottom of last
        //         chunk of
        //            // rows

        // #define FUSED 1

        // #ifdef FUSED

        //     // printMatRank("E_prev_rank0", 0, E_prev_rank, stride_rank + 2, n +
        //     2);
        //     // printMatRank("E_prev_rank1", 1, E_prev_rank, stride_rank + 2, n +
        //     2);

        //     // Solve for the excitation, a PDE
        //     for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex;
        //          j += (n + 2)) {
        //       E_tmp = E_rank + j;
        //       E_prev_tmp = E_prev_rank + j;
        //       R_tmp = R_rank + j;
        //       for (i = 0; i < n; i++) {
        //         E_tmp[i] =
        //             E_prev_tmp[i] +
        //             alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 *
        //             E_prev_tmp[i] +
        //                      E_prev_tmp[i + (n + 2)] + E_prev_tmp[i - (n + 2)]);
        //         E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) *
        //                                (E_prev_tmp[i] - 1) +
        //                            E_prev_tmp[i] * R_tmp[i]);
        //         R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) *
        //                     (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b -
        //                     1));
        //       }
        //     }

        // #else
        //     // Solve for the excitation, a PDE
        //     for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex;
        //          j += (n + 2)) {
        //       E_tmp = E_rank + j;
        //       E_prev_tmp = E_prev_rank + j;
        //       for (i = 0; i < n; i++) {
        //         E_tmp[i] =
        //             E_prev_tmp[i] +
        //             alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 *
        //             E_prev_tmp[i] +
        //                      E_prev_tmp[i + (n + 2)] + E_prev_tmp[i - (n + 2)]);
        //       }
        //     }

        //     /*
        //      * Solve the ODE, advancing excitation and recovery variables
        //      *     to the next timtestep
        //      */

        //     for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex;
        //          j += (n + 2)) {
        //       E_tmp = E_rank + j;
        //       E_prev_tmp = E_prev_rank + j;
        //       R_tmp = R_rank + j;
        //       for (i = 0; i < n; i++) {
        //         E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) *
        //                                (E_prev_tmp[i] - 1) +
        //                            E_prev_tmp[i] * R_tmp[i]);
        //         R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) *
        //                     (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b -
        //                     1));
        //       }
        //     }
        // #endif
        //     /////////////////////////////////////////////////////////////////////////////////

        //     if (cb.stats_freq) {
        //       if (!(niter % cb.stats_freq)) {
        //         stats(E, m, n, &mx, &sumSq);
        //         double l2norm = L2Norm(sumSq);
        //         repNorms(l2norm, mx, dt, m, n, niter, cb.stats_freq);
        //       }
        //     }

        //     if (cb.plot_freq) {
        //       MPI_Gatherv(E_rank + (n + 2), stride_rank * (n + 2), MPI_DOUBLE, E,
        //                   scatterCounts, sourceOffsets, MPI_DOUBLE, 0,
        //                   MPI_COMM_WORLD);
        //       if (world_rank == 0 && !(niter % cb.plot_freq)) {
        //         plotter->updatePlot(E, niter, m, n);
        //       }
        //     }

        //     // Swap current and previous meshes
        //     double *tmp = E_rank;
        //     E_rank = E_prev_rank;
        //     E_prev_rank = tmp;

        //     // printMatRank("E_prev_rank0 after swap", 0, E_prev_rank, stride_rank
        //     + 2,
        //     // n + 2); printMatRank("E_rank0 after computation", 0, E_rank,
        //     stride_rank
        //     // + 2, n + 2);

        //   } // end of 'niter' loop at the beginning

        //   // MPI_Barrier(MPI_COMM_WORLD);

        //   // Gather results back to rank 0
        //   MPI_Gatherv(E_prev_rank + (n + 2), stride_rank * (n + 2), MPI_DOUBLE,
        //   E_prev,
        //               scatterCounts, sourceOffsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //   MPI_Gatherv(R_rank + (n + 2), stride_rank * (n + 2), MPI_DOUBLE, R,
        //               scatterCounts, sourceOffsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //   //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and
        //   //  infinity norms via in-out parameters

        //   // MPI_Barrier(MPI_COMM_WORLD);

        //   if (world_rank == 0) {
        //     stats(E_prev, m, n, &Linf, &sumSq);
        //     L2 = L2Norm(sumSq);

        //     // Swap pointers so we can re-use the arrays
        //     *_E = E;
        //     *_E_prev = E_prev;
    }
}

void printMat2(const char mesg[], double *E, int m, int n)
{
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m > 34)
        return;
#endif
    printf("%s\n", mesg);
    for (i = 0; i < (m + 2) * (n + 2); i++)
    {
        int rowIndex = i / (n + 2);
        int colIndex = i % (n + 2);
        if ((colIndex > 0) && (colIndex < n + 1))
            if ((rowIndex > 0) && (rowIndex < m + 1))
                printf("%6.3f ", E[i]);
        if (colIndex == n + 1)
            printf("\n");
    }
}

void printMatNaive(const char mesg[], double *E, int m, int n)
{
    int i;
    int j;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process
    printf("%s\n", mesg);
    for (i = 0; i < m; i++)
    {
        printf("Rank%d row%d\t", world_rank, i);
        for (j = 0; j < n; j++)
        {
            printf("%6.3f ", E[i * n + j]);
        }
        printf("\n");
    }
}

void printMatRank(const char mesg[], int rank, double *E, int m, int n)
{
    int i;
    int j;
    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process
    for (int r = 0; r < world_size; r++)
    {
        if (world_rank == r && world_rank == rank)
        {
            printf("\n%s\n", mesg);
            for (i = 0; i < m; i++)
            {
                printf("Rank%2d row%2d\t", world_rank, i);
                for (j = 0; j < n; j++)
                {
                    printf("%3.1f ", E[i * n + j]);
                    // cout << E[i * n + j] << " ";
                }
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

double *alloc1DAll(int size)
{
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E = (double *)memalign(16, sizeof(double) * size));
    return (E);
}

double *createSendGhostBuffer(double *E, int dir, int m, int n)
{
    double *E_tmp = E + (n + 2) + 1; // move down 1 row and forward 1 column
    double *buf;
    switch (dir)
    {
    case TOP:
        buf = new double[n];
        for (int i = 0; i < n; i++)
            buf[i] = E_tmp[i];
        break;
    case RIGHT:
        buf = new double[m];
        for (int i = 0; i < m; i++)
            buf[i] = E_tmp[(n - 1) + i * (n + 2)];
        break;
    case BOTTOM:
        buf = new double[n];
        for (int i = 0; i < n; i++)
            buf[i] = E_tmp[i + (n + 2) * (m - 1)];
        break;
    case LEFT:
        buf = new double[m];
        for (int i = 0; i < m; i++)
            buf[i] = E_tmp[i * (n + 2)];
        break;
    }
    return buf;
}

void fillGhostCells(double *E, double *buf, int dir, int m, int n)
{
    double *E_tmp;

    switch (dir)
    {
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

void exchangeGhostCells(double *E, int rankX, int rankY, int m, int n)
{
    double *bufSendTop = createSendGhostBuffer(E, TOP, m, n);
    double *bufSendRight = createSendGhostBuffer(E, RIGHT, m, n);
    double *bufSendBottom = createSendGhostBuffer(E, BOTTOM, m, n);
    double *bufSendLeft = createSendGhostBuffer(E, LEFT, m, n);

    double *bufRecvTop = new double[n];
    double *bufRecvRight = new double[m];
    double *bufRecvBottom = new double[n];
    double *bufRecvLeft = new double[m];

    // Send and receive bottom cells
    if ((rankY + 1) < cb.py)
        MPI_Sendrecv(bufSendBottom, n, MPI_DOUBLE, (rankX + (rankY + 1) * cb.px), BOTTOM,
                     bufRecvBottom, n, MPI_DOUBLE, (rankX + (rankY + 1) * cb.px), TOP,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send and receive right cells
    if ((rankX + 1) < cb.px)
        MPI_Sendrecv(bufSendRight, m, MPI_DOUBLE, (rankX + 1 + rankY * cb.px), RIGHT,
                     bufRecvRight, m, MPI_DOUBLE, (rankX + 1 + rankY * cb.px), LEFT,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send and receive top cells
    if ((rankY - 1) >= 0)
        MPI_Sendrecv(bufSendTop, n, MPI_DOUBLE, (rankX + (rankY - 1) * cb.px), TOP,
                     bufRecvTop, n, MPI_DOUBLE, (rankX + (rankY - 1) * cb.px), BOTTOM,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send and receive left cells
    if ((rankX - 1) >= 0)
        MPI_Sendrecv(bufSendLeft, m, MPI_DOUBLE, (rankX - 1 + rankY * cb.px), LEFT,
                     bufRecvLeft, m, MPI_DOUBLE, (rankX - 1 + rankY * cb.px), RIGHT,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    fillGhostCells(E, bufRecvTop, TOP, m, n);
    fillGhostCells(E, bufRecvRight, RIGHT, m, n);
    fillGhostCells(E, bufRecvBottom, BOTTOM, m, n);
    fillGhostCells(E, bufRecvLeft, LEFT, m, n);

    free(bufSendTop);
    free(bufSendRight);
    free(bufSendBottom);
    free(bufSendLeft);
}