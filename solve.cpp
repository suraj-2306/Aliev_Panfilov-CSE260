/*
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 *
 * Modified and  restructured by Scott B. Baden, UCSD
 *
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#include <mpi.h>
using namespace std;

#define TOP 0
#define BOTTOM 1

void repNorms(double l2norm, double mx, double dt, int m, int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
void printMatNaive(const char mesg[], double *E, int m, int n);

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

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf)
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
    int innerBlockRowStartIndex = (n + 2) + 1;
    // int innerBlockRowEndIndex = (((m + 2) * (n + 2) - 1) - (n)) - (n + 2);
    int innerBlockRowEndIndex;
    // We continue to sweep over the mesh until the simulation has reached
    // the desired number of iterations

    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process
    int smallStride = (m + 2) / world_size;
    int bigStride = smallStride + 1;
    int numBigRanks = (m + 2) % world_size;
    int numSmallRanks = world_size - numBigRanks;
    int stride_rank = (world_rank < numSmallRanks) ? smallStride : bigStride;
    int *scatterCounts = (int *)malloc(sizeof(int) * world_size);
    int *sourceOffsets = (int *)malloc(sizeof(int) * world_size);
    double *E_prev_rank = (double *)malloc(sizeof(double) * (stride_rank + 2) * (n + 2));
    double *R_rank = (double *)malloc(sizeof(double) * (stride_rank + 2) * (n + 2));
    double *topRow_rank = (double *)malloc(sizeof(double) * (n + 2));
    double *bottomRow_rank = (double *)malloc(sizeof(double) * (n + 2));
    int i, j;

    for (int i = 0; i < world_size; i++)
    {
        scatterCounts[i] = ((i < numSmallRanks) ? smallStride : bigStride) * (n + 2);
    }

    for (i = 0; i < world_size; i++)
    {
        sourceOffsets[i] = i * (n + 2) * smallStride + ((i <= numSmallRanks) ? 0 : (n + 2));
    }

    MPI_Scatterv(E_prev, scatterCounts, sourceOffsets, MPI_DOUBLE,
                 E_prev_rank + (n + 2), (n + 2) * stride_rank, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(R, scatterCounts, sourceOffsets, MPI_DOUBLE,
                 R_rank + (n + 2), (n + 2) * stride_rank, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    for (niter = 0; niter < cb.niters; niter++)
    {

        if (cb.debug && (niter == 0))
        {
            stats(E_prev, m, n, &mx, &sumSq);
            double l2norm = L2Norm(sumSq);
            repNorms(l2norm, mx, dt, m, n, -1, cb.stats_freq);
            if (cb.plot_freq)
                plotter->updatePlot(E, -1, m + 1, n + 1);
        }

        // Update physical borders
        if (world_rank == 0)
        {
            // Fills in the TOP Ghost Cells
            for (i = (n + 2); i < 2 * (n + 2); i++)
            {
                E_prev_rank[i] = E_prev_rank[i + (n + 2) * 2];
            }
        }

        else if (world_rank == world_size - 1)
        {
            // Fills in the BOTTOM Ghost Cells
            for (i = (stride_rank * (n + 2)); i < (stride_rank + 1) * (n + 2); i++)
            {
                E_prev_rank[i] = E_prev_rank[i - (n + 2) * 2];
            }
        }

        // Fills in the RIGHT Ghost Cells
        for (i = 2 * (n + 2) - 1; i < (stride_rank + 2) * (n + 2); i += (n + 2))
        {
            E_prev_rank[i] = E_prev_rank[i - 2];
        }

        // Fills in the LEFT Ghost Cells
        for (i = (n + 2); i < (stride_rank + 2) * (n + 2); i += (n + 2))
        {
            E_prev_rank[i] = E_prev_rank[i + 2];
        }

        // Buffer my ghost cells
        for (i = 0; i < (n + 2); i++)
        {
            topRow_rank[i] = E_prev_rank[i + (n + 2)];                  // Copy second row
            bottomRow_rank[i] = E_prev_rank[i + stride_rank * (n + 2)]; // Copy second last row
        }

        // Share ghost cells before computation
        if (world_rank == 0)
        {
            MPI_Send(bottomRow_rank, (n + 2), MPI_DOUBLE, world_rank + 1, BOTTOM, MPI_COMM_WORLD);
            MPI_Recv(E_prev_rank + (stride_rank + 1) * (n + 2), (n + 2), MPI_DOUBLE, world_rank + 1, TOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Copy into last row
        }

        else if (world_rank == world_size - 1)
        {
            MPI_Send(topRow_rank, (n + 2), MPI_DOUBLE, world_rank - 1, TOP, MPI_COMM_WORLD);
            MPI_Recv(E_prev_rank, (n + 2), MPI_DOUBLE, world_rank - 1, BOTTOM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        else
        {
            MPI_Send(topRow_rank, (n + 2), MPI_DOUBLE, world_rank - 1, TOP, MPI_COMM_WORLD);
            MPI_Send(bottomRow_rank, (n + 2), MPI_DOUBLE, world_rank + 1, BOTTOM, MPI_COMM_WORLD);
            MPI_Recv(E_prev_rank + (stride_rank + 1) * (n + 2), (n + 2), MPI_DOUBLE, world_rank + 1, TOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Copy into last row
            MPI_Recv(E_prev_rank, (n + 2), MPI_DOUBLE, world_rank - 1, BOTTOM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);                            // Copy into first row
        }

        printf("\nProcess %d ", world_rank);
        if (world_rank < numSmallRanks)
            printMatNaive("E_prev_rank_small", E_prev_rank, smallStride + 2, n + 2);
        else
            printMatNaive("E_prev_rank_big", E_prev_rank, bigStride + 2, n + 2);

        // Perform computation

        innerBlockRowEndIndex = stride_rank * (n + 2);
#define FUSED 1

#ifdef FUSED

        // Solve for the excitation, a PDE
        // for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n + 2))
        // {
        //     E_tmp = E + stride_rank * (n + 2) + j - innerBlockRowStartIndex;
        //     E_prev_tmp = E_prev_rank + j;
        //     R_tmp = R + stride_rank * (n + 2) + j - innerBlockRowStartIndex;
        //     for (i = 0; i < n; i++)
        //     {
        //         E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + (n + 2)] + E_prev_tmp[i - (n + 2)]);
        //         E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
        //         R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
        //     }
        // }
#else
        // Solve for the excitation, a PDE
        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n + 2))
        {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for (i = 0; i < n; i++)
            {
                E_tmp[i] = E_prev_tmp[i] + alpha * (E_prev_tmp[i + 1] + E_prev_tmp[i - 1] - 4 * E_prev_tmp[i] + E_prev_tmp[i + (n + 2)] + E_prev_tmp[i - (n + 2)]);
            }
        }

        /*
         * Solve the ODE, advancing excitation and recovery variables
         *     to the next timtestep
         */

        for (j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j += (n + 2))
        {
            E_tmp = E + j;
            R_tmp = R + j;
            E_prev_tmp = E_prev + j;
            for (i = 0; i < n; i++)
            {
                E_tmp[i] += -dt * (kk * E_prev_tmp[i] * (E_prev_tmp[i] - a) * (E_prev_tmp[i] - 1) + E_prev_tmp[i] * R_tmp[i]);
                R_tmp[i] += dt * (epsilon + M1 * R_tmp[i] / (E_prev_tmp[i] + M2)) * (-R_tmp[i] - kk * E_prev_tmp[i] * (E_prev_tmp[i] - b - 1));
            }
        }
#endif
        /////////////////////////////////////////////////////////////////////////////////

        if (cb.stats_freq)
        {
            if (!(niter % cb.stats_freq))
            {
                stats(E, m, n, &mx, &sumSq);
                double l2norm = L2Norm(sumSq);
                repNorms(l2norm, mx, dt, m, n, niter, cb.stats_freq);
            }
        }

        if (cb.plot_freq)
        {
            if (!(niter % cb.plot_freq))
            {
                plotter->updatePlot(E, niter, m, n);
            }
        }

        // Swap current and previous meshes
        // double *tmp = E;
        // E = E_prev;
        // E_prev = tmp;

    } // end of 'niter' loop at the beginning

    //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

    // stats(E_prev, m, n, &Linf, &sumSq);
    // L2 = L2Norm(sumSq);

    // // Swap pointers so we can re-use the arrays
    // *_E = E;
    // *_E_prev = E_prev;

    MPI_Barrier(MPI_COMM_WORLD);
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
        printf("Rank%d col%d\t", world_rank, i);
        for (j = 0; j < n; j++)
        {
            printf("%6.3f ", E[i * n + j]);
        }
        printf("\n");
    }
}