/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**

** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "mpi.h"

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define MASTER          0

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  int    start;         /* start position */
  int    end;           /* end position */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** restrict cells_ptr, t_speed** restrict tmp_cells_ptr,
               int** restrict obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int accelerate_flow(const t_param params, t_speed* restrict cells, int* restrict obstacles);
float collision(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles, int tot_cells);
int write_values(const t_param params, t_speed* restrict cells, int* restrict obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  t_speed* swap = NULL;
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  int rank;
  int size;
  int tag = 0;
  MPI_Status status;
  MPI_Request request[4];
  int local_nrows;
  int local_ncols;
  int down;
  int up;
  int start;
  int end;
  int num;
  int nrows;
  int rem;
  int recv_up;
  int recv_down;
  float reynolds;


  //float total_vels;
  //int remote_ncols;
  //t_speed *w;             /* local temperature grid at time t     */
  //float *sendbuf;       /* buffer to hold values to send */
  //float *recvbuf1;       /* buffer to hold received values */
  //float *recvbuf2;

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  //MPI initialize
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // printf("size: %d\n", size);

  /* calculate tot_cells value */
  int tot_cells=0;
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
      if (!obstacles[jj * params.nx + ii])
        tot_cells++;
  }

  down = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  up = (rank + 1) % size;

  //local rows and columns for every local grid
  nrows = params.ny / size;       /* integer division */
  rem   = params.ny % size;
  if (rem != 0)
  {  /* if there is a remainder */
    if (rank < rem)
      nrows += 1;  /* redistribute remainder to other ranks */
  }
  local_nrows = nrows;
  local_ncols = params.nx;
  num = local_ncols * NSPEEDS;  /*the number of cells that need to exchange*/
  /* No less than 3 rows for every rank */
  /*if (local_nrows < 3)
  {
    if (rank == size - 1)
    {
      fprintf(stderr,"Error: too many processes:- local_ncols < 3. Use less processes.\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }*/

  //allocate buffer memory
  //sendbuf = (float *)malloc(sizeof(float) * (local_nrows + 2) * num);
  //recvbuf1 = (float *)malloc(sizeof(float) * num);
  //recvbuf2 = (float *)malloc(sizeof(float) * num);

  //calculate the start and the end position for every rank
  if (rem != 0 && rank >= rem)
  {
    start = (local_nrows + 1) * rem + local_nrows * (rank - rem);
    end   = (local_nrows + 1) * rem + local_nrows * (rank + 1 - rem);
  }
  else
  {
    start = local_nrows * rank;
    end   = local_nrows * (rank + 1);
  }
  params.start = start;
  params.end   = end;
  //printf("rank: %d -- %d, %d\n", rank, params.start, params.end);
  /* the position to receive the cell */
  if (start == 0)
  {
    recv_down = (params.ny - 1) * params.nx;
  }
  else
  {
    recv_down = (start - 1) * local_ncols;
  }

  if (end == params.ny)
  {
    recv_up = 0;
  }
  else
  {
    recv_up = (start + local_nrows) * local_ncols;
  }
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* iterate for maxIters timesteps */
  for (int tt = 0; tt < params.maxIters; tt++)
  {

    accelerate_flow(params, cells, obstacles);
    av_vels[tt] = collision(params, cells, tmp_cells, obstacles, tot_cells);

    swap = tmp_cells;
    tmp_cells = cells;
    cells = swap;



    //halo exchange
    /* send to the down, receive from up */
    //memcpy(sendbuf, cells[rank * nrows * local_ncols].speeds, num * sizeof(float));
    /*MPI_Sendrecv(cells + start * local_ncols, num, MPI_FLOAT, down, tag,
           cells + recv_up, num, MPI_FLOAT, up, tag,
           MPI_COMM_WORLD, &status);*/
    MPI_Isend(cells + start * local_ncols, num, MPI_FLOAT, down, tag, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(cells + recv_up, num, MPI_FLOAT, up, tag, MPI_COMM_WORLD, &request[1]);

    /* send to the up, receive from down */
    //memcpy(sendbuf, cells[rank * nrows * local_ncols + (local_nrows - 1) * local_ncols].speeds, num * sizeof(float));
    /*MPI_Sendrecv(cells + start * local_ncols + (local_nrows - 1) * local_ncols, num, MPI_FLOAT, up, tag,
           cells + recv_down, num, MPI_FLOAT, down, tag,
           MPI_COMM_WORLD, &status);*/
    MPI_Isend(cells + start * local_ncols + (local_nrows - 1) * local_ncols, num, MPI_FLOAT, up, tag + 1, MPI_COMM_WORLD, &request[2]);
    MPI_Irecv(cells + recv_down, num, MPI_FLOAT, down, tag + 1, MPI_COMM_WORLD, &request[3]);

    MPI_Waitall(4, request, MPI_STATUS_IGNORE);
#ifdef DEBUG
    if (rank == MASTER)
    {
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", av_vels[tt]);
      printf("tot density: %.12E\n", total_density(params, cells));
    }
#endif
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  //send the cells back to MASTER
  if (rank == MASTER)
  {
    if (rem != 0)
    {
      for (int i = 1; i < size; i++)
      {
        if (i < rem)
        {
          MPI_Recv(cells + i * local_ncols * local_nrows, local_nrows * num, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
        }
        else
        {
          MPI_Recv(cells + rem * local_ncols * local_nrows + (local_nrows - 1) * local_ncols * (i - rem), (local_nrows - 1) * num, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
        }
      }
    }
    else
    {
      for (int i = 1; i < size; i++)
      {
        MPI_Recv(cells + i * local_ncols * local_nrows, local_nrows * num, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
      }
    }

  }
  else
  {
    MPI_Ssend(cells + start * local_ncols, local_nrows * num, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
  }

  /* write final values and free memory */
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  reynolds =  av_vels[params.maxIters - 1] * params.reynolds_dim / viscosity;
  if (rank == MASTER)
  {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", reynolds);
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, cells, obstacles, av_vels);
  }

  MPI_Finalize();
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* restrict cells, int* restrict obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;
  for (int ii = 0; ii < params.nx; ii++)
  {
    int index = ii + jj*params.nx;
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[index]
        && (cells[index].speeds[3] - w1) > 0.f
        && (cells[index].speeds[6] - w2) > 0.f
        && (cells[index].speeds[7] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells[index].speeds[1] += w1;
      cells[index].speeds[5] += w2;
      cells[index].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[index].speeds[3] -= w1;
      cells[index].speeds[6] -= w2;
      cells[index].speeds[7] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

//rebound, collision and av_velocity
float collision(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles, int tot_cells)
{
  //const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  // const1 = 1 / c_sq, const2 = 1 / (2 * c_sq), const3 = 1 / (2 * c_sq * c_sq)
  const float const1 = 3.0f, const2 = 1.5f, const3 = 4.5f;

  //int    tot_cells = 0;  /* no. of cells used in calculation */
  //int    local_cells;
  float tot_u;          /* accumulated magnitudes of velocity for each cell */
  float local_u;

  float tmp[NSPEEDS];
  //int tag = 0;
  //MPI_Status status;

  /* initialise */
  tot_u = 0.f;

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for (int jj = params.start; jj < params.end; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int index = ii + jj*params.nx;
      //int index = ii + jj*params.nx;
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp[0] = cells[ii + jj*params.nx].speeds[0]; /* central cell, no movement */
      tmp[1] = cells[x_w + jj*params.nx].speeds[1]; /* east */
      tmp[2] = cells[ii + y_s*params.nx].speeds[2]; /* north */
      tmp[3] = cells[x_e + jj*params.nx].speeds[3]; /* west */
      tmp[4] = cells[ii + y_n*params.nx].speeds[4]; /* south */
      tmp[5] = cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
      tmp[6] = cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
      tmp[7] = cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
      tmp[8] = cells[x_w + y_n*params.nx].speeds[8]; /* south-east */

      /* don't consider occupied cells */
      if (obstacles[index])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells[index].speeds[1] = tmp[3];
        tmp_cells[index].speeds[2] = tmp[4];
        tmp_cells[index].speeds[3] = tmp[1];
        tmp_cells[index].speeds[4] = tmp[2];
        tmp_cells[index].speeds[5] = tmp[7];
        tmp_cells[index].speeds[6] = tmp[8];
        tmp_cells[index].speeds[7] = tmp[5];
        tmp_cells[index].speeds[8] = tmp[6];
      }
      else
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp[1]
                      + tmp[5]
                      + tmp[8]
                      - (tmp[3]
                         + tmp[6]
                         + tmp[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp[2]
                      + tmp[5]
                      + tmp[6]
                      - (tmp[4]
                         + tmp[7]
                         + tmp[8]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        float w11=w1 * local_density, w22= w2 * local_density, u_sq1= 1.0f - u_sq*const2;
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (u_sq1);
        /* axis speeds: weight w1 */
        d_equ[1] = w11 * (u[1] * const1
                                         + (u[1] * u[1]) * const3
                                         + u_sq1);
        d_equ[2] = w11 * (u[2] * const1
                                         + (u[2] * u[2]) * const3
                                         + u_sq1);
        d_equ[3] = w11 * (u[3] * const1
                                         + (u[3] * u[3]) * const3
                                         + u_sq1);
        d_equ[4] = w11 * (u[4] * const1
                                         + (u[4] * u[4]) * const3
                                         + u_sq1);
        /* diagonal speeds: weight w2 */
        d_equ[5] = w22 * (u[5] * const1
                                         + (u[5] * u[5]) * const3
                                         + u_sq1);
        d_equ[6] = w22 * (u[6] * const1
                                         + (u[6] * u[6]) * const3
                                         + u_sq1);
        d_equ[7] = w22 * (u[7] * const1
                                         + (u[7] * u[7]) * const3
                                         + u_sq1);
        d_equ[8] = w22 * (u[8] * const1
                                         + (u[8] * u[8]) * const3
                                         + u_sq1);

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          tmp_cells[index].speeds[kk] = tmp[kk]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp[kk]);
        }

        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[index].speeds[kk];
        }

        /* x-component of velocity */
        u_x = (tmp_cells[index].speeds[1]
                      + tmp_cells[index].speeds[5]
                      + tmp_cells[index].speeds[8]
                      - (tmp_cells[index].speeds[3]
                         + tmp_cells[index].speeds[6]
                         + tmp_cells[index].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        u_y = (tmp_cells[index].speeds[2]
                      + tmp_cells[index].speeds[5]
                      + tmp_cells[index].speeds[6]
                      - (tmp_cells[index].speeds[4]
                         + tmp_cells[index].speeds[7]
                         + tmp_cells[index].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        //++tot_cells;

      }
    }
  }

  MPI_Reduce(&tot_u, &local_u, 1, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
  //MPI_Reduce(&tot_cells, &local_cells, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
  return local_u / (float)tot_cells;
  //return EXIT_SUCCESS;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** restrict cells_ptr, t_speed** restrict tmp_cells_ptr,
               int** restrict obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int index = ii + jj*params.nx;
      /* an occupied cell */
      if (obstacles[index])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[index].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[index].speeds[1]
               + cells[index].speeds[5]
               + cells[index].speeds[8]
               - (cells[index].speeds[3]
                  + cells[index].speeds[6]
                  + cells[index].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[index].speeds[2]
               + cells[index].speeds[5]
               + cells[index].speeds[6]
               - (cells[index].speeds[4]
                  + cells[index].speeds[7]
                  + cells[index].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
