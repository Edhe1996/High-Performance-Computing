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
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<sys/time.h>
#include<sys/resource.h>
#include "mpi.h"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define NSPEEDS         9
#define LOCAL_X			    16
#define LOCAL_Y			    16
#define MAX_DEVICES     32
#define MAX_DEVICE_NAME 1024
#define MASTER          0
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define OCLFILE         "kernels.cl"

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

/* struct to hold OpenCL variables */
typedef struct
{
  cl_device_id      device;
  cl_context        context;
  cl_command_queue  queue;

  cl_program program;
  cl_kernel  accelerate_flow;
  cl_kernel PRC;

  cl_mem swap;
  cl_mem tmp_cells;
  cl_mem cells;
  cl_mem obstacles;
  cl_mem av_vels;
 } t_ocl;



/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl* ocl, int rank, int size);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/

int accelerate_flow(const t_param params, t_ocl ocl, float w1, float w2, int local_nrows, int local_ncols);
int PRC(const t_param params, int tot_cells, t_ocl ocl, int local_nrows, int local_ncols, int start);
int write_values(const t_param params, float* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl);
/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float* cells);
/* calculate Reynolds number */
float calc_reynolds(const t_param params, float av_vels);

/* utility functions */
void checkError(cl_int err, const char *op, const int line);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/* select an OpenCL device from all available devices */
cl_device_id DeviceInfo(int rank);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_ocl    ocl;                 /* struct to hold OpenCL objects */
  float* cells     = NULL;    /* grid containing fluid densities */
  float* tmp_cells = NULL;
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  cl_int err;
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */
  int ii, jj;

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
  //float* sendbuf;
  //float* av_buf;

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

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &ocl, rank, size);

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

  //number of work groups in OpenCL device
  int NumWorkGroups = (local_nrows * local_ncols)/ (LOCAL_X * LOCAL_Y);
  //printf("Number of work-groups first: %d\n", NumWorkGroups);
  /* initialise our data structures and load values from files, including kernels file */

  int tot_cells=0;
  for (ii=0; ii<params.ny; ii++)
  {
	for (jj=0; jj<params.nx; jj++)
		if (!obstacles[ii*params.nx +jj])
		  tot_cells++;
  }

  down = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  up = (rank + 1) % size;

  num = local_ncols * NSPEEDS;  /*the number of cells that need to exchange*/

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
    recv_down = (params.ny - 1) * params.nx * NSPEEDS;
  }
  else
  {
    recv_down = (start - 1) * local_ncols * NSPEEDS;
  }

  if (end == params.ny)
  {
    recv_up = 0;
  }
  else
  {
    recv_up = (start + local_nrows) * local_ncols * NSPEEDS;
  }

  //sendbuf = (float *)malloc(sizeof(float) * local_nrows * local_ncols * NSPEEDS);
  //memcpy(sendbuf, cells + start * NSPEEDS, local_nrows * local_ncols * NSPEEDS * sizeof(float));
 // Write cells, tmp_cells and obstacles to OpenCL device
  err = clEnqueueWriteBuffer(ocl.queue, ocl.cells, CL_TRUE, 0,
                             sizeof(cl_float) * NSPEEDS * params.nx * params.ny,
                             cells, 0, NULL, NULL);
  checkError(err, "writing cells data", __LINE__);
  //printf("Get here!");
  //memcpy(sendbuf, tmp_cells + start * NSPEEDS, local_nrows * local_ncols * NSPEEDS * sizeof(float));
  err = clEnqueueWriteBuffer(ocl.queue, ocl.tmp_cells, CL_TRUE, 0,
                             sizeof(cl_float) * NSPEEDS * params.nx * params.ny,
                             tmp_cells, 0, NULL, NULL);
  checkError(err, "writing tmp_cells data", __LINE__);

  err = clEnqueueWriteBuffer(ocl.queue, ocl.obstacles, CL_TRUE, 0,
                             sizeof(cl_int) * params.nx * params.ny,
                             obstacles, 0, NULL, NULL);
  checkError(err, "writing obstacles data", __LINE__);

  err = clEnqueueWriteBuffer(ocl.queue, ocl.av_vels, CL_TRUE, 0,
                             sizeof(cl_float) * NumWorkGroups * params.maxIters,
                             av_vels, 0, NULL, NULL);
  checkError(err, "writing av_vels data", __LINE__);


  const float w1 = params.density * params.accel / 9.0f;
  const float w2 = params.density * params.accel / 36.0f;

  // set part of the parameters in kernel functions
  accelerate_flow(params, ocl, w1, w2, local_nrows, local_ncols);
  //printf("Rank: %d, Start: %d\n", rank, start);
  PRC(params, tot_cells, ocl, local_nrows, local_ncols, start);

  err = clEnqueueWriteBuffer(ocl.queue, ocl.cells, CL_TRUE, 0,
                             sizeof(cl_float) * NSPEEDS * params.nx * params.ny,
                             cells, 0, NULL, NULL);

  size_t global_size_af = params.nx; //global size for accelerate flow
  size_t global_size[2] = {local_ncols, local_nrows};   //global size for propagate, rebound and collision
  size_t local_size[2] = {LOCAL_X, LOCAL_Y};    //local size for propagate, rebound and collision

  /* iterate for maxIters timesteps */
  int send1 = start * local_ncols * NSPEEDS;
  int send2 = (start * local_ncols + (local_nrows - 1) * local_ncols) * NSPEEDS;

  float* send_buffer1 = NULL;
  float* send_buffer2 = NULL;

  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  for (int tt = 0; tt < params.maxIters; tt++)
  {
    //halo exchange
    /* send to the down, receive from up */
    //memcpy(sendbuf, cells[rank * nrows * local_ncols].speeds, num * sizeof(float));
    /*MPI_Sendrecv(cells + start * local_ncols * NSPEEDS, num, MPI_FLOAT, down, tag,
           cells + recv_up, num, MPI_FLOAT, up, tag,
           MPI_COMM_WORLD, &status);*/

    /* send to the up, receive from down */
    //memcpy(sendbuf, cells[rank * nrows * local_ncols + (local_nrows - 1) * local_ncols].speeds, num * sizeof(float));
    /*MPI_Sendrecv(cells + (start * local_ncols + (local_nrows - 1) * local_ncols) * NSPEEDS, num, MPI_FLOAT, up, tag,
           cells + recv_down, num, MPI_FLOAT, down, tag,
           MPI_COMM_WORLD, &status);*/
    //printf(" Before Rank %d Get here!\n", rank);
    //cells = (float *)clEnqueueMapBuffer(ocl.queue, ocl.cells, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * NSPEEDS * params.nx * params.ny, 0, NULL, NULL, NULL);
    //printf("Rank %d Get here!\n", rank);
    //checkError(err, "writing cells data", __LINE__);

    /*err = clEnqueueWriteBuffer(ocl.queue, ocl.tmp_cells, CL_TRUE, 0,
                               sizeof(cl_float) * NSPEEDS * params.nx * params.ny,
                               tmp_cells, 0, NULL, NULL);
    checkError(err, "writing tmp_cells data", __LINE__);*/

	  //set part of the accelerate_flow kernel arguments that change for every iteration
	  err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.cells);
	  //checkError(err, "setting accelerate_flow argument 0", __LINE__);

    //enqueue the kernel for execution
	  err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow, 1, NULL, &global_size_af, NULL, 0, NULL, NULL);
	  //checkError(err, "enqueueing accelerate_flow kernel", __LINE__);

    /*err = clFlush(ocl.queue);
    checkError(err, "waiting for kernel", __LINE__);*/

	  //set part of the propagate_rebound_collision (PRC) kernel arguments that change for every iteration
    err = clSetKernelArg(ocl.PRC, 0, sizeof(cl_mem), &ocl.cells);
	  //checkError(err, "setting PRC argument 0", __LINE__);

    err = clSetKernelArg(ocl.PRC, 1, sizeof(cl_mem), &ocl.tmp_cells);
	  //checkError(err, "setting PRC argument 1", __LINE__);

	  err = clSetKernelArg(ocl.PRC, 8, sizeof(cl_int), &tt);
	  //checkError(err, "setting PRC argument 8", __LINE__);

    //enqueue the kernel for execution
	  err = clEnqueueNDRangeKernel(ocl.queue, ocl.PRC, 2, NULL, global_size, local_size, 0, NULL, NULL);
	  //checkError(err, "enqueueing PRC kernel", __LINE__);

    //err = clFinish(ocl.queue);
    //err = clFlush(ocl.queue);
    //checkError(err, "waiting for kernel", __LINE__);

	  //swap the pointers
    ocl.swap=ocl.tmp_cells;
    ocl.tmp_cells=ocl.cells;
    ocl.cells=ocl.swap;
    /* #ifdef DEBUG
        printf("==timestep: %d==\n", tt);
        printf("av velocity: %.12E\n", av_vels[tt]);
        printf("tot density: %.12E\n", total_density(params, cells));
    #endif */

    // Read back cells from openCL device
    /*err = clEnqueueReadBuffer(ocl.queue, ocl.cells, CL_TRUE, 0,
                              sizeof(float) * NSPEEDS * params.nx * params.ny,
                              cells, 0, NULL, NULL);*/
    //checkError(err, "reading back cells", __LINE__);
    //
    send_buffer1 = (float *)clEnqueueMapBuffer(ocl.queue, ocl.cells, CL_TRUE, CL_MAP_READ, sizeof(cl_float) * send1, sizeof(float) * num, 0, NULL, NULL, NULL);
    clEnqueueUnmapMemObject(ocl.queue, ocl.cells, send_buffer1, 0, NULL, NULL);
    send_buffer2 = (float *)clEnqueueMapBuffer(ocl.queue, ocl.cells, CL_TRUE, CL_MAP_READ, sizeof(cl_float) * send2, sizeof(float) * num, 0, NULL, NULL, NULL);
    clEnqueueUnmapMemObject(ocl.queue, ocl.cells, send_buffer2, 0, NULL, NULL);
    /*err = clEnqueueReadBuffer(ocl.queue, ocl.cells, CL_TRUE, sizeof(cl_float) * send1,
                              sizeof(float) * num,
                              cells + send1, 0, NULL, NULL);
    err = clEnqueueReadBuffer(ocl.queue, ocl.cells, CL_TRUE, sizeof(cl_float) * send2,
                              sizeof(float) * num,
                              cells + send2, 0, NULL, NULL);*/
    MPI_Isend(send_buffer1, num, MPI_FLOAT, down, tag, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(cells + recv_up, num, MPI_FLOAT, up, tag, MPI_COMM_WORLD, &request[1]);

    MPI_Isend(send_buffer2, num, MPI_FLOAT, up, tag + 1, MPI_COMM_WORLD, &request[2]);
    MPI_Irecv(cells + recv_down, num, MPI_FLOAT, down, tag + 1, MPI_COMM_WORLD, &request[3]);

    MPI_Waitall(4, request, MPI_STATUS_IGNORE);


    //err = clFinish(ocl.queue);
    /*write_cells = (float *)clEnqueueMapBuffer(ocl.queue, ocl.cells, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(float) * NSPEEDS * params.nx * params.ny, 0, NULL, NULL, NULL);
    memcpy(write_cells, send_cells, sizeof(float) * NSPEEDS * params.nx * params.ny);
    clEnqueueUnmapMemObject(ocl.queue, ocl.cells, write_cells, 0, NULL, NULL);
    err = clFinish(ocl.queue);*/
    err = clEnqueueWriteBuffer(ocl.queue, ocl.cells, CL_TRUE, sizeof(cl_float) * recv_up,
                               sizeof(cl_float) * num,
                               cells + recv_up, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(ocl.queue, ocl.cells, CL_TRUE, sizeof(cl_float) * recv_down,
                               sizeof(cl_float) * num,
                               cells + recv_down, 0, NULL, NULL);
    /*err = clEnqueueWriteBuffer(ocl.queue, ocl.cells, CL_TRUE, 0,
                               sizeof(cl_float) * NSPEEDS * params.nx * params.ny,
                               send_cells, 0, NULL, NULL);*/

  }
  err = clEnqueueReadBuffer(ocl.queue, ocl.cells, CL_TRUE, 0,
                            sizeof(float) * NSPEEDS * params.nx * params.ny,
                            cells, 0, NULL, NULL);


  /*printf("Rank: %d\n", rank);
  for(jj = 8448; jj < 8498; jj++)
  {
    printf("%f\n", cells[jj * NSPEEDS]);
  }*/

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  //read back av_vels from OpenCL device
  err = clEnqueueReadBuffer(ocl.queue, ocl.av_vels, CL_TRUE, 0,
                            sizeof(float) *  NumWorkGroups * params.maxIters,
                            av_vels, 0, NULL, NULL);
  checkError(err, "reading back av_vels", __LINE__);

  // Calculate the final av_vels[tt]
  float* av_vels_final;
  float* av_vels_reduction;
  //use calloc to set all variables to a zero value
  av_vels_final=calloc(params.maxIters, sizeof(float));  //reduction for every work-group
  av_vels_reduction=calloc(params.maxIters, sizeof(float)); // reduction for every rank

  for (int tt=0; tt< params.maxIters; tt++)
  {
	  for (int ii=0; ii < NumWorkGroups; ii++)
	  {
		  av_vels_final[tt]+= av_vels[tt * NumWorkGroups + ii];
      //av_vels_reduction[tt]+= av_vels[tt * NumWorkGroups + ii];
	  }
    av_vels_reduction[tt] = av_vels_final[tt];
    MPI_Reduce(&(av_vels_final[tt]), &(av_vels_reduction[tt]), 1, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
	  av_vels_reduction[tt] = av_vels_reduction[tt]/(float)tot_cells;
  }

  /* printf("Rank: %d\n", rank);
  for(jj = 128; jj < 200; jj++)
  {
    printf("%f\n", cells[jj * NSPEEDS]);
  } */

  //printf("Host rank: %d, tt: %d\n", rank, tt);
  if (rank == MASTER)
  {
    if (rem != 0)
    {
      for (int i = 1; i < size; i++)
      {
        if (i < rem)
        {
          MPI_Recv(cells + i * local_ncols * local_nrows * NSPEEDS, local_nrows * num, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
        }
        else
        {
          MPI_Recv(cells + (rem * local_ncols * local_nrows + (local_nrows - 1) * local_ncols * (i - rem)) * NSPEEDS, (local_nrows - 1) * num, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
        }
      }
    }
    else
    {
      for (int i = 1; i < size; i++)
      {
        MPI_Recv(cells + i * local_ncols * local_nrows * NSPEEDS, local_nrows * num, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
      }
    }

  }
  else
  {
    MPI_Ssend(cells + start * local_ncols * NSPEEDS, local_nrows * num, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
  }

  reynolds = av_vels_reduction[params.maxIters-1];
  /* write final values and free memory */
  if (rank == MASTER)
  {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, reynolds));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, cells, obstacles, av_vels_reduction);
  }
  free(av_vels_final);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels_reduction, ocl);
  MPI_Finalize();

  return EXIT_SUCCESS;
}


int accelerate_flow(const t_param params, t_ocl ocl, float w1, float w2, int local_nrows, int local_ncols)
{
  cl_int err;

  //set part of the kernel arguments that do not change for every iteration
  err = clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting accelerate_flow argument 1", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_int), &params.nx);
  checkError(err, "setting accelerate_flow argument 2", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_int), &params.ny);
  checkError(err, "setting accelerate_flow argument 3", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 4, sizeof(cl_float), &w1);
  checkError(err, "setting accelerate_flow argument 4", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 5, sizeof(cl_float), &w2);
  checkError(err, "setting accelerate_flow argument 5", __LINE__);

  return EXIT_SUCCESS;
}



int PRC(const t_param params, int tot_cells, t_ocl ocl, int local_nrows, int local_ncols, int start)
{
  cl_int err;

  //set part of the kernel arguments that do not change for every iteration
  err = clSetKernelArg(ocl.PRC, 2, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting PRC argument 2", __LINE__);
  err = clSetKernelArg(ocl.PRC, 3, sizeof(cl_mem), &ocl.av_vels);
  checkError(err, "setting PRC argument 3", __LINE__);
  err = clSetKernelArg(ocl.PRC, 4, sizeof(cl_int), &params.nx);
  checkError(err, "setting PRC argument 4", __LINE__);
  err = clSetKernelArg(ocl.PRC, 5, sizeof(cl_int), &params.ny);
  checkError(err, "setting PRC argument 5", __LINE__);
  err = clSetKernelArg(ocl.PRC, 6, sizeof(cl_int), &tot_cells);
  checkError(err, "setting PRC argument 6", __LINE__);
  err = clSetKernelArg(ocl.PRC, 7, sizeof(cl_float), &params.omega);
  checkError(err, "setting PRC argument 7", __LINE__);
  err = clSetKernelArg(ocl.PRC, 9, sizeof(cl_float) * LOCAL_X * LOCAL_Y, NULL);
  checkError(err, "setting PRC argument 9", __LINE__);
  err = clSetKernelArg(ocl.PRC, 10, sizeof(cl_int), &start);
  checkError(err, "setting PRC argument 10", __LINE__);

  return EXIT_SUCCESS;
}



int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl *ocl, int rank, int size)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  char*  ocl_src;        /* OpenCL kernel source */
  long   ocl_size;       /* size of OpenCL kernel source */

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
  *cells_ptr = (float*)malloc(sizeof(float) * NSPEEDS * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

   *tmp_cells_ptr = (float*)malloc(sizeof(float) * NSPEEDS * (params->ny * params->nx));

   if (*tmp_cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

    /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.0 / 9.0;
  float w1 = params->density      / 9.0;
  float w2 = params->density      / 36.0;

  //int grid_size = params->nx * params->ny;

  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      /* centre */
      (*cells_ptr)[0 + (ii * params->nx + jj) * NSPEEDS] = w0;
      /* axis directions */
      (*cells_ptr)[1 + (ii * params->nx + jj) * NSPEEDS] = w1;
      (*cells_ptr)[2 + (ii * params->nx + jj) * NSPEEDS] = w1;
      (*cells_ptr)[3 + (ii * params->nx + jj) * NSPEEDS] = w1;
      (*cells_ptr)[4 + (ii * params->nx + jj) * NSPEEDS] = w1;
      /* diagonals */
      (*cells_ptr)[5 + (ii * params->nx + jj) * NSPEEDS] = w2;
      (*cells_ptr)[6 + (ii * params->nx + jj) * NSPEEDS] = w2;
      (*cells_ptr)[7 + (ii * params->nx + jj) * NSPEEDS] = w2;
      (*cells_ptr)[8 + (ii * params->nx + jj) * NSPEEDS] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      (*obstacles_ptr)[ii * params->nx + jj] = 0;
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
    (*obstacles_ptr)[yy * params->nx + xx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  int nrows = params->ny / size;       /* integer division */
  int rem   = params->ny % size;
  if (rem != 0)
  {  /* if there is a remainder */
    if (rank < rem)
      nrows += 1;  /* redistribute remainder to other ranks */
  }
  int local_nrows = nrows;
  int local_ncols = params->nx;

  int av_size = ((local_nrows * local_ncols) / (LOCAL_X * LOCAL_Y)) * params->maxIters;
  //printf("Number of work-groups second: %d\n", (local_nrows * local_ncols) / (LOCAL_X * LOCAL_Y));
  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * av_size);


  cl_int err;

  ocl->device = DeviceInfo(rank);

  // Create OpenCL context
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCLFILE, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
    die(message, __LINE__, __FILE__);
  }

  // Create OpenCL command queue
  ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
  checkError(err, "creating command queue", __LINE__);
  //printf("Rank %d queue %u\n", rank, &(ocl->queue));
  // Load OpenCL kernel source
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char*)malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  // Create OpenCL program
  ocl->program = clCreateProgramWithSource(
    ocl->context, 1, (const char**)&ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);

  // Build OpenCL program
  err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t sz;
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    char *buildlog = malloc(sz);
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  // Create OpenCL kernels
  ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
  checkError(err, "creating accelerate_flow kernel", __LINE__);

  ocl->PRC = clCreateKernel(ocl->program, "PRC", &err);
  checkError(err, "creating PRC kernel", __LINE__);
  //printf("Rank %d kernel %u\n", rank, &(ocl->PRC));

  // Allocate OpenCL buffers
  ocl->cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(float) * NSPEEDS * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);

   ocl->tmp_cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(float) * NSPEEDS * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);

  ocl->obstacles = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(int) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating obstacles buffer", __LINE__);

  ocl->av_vels = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(float) * av_size, NULL, &err);
  checkError(err, "creating av_vels buffer", __LINE__);



  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl)
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

  clReleaseMemObject(ocl.cells);
  clReleaseMemObject(ocl.tmp_cells);
  clReleaseMemObject(ocl.obstacles);
  clReleaseKernel(ocl.accelerate_flow);
  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float av_vels)
{
  const float viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
  /* compute Reynold's number by using the last average velocity */
  return av_vels * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, float* cells)
{
  float total = 0.0;  /* accumulator */
  //int grid_size = params.nx * params.ny;

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[kk + (ii * params.nx + jj) * NSPEEDS];
      }
    }
  }

  return total;
}

int write_values(const t_param params, float* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.0 / 3.0; /* sq. of speed of sound */
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

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* an occupied cell */
      if (obstacles[ii * params.nx + jj])
      {
        u_x = u_y = u = 0.0;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
		//int grid_size = params.nx * params.ny;
        local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[kk +(ii * params.nx + jj) * NSPEEDS];
        }

        /* compute x velocity component */
        u_x = (cells[1 +(ii * params.nx + jj) * NSPEEDS]
               + cells[5 +(ii * params.nx + jj) * NSPEEDS]
               + cells[8 +(ii * params.nx + jj) * NSPEEDS]
               - (cells[3 +(ii * params.nx + jj) * NSPEEDS]
                  + cells[6 +(ii * params.nx + jj) * NSPEEDS]
                  + cells[7 +(ii * params.nx + jj) * NSPEEDS]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[2 +(ii * params.nx + jj) * NSPEEDS]
               + cells[5 +(ii * params.nx + jj) * NSPEEDS]
               + cells[6 +(ii * params.nx + jj) * NSPEEDS]
               - (cells[4 +(ii * params.nx + jj) * NSPEEDS]
                  + cells[7 +(ii * params.nx + jj) * NSPEEDS]
                  + cells[8 +(ii * params.nx + jj) * NSPEEDS]))
              / local_density;
        /* compute norm of velocity */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
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

void checkError(cl_int err, const char *op, const int line)
{
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
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

cl_device_id DeviceInfo(int rank)
{
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[8];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];

  // Get list of platforms
  err = clGetPlatformIDs(8, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);

  // Get list of devices
  for (cl_uint p = 0; p < num_platforms; p++)
  {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-total_devices, devices+total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }

  // Print list of devices
  /*printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; d++)
  {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("%2d: %s\n", d, name);
  }
  printf("\n");*/

  // Use second device unless OCL_DEVICE environment variable used
  cl_uint device_index = (rank + 1) % 2 + 1;
  char *dev_env = getenv("OCL_DEVICE");
  if (dev_env)
  {
    char *end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end))
      die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  /* check if index is wrong */
  if (device_index >= total_devices)
  {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  // Print selected OpenCL device information
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
                  MAX_DEVICE_NAME, name, NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

  return devices[device_index];
}
