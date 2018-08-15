/* support for double floating-point precision */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

kernel void accelerate_flow(global float* cells,
                            global unsigned int* obstacles,
                            const unsigned int nx, const unsigned int ny,
                            const float w1,
							              const float w2
							             )
{
  /* modify the 2nd row of the grid */
  const unsigned int ii = ny - 2;

  const unsigned int gridsize = nx * ny;

  /* get global index */
  const unsigned int jj = get_global_id(0);

  const unsigned int id = ii * nx + jj;

  /* accelerate flow without if-statement */
  float mask = (((!obstacles[id])
	&& (cells[3 + id * NSPEEDS] - w1) > 0.0f
	&& (cells[6 + id * NSPEEDS] - w2) > 0.0f
	&& (cells[7 + id * NSPEEDS] - w2) > 0.0f )? 1.0f : 0.0f );

  cells[1 + id * NSPEEDS]+= mask * w1;
  cells[5 + id * NSPEEDS]+= mask * w2;
  cells[8 + id * NSPEEDS]+= mask * w2;

  cells[3 + id * NSPEEDS]-= mask * w1;
  cells[6 + id * NSPEEDS]-= mask * w2;
  cells[7 + id * NSPEEDS]-= mask * w2;
}


kernel void PRC(global float* cells,
					            global float* tmp_cells,
					            global unsigned int* obstacles,
					            global float* av_vels,
					            unsigned int nx,
					            unsigned int ny,
					            unsigned int tot_cells,
					            float omega,
					            unsigned int tt,
					            local float* tot_u,
                      unsigned int start
					           )
{
  //const float c_sq = 1.0f / 3.0f; /* square of speed of sound */
  const float w0 = 4.0f / 9.0f;  /* weighting factor */
  const float w1 = 1.0f / 9.0f;  /* weighting factor */
  const float w2 = 1.0f / 36.0f; /* weighting factor */

  /* compute all the intensive float precisions and store them in const variables*/
  const float const1 = 3.0f, const2 = 1.5f , const3 = 4.5f ;

  int gridsize= nx * ny;
  //int id = get_global_id(0);
  int x = get_global_id(0);
  int y = get_global_id(1);
  y = y + start;
  int id = y * nx + x;

  int lx = get_local_id(0);
  int ly = get_local_id(1);
  //int lid = get_local_id(0);
  int lid = ly * get_local_size(0) + lx;

  private float temp[NSPEEDS];

  int offset_id = y * nx + x;
  /* propogate */

  int y_n = (y + 1) % ny;
  int x_e = (x + 1) % nx;
  int y_s = (y == 0) ? (y + ny - 1) : (y - 1);
  int x_w = (x == 0) ? (x + nx - 1) : (x - 1);
  /*if ((offset_id < 2) && (tt < 5))
  {
    printf("Start; %d \n", start);
    printf("GPU before\n");
    printf("%f\n", cells[0 + offset_id * NSPEEDS]);
  }*/
  temp[0] = cells[0 + offset_id * NSPEEDS];
  temp[1] = cells[1 + (y * nx + x_w) * NSPEEDS]; /* east */
  temp[2] = cells[2 + (y_s * nx + x) * NSPEEDS]; /* north */
  temp[3] = cells[3 + (y * nx + x_e) * NSPEEDS]; /* west */
  temp[4] = cells[4 + (y_n * nx + x) * NSPEEDS]; /* south */
  temp[5] = cells[5 + (y_s * nx + x_w) * NSPEEDS]; /* north-east */
  temp[6] = cells[6 + (y_s * nx + x_e) * NSPEEDS]; /* north-west */
  temp[7] = cells[7 + (y_n * nx + x_e) * NSPEEDS]; /* south-west */
  temp[8] = cells[8 + (y_n * nx + x_w) * NSPEEDS]; /* south-east */

  //barrier(CLK_LOCAL_MEM_FENCE);

  /* rebound */
  if (obstacles[offset_id])
  {
    /* called after propagate, so taking values from scratch space
    ** mirroring, and writing into main grid */
		tmp_cells[1 + offset_id * NSPEEDS]=temp[3];
    tmp_cells[2 + offset_id * NSPEEDS]=temp[4];
    tmp_cells[3 + offset_id * NSPEEDS]=temp[1];
    tmp_cells[4 + offset_id * NSPEEDS]=temp[2];
    tmp_cells[5 + offset_id * NSPEEDS]=temp[7];
    tmp_cells[6 + offset_id * NSPEEDS]=temp[8];
    tmp_cells[7 + offset_id * NSPEEDS]=temp[5];
    tmp_cells[8 + offset_id * NSPEEDS]=temp[6];

    tot_u[lid] = 0.0f;

  }

  /* collision */
  else
  {
    /* compute local density total */
    float local_density = 0.0f;

		#pragma unroll
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      local_density += temp[kk];
    }
		/*compute the inverse of local density*/
		float loc1= 1 / local_density;

    /* compute x velocity component */
    float u_x = (temp[1]
                      + temp[5]
                      + temp[8]
                      - (temp[3]
                         + temp[6]
                         + temp[7]))
                     * loc1;

    /* compute y velocity component */
    float u_y = (temp[2]
                      + temp[5]
                      + temp[6]
                      - (temp[4]
                         + temp[7]
                         + temp[8]))
                     *loc1;

		float u_sq = pow(u_x,2) + pow(u_y,2);

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

    /* equilibrium densities, weights and u_sq1 coefficient */
    float d_equ[NSPEEDS], w11=w1 * local_density, w22= w2 * local_density, u_sq1= 1.0f - u_sq*const2;

		/* zero velocity density: weight w0 */
    d_equ[0] = w0 * local_density * (u_sq1);
    /* axis speeds: weight w1 */
		d_equ[1] = w11 * (u[1]*const1 + pow(u[1],2)*const3 + u_sq1);
    d_equ[2] = w11 * (u[2]*const1 + pow(u[2],2)*const3 + u_sq1);
    d_equ[3] = w11 * (u[3]*const1 + pow(u[3],2)*const3 + u_sq1);
    d_equ[4] = w11 * (u[4]*const1 + pow(u[4],2)*const3 + u_sq1);
    /* diagonal speeds: weight w2 */
    d_equ[5] = w22 * (u[5]*const1 + pow(u[5],2)*const3 + u_sq1);
    d_equ[6] = w22 * (u[6]*const1 + pow(u[6],2)*const3 + u_sq1);
    d_equ[7] = w22 * (u[7]*const1 + pow(u[7],2)*const3 + u_sq1);
    d_equ[8] = w22 * (u[8]*const1 + pow(u[8],2)*const3 + u_sq1);

    /* relaxation step */
		#pragma unroll
    for (int kk = 0; kk < NSPEEDS; kk++)
      tmp_cells[kk + offset_id * NSPEEDS] = temp[kk] + omega * (d_equ[kk] - temp[kk]);
    /* if ((tt < 3) && (offset_id < 200))
    {
        printf("tmp_cells: %f\n", tmp_cells[0 + offset_id * NSPEEDS]);
    } */
		/* velocity squared */
    tot_u[lid] = sqrt(u_sq);


  }

  /* if ((offset_id < 2) && (tt < 5))
  {
    printf("GPU end\n");
    printf("%f\n", tmp_cells[0 + offset_id * NSPEEDS]);
  } */

  barrier(CLK_LOCAL_MEM_FENCE);
   // see stackoverflow.com/questions/20613013/opencl-float-sum-reduction
  /* av_vel */
  int local_size = get_local_size(0) * get_local_size(1);
  int num_group = get_num_groups(0) * get_num_groups(1);
  int group_id   = get_group_id(1) * get_num_groups(0) + get_group_id(0);

  //add them all to the first work-item
	for (unsigned int mm = local_size / 2; mm > 0; mm >>= 1)
	{
		if (lid < mm)
		{
		  tot_u[lid] += tot_u[lid + mm];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (lid==0)
	{
      av_vels[tt * num_group + group_id]=tot_u[0];
	}
}
