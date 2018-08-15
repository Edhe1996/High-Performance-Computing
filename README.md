# High-Performance-Computing
## Deploy MPI and OpenMP parallel computing for d2q9-bgk problem
The cpu.zip is the flat MPI implementation, this code can run about 8.10s on 112 cores for 1024 * 1024 input file.
1024 X 1024
==done==
Reynolds number:		3.376889228821E+00
Elapsed time:			8.134349 (s)
Elapsed user CPU time:		8.102975 (s)
Elapsed system CPU time:	0.146999 (s)

This archive include: d2q9-bgk.c, env.sh, Makefile, job_submit_d2q9-bgk
###########################################################################################
The gpu.zip includes the flat OpenCL implementation and the MPI + OpenCL version.
The flat OpenCL version can run about 4.3s on one GPU for 1024 * 1024 input file.
1024 X 1024
==done==
Reynolds number:		3.376167535782E+00
Elapsed time:			5.876015 (s)
Elapsed user CPU time:		4.323042 (s)
Elapsed system CPU time:	2.906073 (s)

The MPI + OpenCL version can run about 4.45s on 4 GPUs for 1024 * 1024 input file.
1024 X 1024
==done==
Reynolds number:		3.376169204712E+00
Elapsed time:			5.201062 (s)
Elapsed user CPU time:		4.552351 (s)
Elapsed system CPU time:	2.051124 (s)
This archive include two folders, everyone includes: d2q9-bgk.c, env.sh, Makefile, job_submit_d2q9-bgk, kernels.cl
