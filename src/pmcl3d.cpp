/**
@section LICENSE
Copyright (c) 2013-2016, Regents of the University of California
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "pmcl3d.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "kernel.h"

constexpr double MICRO_SECOND = 1.0e-6;

// Calculates recording points for each core
// rec_nbgxyz rec_nedxyz...
// WARNING: Assumes NPZ = 1! Only surface outputs are needed!
void calcRecordingPoints(int* rec_nbgx, int* rec_nedx, int* rec_nbgy, int* rec_nedy, int* rec_nbgz, int* rec_nedz, int* rec_nxt, int* rec_nyt, int* rec_nzt, MPI_Offset* displacement, long int nxt, long int nyt, long int nzt, int rec_NX, int rec_NY, int rec_NZ, int NBGX, int NEDX, int NSKPX, int NBGY, int NEDY, int NSKPY, int NBGZ, int NEDZ, int NSKPZ, int* coord) {
  *displacement = 0;

  if (NBGX > nxt * (coord[0] + 1))
    *rec_nxt = 0;
  else if (NEDX < nxt * coord[0] + 1)
    *rec_nxt = 0;
  else {
    if (nxt * coord[0] >= NBGX) {
      *rec_nbgx = (nxt * coord[0] + NBGX - 1) % NSKPX;
      *displacement += (nxt * coord[0] - NBGX) / NSKPX + 1;
    } else
      *rec_nbgx = NBGX - nxt * coord[0] - 1;  // since rec_nbgx is 0-based
    if (nxt * (coord[0] + 1) <= NEDX)
      *rec_nedx = (nxt * (coord[0] + 1) + NBGX - 1) % NSKPX - NSKPX + nxt;
    else
      *rec_nedx = NEDX - nxt * coord[0] - 1;
    *rec_nxt = (*rec_nedx - *rec_nbgx) / NSKPX + 1;
  }

  if (NBGY > nyt * (coord[1] + 1))
    *rec_nyt = 0;
  else if (NEDY < nyt * coord[1] + 1)
    *rec_nyt = 0;
  else {
    if (nyt * coord[1] >= NBGY) {
      *rec_nbgy = (nyt * coord[1] + NBGY - 1) % NSKPY;
      *displacement += ((nyt * coord[1] - NBGY) / NSKPY + 1) * rec_NX;
    } else
      *rec_nbgy = NBGY - nyt * coord[1] - 1;  // since rec_nbgy is 0-based
    if (nyt * (coord[1] + 1) <= NEDY)
      *rec_nedy = (nyt * (coord[1] + 1) + NBGY - 1) % NSKPY - NSKPY + nyt;
    else
      *rec_nedy = NEDY - nyt * coord[1] - 1;
    *rec_nyt = (*rec_nedy - *rec_nbgy) / NSKPY + 1;
  }

  if (NBGZ > nzt)
    *rec_nzt = 0;
  else {
    *rec_nbgz = NBGZ - 1;  // since rec_nbgz is 0-based
    *rec_nedz = NEDZ - 1;
    *rec_nzt = (*rec_nedz - *rec_nbgz) / NSKPZ + 1;
  }

  if (*rec_nxt == 0 || *rec_nyt == 0 || *rec_nzt == 0) {
    *rec_nxt = 0;
    *rec_nyt = 0;
    *rec_nzt = 0;
  }

  // displacement assumes NPZ=1!
  *displacement *= sizeof(float);
}

double gethrtime() {
  struct timeval TV;
  int RC = gettimeofday(&TV, NULL);

  if (RC == -1) {
    printf("Bad call to gettimeofday\n");
    return (-1);
  }

  return (((double)TV.tv_sec) + MICRO_SECOND * ((double)TV.tv_usec));
}

int main(int argc, char** argv) {
  float TMAX, DH, DT, ARBC, PHT;
  int NPC, ND, NSRC, NST;
  int NVE, NVAR, MEDIASTART, IFAULT, READ_STEP, READ_STEP_GPU;
  int NX, NY, NZ, PX, PY, IDYNA, SoCalQ;
  int NBGX, NEDX, NSKPX, NBGY, NEDY, NSKPY, NBGZ, NEDZ, NSKPZ;
  int nxt, nyt, nzt;
  MPI_Offset displacement;
  float FL, FH, FP;
  char INSRC[50], INVEL[50], OUT[50], INSRC_I2[50], CHKFILE[50];

  Grid3D u1 = NULL, v1 = NULL, w1 = NULL;
  Grid3D d1 = NULL, mu = NULL, lam = NULL;
  Grid3D xx = NULL, yy = NULL, zz = NULL, xy = NULL, yz = NULL, xz = NULL;
  Grid3D r1 = NULL, r2 = NULL, r3 = NULL, r4 = NULL, r5 = NULL, r6 = NULL;
  Grid3D qp = NULL, qs = NULL;
  PosInf tpsrc = NULL;
  Grid1D taxx = NULL, tayy = NULL, tazz = NULL, taxz = NULL, tayz = NULL, taxy = NULL;
  Grid1D Bufx = NULL;
  Grid1D Bufy = NULL, Bufz = NULL;
  Grid3D vx1 = NULL, vx2 = NULL, lam_mu = NULL;
  Grid1D dcrjx = NULL, dcrjy = NULL, dcrjz = NULL;
  float vse[2], vpe[2], dde[2];
  FILE* fchk;

  // GPU variables
  long int num_bytes;
  float* d_d1;
  float* d_u1;
  float* d_v1;
  float* d_w1;
  float* d_f_u1;
  float* d_f_v1;
  float* d_f_w1;
  float* d_b_u1;
  float* d_b_v1;
  float* d_b_w1;
  float* d_dcrjx;
  float* d_dcrjy;
  float* d_dcrjz;
  float* d_lam;
  float* d_mu;
  float* d_qp;
  float* d_qs;
  float* d_vx1;
  float* d_vx2;
  float* d_xx;
  float* d_yy;
  float* d_zz;
  float* d_xy;
  float* d_xz;
  float* d_yz;
  float* d_r1;
  float* d_r2;
  float* d_r3;
  float* d_r4;
  float* d_r5;
  float* d_r6;
  float* d_lam_mu;
  int* d_tpsrc;
  float* d_taxx;
  float* d_tayy;
  float* d_tazz;
  float* d_taxz;
  float* d_tayz;
  float* d_taxy;

  int i, j, k, idx, idy, idz;
  long int idtmp;
  long int tmpInd;
  const int maxdim = 3;
  float taumax, taumin, tauu;
  Grid3D tau = NULL, tau1 = NULL, tau2 = NULL;
  int npsrc;
  long int nt, cur_step, source_step;
  double time_un = 0.0;

  // MPI + CUDA variables
  cudaError_t cerr;
  cudaStream_t stream_1, stream_2, stream_i;
  int rank, size, err, srcproc, rank_gpu;
  int dim[2], period[2], coord[2], reorder;
  int x_rank_L = -1, x_rank_R = -1, y_rank_F = -1, y_rank_B = -1;
  MPI_Comm MCW, MC1;
  MPI_Request request_x[4], request_y[4];
  MPI_Status status_x[4], status_y[4], filestatus;
  MPI_Datatype filetype;
  MPI_File fh;
  int msg_v_size_x, msg_v_size_y, count_x = 0, count_y = 0;
  int xls, xre, xvs, xve, xss1, xse1, xss2, xse2, xss3, xse3;
  int yfs, yfe, ybs, ybe, yls, yre;
  float* SL_vel;  // Velocity to be sent to   Left  in x direction (u1,v1,w1)
  float* SR_vel;  // Velocity to be Sent to   Right in x direction (u1,v1,w1)
  float* RL_vel;  // Velocity to be Recv from Left  in x direction (u1,v1,w1)
  float* RR_vel;  // Velocity to be Recv from Right in x direction (u1,v1,w1)
  float* SF_vel;  // Velocity to be sent to   Front in y direction (u1,v1,w1)
  float* SB_vel;  // Velocity to be Sent to   Back  in y direction (u1,v1,w1)
  float* RF_vel;  // Velocity to be Recv from Front in y direction (u1,v1,w1)
  float* RB_vel;  // Velocity to be Recv from Back  in y direction (u1,v1,w1)

  int tmpSize;
  int WRITE_STEP;
  int NTISKP;
  int rec_NX;
  int rec_NY;
  int rec_NZ;
  int rec_nxt;
  int rec_nyt;
  int rec_nzt;
  int rec_nbgx;  // 0-based indexing, however NBG* is 1-based
  int rec_nedx;  // 0-based indexing, however NED* is 1-based
  int rec_nbgy;  // 0-based indexing
  int rec_nedy;  // 0-based indexing
  int rec_nbgz;  // 0-based indexing
  int rec_nedz;  // 0-based indexing
  char filename[50];
  char filenamebasex[50];
  char filenamebasey[50];
  char filenamebasez[50];

  //  variable initialization begins
  parseArguments(argc, argv, &TMAX, &DH, &DT, &ARBC, &PHT, &NPC, &ND, &NSRC, &NST, &NVAR, &NVE, &MEDIASTART, &IFAULT, &READ_STEP, &READ_STEP_GPU, &NTISKP, &WRITE_STEP, &NX, &NY, &NZ, &PX, &PY, &NBGX, &NEDX, &NSKPX, &NBGY, &NEDY, &NSKPY, &NBGZ, &NEDZ, &NSKPZ, &FL, &FH, &FP, &IDYNA, &SoCalQ, INSRC, INVEL, OUT, INSRC_I2, CHKFILE);

  sprintf(filenamebasex, "%s/SX", OUT);
  sprintf(filenamebasey, "%s/SY", OUT);
  sprintf(filenamebasez, "%s/SZ", OUT);

  // TODO: Remove
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_dup(MPI_COMM_WORLD, &MCW);
  MPI_Barrier(MCW);

  nxt = NX / PX;
  nyt = NY / PY;
  nzt = NZ;
  nt = (int)(TMAX / DT) + 1;
  dim[0] = PX;
  dim[1] = PY;
  period[0] = 0;
  period[1] = 0;
  reorder = 1;

  // TODO: Remove
  err = MPI_Cart_create(MCW, 2, dim, period, reorder, &MC1);
  err = MPI_Cart_shift(MC1, 0, 1, &x_rank_L, &x_rank_R);
  err = MPI_Cart_shift(MC1, 1, 1, &y_rank_F, &y_rank_B);
  err = MPI_Cart_coords(MC1, rank, 2, coord);
  err = MPI_Barrier(MCW);

  // If any neighboring rank is out of bounds, then MPI_Cart_shift sets the
  // destination argument to a negative number. We use the convention that -1
  // denotes ranks out of bounds.
  if (x_rank_L < 0) {
    x_rank_L = -1;
  }

  if (x_rank_R < 0) {
    x_rank_R = -1;
  }

  if (y_rank_F < 0) {
    y_rank_F = -1;
  }

  if (y_rank_B < 0) {
    y_rank_B = -1;
  }

  // Select GPU for 1 GPU per node systems
  rank_gpu = 0;
  cudaSetDevice(rank_gpu);

  printf("\n\nrank=%d) RS=%d, RSG=%d, NST=%d, IF=%d\n\n\n", rank, READ_STEP, READ_STEP_GPU, NST, IFAULT);

  // Same for each processor:
  if (NEDX == -1) NEDX = NX;
  if (NEDY == -1) NEDY = NY;
  if (NEDZ == -1) NEDZ = NZ;

  // Make NED's a record point
  // For instance if NBGX:NSKPX:NEDX = 1:3:9,
  // then we have 1,4,7 but NEDX=7 is better.
  NEDX = NEDX - (NEDX - NBGX) % NSKPX;
  NEDY = NEDY - (NEDY - NBGY) % NSKPY;
  NEDZ = NEDZ - (NEDZ - NBGZ) % NSKPZ;

  // Number of recording points in total
  rec_NX = (NEDX - NBGX) / NSKPX + 1;
  rec_NY = (NEDY - NBGY) / NSKPY + 1;
  rec_NZ = (NEDZ - NBGZ) / NSKPZ + 1;

  // Specific to each processor:
  calcRecordingPoints(&rec_nbgx, &rec_nedx, &rec_nbgy, &rec_nedy, &rec_nbgz, &rec_nedz, &rec_nxt, &rec_nyt, &rec_nzt, &displacement, (long int)nxt, (long int)nyt, (long int)nzt, rec_NX, rec_NY, rec_NZ, NBGX, NEDX, NSKPX, NBGY, NEDY, NSKPY, NBGZ, NEDZ, NSKPZ, coord);
  printf("%d = (%d,%d)) NX,NY,NZ=%d,%d,%d\nnxt,nyt,nzt=%d,%d,%d\nrec_N=(%d,%d,%d)\nrec_nxt,=%d,%d,%d\nNBGX,SKP,END=(%d:%d:%d),(%d:%d:%d),(%d:%d:%d)\nrec_nbg,ed=(%d,%d),(%d,%d),(%d,%d)\ndisp=%ld\n", rank, coord[0], coord[1], NX, NY, NZ, nxt, nyt, nzt, rec_NX, rec_NY, rec_NZ, rec_nxt, rec_nyt, rec_nzt, NBGX, NSKPX, NEDX, NBGY, NSKPY, NEDY, NBGZ, NSKPZ, NEDZ, rec_nbgx, rec_nedx, rec_nbgy, rec_nedy, rec_nbgz, rec_nedz, (long int)displacement);

  int maxNX_NY_NZ_WS = std::max(rec_NX, rec_NY);
  maxNX_NY_NZ_WS = std::max(maxNX_NY_NZ_WS, rec_NZ);
  maxNX_NY_NZ_WS = std::max(maxNX_NY_NZ_WS, WRITE_STEP);
  int ones[maxNX_NY_NZ_WS];
  MPI_Aint dispArray[maxNX_NY_NZ_WS];
  for (i = 0; i < maxNX_NY_NZ_WS; ++i) {
    ones[i] = 1;
  }

  // TODO: Remove
  err = MPI_Type_contiguous(rec_nxt, MPI_FLOAT, &filetype);
  err = MPI_Type_commit(&filetype);
  for (i = 0; i < rec_nyt; i++) {
    dispArray[i] = sizeof(float);
    dispArray[i] = dispArray[i] * rec_NX * i;
  }
  err = MPI_Type_create_hindexed(rec_nyt, ones, dispArray, filetype, &filetype);
  err = MPI_Type_commit(&filetype);
  for (i = 0; i < rec_nzt; i++) {
    dispArray[i] = sizeof(float);
    dispArray[i] = dispArray[i] * rec_NY * rec_NX * i;
  }
  err = MPI_Type_create_hindexed(rec_nzt, ones, dispArray, filetype, &filetype);
  err = MPI_Type_commit(&filetype);
  for (i = 0; i < WRITE_STEP; i++) {
    dispArray[i] = sizeof(float);
    dispArray[i] = dispArray[i] * rec_NZ * rec_NY * rec_NX * i;
  }
  err = MPI_Type_create_hindexed(WRITE_STEP, ones, dispArray, filetype, &filetype);
  err = MPI_Type_commit(&filetype);
  MPI_Type_size(filetype, &tmpSize);
  if (rank == 0) printf("filetype size (supposedly=rec_nxt*nyt*nzt*WS*4=%d) =%d\n", rec_nxt * rec_nyt * rec_nzt * WRITE_STEP * 4, tmpSize);

  printf("rank=%d, x_rank_L=%d, x_rank_R=%d, y_rank_F=%d, y_rank_B=%d\n", rank, x_rank_L, x_rank_R, y_rank_F, y_rank_B);

  if (x_rank_L < 0)
    xls = 2 + 4 * loop;
  else
    xls = 4 * loop;

  if (x_rank_R < 0)
    xre = nxt + 4 * loop + 1;
  else
    xre = nxt + 4 * loop + 3;

  xvs = 2 + 4 * loop;
  xve = nxt + 4 * loop + 1;

  xss1 = xls;
  xse1 = 4 * loop + 3;
  xss2 = 4 * loop + 4;
  xse2 = nxt + 4 * loop - 1;
  xss3 = nxt + 4 * loop;
  xse3 = xre;

  if (y_rank_F < 0)
    yls = 2 + 4 * loop;
  else
    yls = 4 * loop;

  if (y_rank_B < 0)
    yre = nyt + 4 * loop + 1;
  else
    yre = nyt + 4 * loop + 3;

  yfs = 2 + 4 * loop;
  yfe = 2 + 8 * loop - 1;
  ybs = nyt + 2;
  ybe = nyt + 4 * loop + 1;

  if (rank == 0) printf("Before initializeSource\n");
  err = initializeSource(rank, IFAULT, NSRC, READ_STEP, NST, &srcproc, NZ, MCW, nxt, nyt, nzt, coord, maxdim, &npsrc, &tpsrc, &taxx, &tayy, &tazz, &taxz, &tayz, &taxy, INSRC, INSRC_I2);
  if (err) {
    printf("Source initialization failed\n");
    return -1;
  }
  if (rank == 0) printf("After initializeSource\n");

  if (rank == srcproc) {
    printf("rank=%d, source rank, npsrc=%d\n", rank, npsrc);
    num_bytes = sizeof(float) * npsrc * READ_STEP_GPU;
    cudaMalloc(&d_taxx, num_bytes);
    cudaMalloc(&d_tayy, num_bytes);
    cudaMalloc(&d_tazz, num_bytes);
    cudaMalloc(&d_taxz, num_bytes);
    cudaMalloc(&d_tayz, num_bytes);
    cudaMalloc(&d_taxy, num_bytes);
    cudaMemcpy(d_taxx, taxx, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tayy, tayy, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tazz, tazz, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_taxz, taxz, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tayz, tayz, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_taxy, taxy, num_bytes, cudaMemcpyHostToDevice);
    num_bytes = sizeof(int) * npsrc * maxdim;
    cudaMalloc(&d_tpsrc, num_bytes);
    cudaMemcpy(d_tpsrc, tpsrc, num_bytes, cudaMemcpyHostToDevice);
  }

  d1 = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  mu = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  lam = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  lam_mu = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, 1);

  if (NVE == 1) {
    qp = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
    qs = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  }

  if (rank == 0) printf("Before initializeMesh\n");
  initializeMesh(MEDIASTART, d1, mu, lam, qp, qs, &taumax, &taumin, NVAR, FP, FL, FH, nxt, nyt, nzt, PX, PY, NX, NY, NZ, coord, MCW, IDYNA, NVE, SoCalQ, INVEL, vse, vpe, dde);
  if (rank == 0) printf("After initializeMesh\n");
  if (rank == 0)
    writeCHK(CHKFILE, NTISKP, DT, DH, nxt, nyt, nzt, nt, ARBC, NPC, NVE, FL, FH, FP, vse, vpe, dde);

  mediaswap(d1, mu, lam, qp, qs, rank, x_rank_L, x_rank_R, y_rank_F, y_rank_B, nxt, nyt, nzt, MCW);

  for (i = xls; i < xre + 1; i++)
    for (j = yls; j < yre + 1; j++) {
      float t_xl, t_xl2m;
      t_xl = 1.0 / lam[i][j][nzt + align - 1];
      t_xl2m = 2.0 / mu[i][j][nzt + align - 1] + t_xl;
      lam_mu[i][j][0] = t_xl / t_xl2m;
    }

  num_bytes = sizeof(float) * (nxt + 4 + 8 * loop) * (nyt + 4 + 8 * loop);
  cudaMalloc(&d_lam_mu, num_bytes);
  cudaMemcpy(d_lam_mu, &lam_mu[0][0][0], num_bytes, cudaMemcpyHostToDevice);

  vx1 = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  vx2 = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  if (NPC == 0) {
    dcrjx = Alloc1D(nxt + 4 + 8 * loop);
    dcrjy = Alloc1D(nyt + 4 + 8 * loop);
    dcrjz = Alloc1D(nzt + 2 * align);

    for (i = 0; i < nxt + 4 + 8 * loop; i++)
      dcrjx[i] = 1.0;
    for (j = 0; j < nyt + 4 + 8 * loop; j++)
      dcrjy[j] = 1.0;
    for (k = 0; k < nzt + 2 * align; k++)
      dcrjz[k] = 1.0;

    inicrj(ARBC, coord, nxt, nyt, nzt, NX, NY, ND, dcrjx, dcrjy, dcrjz);
  }

  if (NVE == 1) {
    tau = Alloc3D(2, 2, 2);
    tau1 = Alloc3D(2, 2, 2);
    tau2 = Alloc3D(2, 2, 2);
    tausub(tau, taumin, taumax);
    float dt1 = 1.0 / DT;
    for (i = 0; i < 2; i++)
      for (j = 0; j < 2; j++)
        for (k = 0; k < 2; k++) {
          tauu = tau[i][j][k];
          tau1[i][j][k] = 1.0 / ((tauu * dt1) + (1.0 / 2.0));
          tau2[i][j][k] = (tauu * dt1) - (1.0 / 2.0);
        }

    init_texture(nxt, nyt, nzt, tau1, tau2, vx1, vx2, xls, xre, yls, yre);

    Delloc3D(tau);
    Delloc3D(tau1);
    Delloc3D(tau2);
  }

  if (rank == 0) printf("Allocate device media pointers and copy.\n");
  num_bytes = sizeof(float) * (nxt + 4 + 8 * loop) * (nyt + 4 + 8 * loop) * (nzt + 2 * align);
  cudaMalloc(&d_d1, num_bytes);
  cudaMemcpy(d_d1, &d1[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_lam, num_bytes);
  cudaMemcpy(d_lam, &lam[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_mu, num_bytes);
  cudaMemcpy(d_mu, &mu[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_qp, num_bytes);
  cudaMemcpy(d_qp, &qp[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_qs, num_bytes);
  cudaMemcpy(d_qs, &qs[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_vx1, num_bytes);
  cudaMemcpy(d_vx1, &vx1[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_vx2, num_bytes);
  cudaMemcpy(d_vx2, &vx2[0][0][0], num_bytes, cudaMemcpyHostToDevice);

  cudaTextureObject_t d_vx1_tex, d_vx2_tex;
  create1DFloatTextureObject(&d_vx1_tex, d_vx1, num_bytes);
  create1DFloatTextureObject(&d_vx2_tex, d_vx2, num_bytes);

  if (NPC == 0) {
    num_bytes = sizeof(float) * (nxt + 4 + 8 * loop);
    cudaMalloc(&d_dcrjx, num_bytes);
    cudaMemcpy(d_dcrjx, dcrjx, num_bytes, cudaMemcpyHostToDevice);
    num_bytes = sizeof(float) * (nyt + 4 + 8 * loop);
    cudaMalloc(&d_dcrjy, num_bytes);
    cudaMemcpy(d_dcrjy, dcrjy, num_bytes, cudaMemcpyHostToDevice);
    num_bytes = sizeof(float) * (nzt + 2 * align);
    cudaMalloc(&d_dcrjz, num_bytes);
    cudaMemcpy(d_dcrjz, dcrjz, num_bytes, cudaMemcpyHostToDevice);
  }

  if (rank == 0) printf("Allocate host velocity and stress pointers.\n");
  u1 = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  v1 = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  w1 = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  xx = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  yy = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  zz = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  xy = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  yz = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  xz = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  if (NVE == 1) {
    r1 = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
    r2 = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
    r3 = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
    r4 = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
    r5 = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
    r6 = Alloc3D(nxt + 4 + 8 * loop, nyt + 4 + 8 * loop, nzt + 2 * align);
  }

  source_step = 1;
  if (rank == srcproc) {
    printf("%d) add initial src\n", rank);
    addsrc(source_step, DH, DT, NST, npsrc, READ_STEP, maxdim, tpsrc, taxx, tayy, tazz, taxz, tayz, taxy, xx, yy, zz, xy, yz, xz);
  }

  if (rank == 0) printf("Allocate device velocity and stress pointers and copy.\n");
  num_bytes = sizeof(float) * (nxt + 4 + 8 * loop) * (nyt + 4 + 8 * loop) * (nzt + 2 * align);
  cudaMalloc(&d_u1, num_bytes);
  cudaMemcpy(d_u1, &u1[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_v1, num_bytes);
  cudaMemcpy(d_v1, &v1[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_w1, num_bytes);
  cudaMemcpy(d_w1, &w1[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_xx, num_bytes);
  cudaMemcpy(d_xx, &xx[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_yy, num_bytes);
  cudaMemcpy(d_yy, &yy[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_zz, num_bytes);
  cudaMemcpy(d_zz, &zz[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_xy, num_bytes);
  cudaMemcpy(d_xy, &xy[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_xz, num_bytes);
  cudaMemcpy(d_xz, &xz[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_yz, num_bytes);
  cudaMemcpy(d_yz, &yz[0][0][0], num_bytes, cudaMemcpyHostToDevice);

  if (NVE == 1) {
    if (rank == 0) printf("Allocate additional device pointers (r) and copy.\n");
    cudaMalloc(&d_r1, num_bytes);
    cudaMemcpy(d_r1, &r1[0][0][0], num_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_r2, num_bytes);
    cudaMemcpy(d_r2, &r2[0][0][0], num_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_r3, num_bytes);
    cudaMemcpy(d_r3, &r3[0][0][0], num_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_r4, num_bytes);
    cudaMemcpy(d_r4, &r4[0][0][0], num_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_r5, num_bytes);
    cudaMemcpy(d_r5, &r5[0][0][0], num_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_r6, num_bytes);
    cudaMemcpy(d_r6, &r6[0][0][0], num_bytes, cudaMemcpyHostToDevice);
  }

  // Variable initialization ends
  if (rank == 0) printf("Allocate buffers of #elements: %d\n", rec_nxt * rec_nyt * rec_nzt * WRITE_STEP);
  Bufx = Alloc1D(rec_nxt * rec_nyt * rec_nzt * WRITE_STEP);
  Bufy = Alloc1D(rec_nxt * rec_nyt * rec_nzt * WRITE_STEP);
  Bufz = Alloc1D(rec_nxt * rec_nyt * rec_nzt * WRITE_STEP);

  num_bytes = sizeof(float) * 3 * (4 * loop) * (nyt + 4 + 8 * loop) * (nzt + 2 * align);
  cudaMallocHost(&SL_vel, num_bytes);
  cudaMallocHost(&SR_vel, num_bytes);
  cudaMallocHost(&RL_vel, num_bytes);
  cudaMallocHost(&RR_vel, num_bytes);

  num_bytes = sizeof(float) * 3 * (4 * loop) * (nxt + 4 + 8 * loop) * (nzt + 2 * align);
  cudaMallocHost(&SF_vel, num_bytes);
  cudaMallocHost(&SB_vel, num_bytes);
  cudaMallocHost(&RF_vel, num_bytes);
  cudaMallocHost(&RB_vel, num_bytes);

  num_bytes = sizeof(float) * (4 * loop) * (nxt + 4 + 8 * loop) * (nzt + 2 * align);
  cudaMalloc(&d_f_u1, num_bytes);
  cudaMalloc(&d_f_v1, num_bytes);
  cudaMalloc(&d_f_w1, num_bytes);
  cudaMalloc(&d_b_u1, num_bytes);
  cudaMalloc(&d_b_v1, num_bytes);
  cudaMalloc(&d_b_w1, num_bytes);

  msg_v_size_x = 3 * (4 * loop) * (nyt + 4 + 8 * loop) * (nzt + 2 * align);
  msg_v_size_y = 3 * (4 * loop) * (nxt + 4 + 8 * loop) * (nzt + 2 * align);

  setDeviceConstValue(DH, DT, nxt, nyt, nzt);

  cudaStreamCreate(&stream_1);
  cudaStreamCreate(&stream_2);
  cudaStreamCreate(&stream_i);

  if (rank == 0)
    fchk = fopen(CHKFILE, "a+");

  // Main loop starts
  if (NPC == 0 && NVE == 1) {
    time_un -= gethrtime();

    // This loop has no loverlapping because there is source input
    for (cur_step = 1; cur_step <= nt; cur_step++) {
      if (rank == 0) {
        if (cur_step % 100 == 0) {
          printf("Time Step = %ld of Total Timesteps = %ld\n", cur_step, nt);
          printf("Time per timestep:\t%lf seconds\n", (gethrtime() + time_un) / cur_step);
        }
      }
      cerr = cudaGetLastError();
      if (cerr != cudaSuccess) printf("CUDA ERROR! rank=%d before timestep: %s\n", rank, cudaGetErrorString(cerr));

      // Velocity computation in y boundary, two ghost cell regions
      dvelcy_H(d_u1, d_v1, d_w1, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_dcrjx, d_dcrjy, d_dcrjz, d_d1, nxt, nzt, d_f_u1, d_f_v1, d_f_w1, stream_i, yfs, yfe, y_rank_F);
      dvelcy_H(d_u1, d_v1, d_w1, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_dcrjx, d_dcrjy, d_dcrjz, d_d1, nxt, nzt, d_b_u1, d_b_v1, d_b_w1, stream_i, ybs, ybe, y_rank_B);
      Cpy2Host_VY(d_f_u1, d_f_v1, d_f_w1, SF_vel, nxt, nzt, stream_i, y_rank_F);
      Cpy2Host_VY(d_b_u1, d_b_v1, d_b_w1, SB_vel, nxt, nzt, stream_i, y_rank_B);
      cudaThreadSynchronize();

      // Velocity communication in y direction
      Cpy2Device_VY(d_u1, d_v1, d_w1, d_f_u1, d_f_v1, d_f_w1, d_b_u1, d_b_v1, d_b_w1, RF_vel, RB_vel, nxt, nyt, nzt, stream_i, stream_i, y_rank_F, y_rank_B);

      // Velocity computation whole 3D Grid (nxt, nyt, nzt)
      dvelcx_H(d_u1, d_v1, d_w1, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_dcrjx, d_dcrjy, d_dcrjz, d_d1, nyt, nzt, stream_i, xvs, xve);
      Cpy2Host_VX(d_u1, d_v1, d_w1, SL_vel, nxt, nyt, nzt, stream_i, x_rank_L, Left);
      Cpy2Host_VX(d_u1, d_v1, d_w1, SR_vel, nxt, nyt, nzt, stream_i, x_rank_R, Right);
      cudaThreadSynchronize();

      // Velocity communication in x direction
      Cpy2Device_VX(d_u1, d_v1, d_w1, RL_vel, RR_vel, nxt, nyt, nzt, stream_i, stream_i, x_rank_L, x_rank_R);

      // Stress computation whole 3D Grid (nxt+4, nyt+4, nzt)
      dstrqc_H(d_vx1_tex, d_vx2_tex, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_r1, d_r2, d_r3, d_r4, d_r5, d_r6, d_u1, d_v1, d_w1, d_lam, d_mu, d_qp, d_qs, d_dcrjx, d_dcrjy, d_dcrjz, nyt, nzt, stream_i, d_lam_mu, NX, coord[0], coord[1], xls, xre, yls, yre);

      // Update source input
      if (rank == srcproc && cur_step < NST) {
        ++source_step;
        addsrc_H(source_step, READ_STEP_GPU, maxdim, d_tpsrc, npsrc, stream_i, d_taxx, d_tayy, d_tazz, d_taxz, d_tayz, d_taxy, d_xx, d_yy, d_zz, d_xy, d_yz, d_xz);
      }
      cudaThreadSynchronize();

      if (cur_step % NTISKP == 0) {
        num_bytes = sizeof(float) * (nxt + 4 + 8 * loop) * (nyt + 4 + 8 * loop) * (nzt + 2 * align);
        cudaMemcpy(&u1[0][0][0], d_u1, num_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(&v1[0][0][0], d_v1, num_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(&w1[0][0][0], d_w1, num_bytes, cudaMemcpyDeviceToHost);
        idtmp = ((cur_step / NTISKP + WRITE_STEP - 1) % WRITE_STEP);
        idtmp = idtmp * rec_nxt * rec_nyt * rec_nzt;
        tmpInd = idtmp;

        for (k = nzt + align - 1 - rec_nbgz; k >= nzt + align - 1 - rec_nedz; k = k - NSKPZ)
          for (j = 2 + 4 * loop + rec_nbgy; j <= 2 + 4 * loop + rec_nedy; j = j + NSKPY)
            for (i = 2 + 4 * loop + rec_nbgx; i <= 2 + 4 * loop + rec_nedx; i = i + NSKPX) {
              Bufx[tmpInd] = u1[i][j][k];
              Bufy[tmpInd] = v1[i][j][k];
              Bufz[tmpInd] = w1[i][j][k];
              tmpInd++;
            }

        // Write-statistics to chk file:
        if (rank == 0) {
          i = ND + 2 + 4 * loop;
          j = i;
          k = nzt + align - 1 - ND;
          fprintf(fchk, "%ld :\t%e\t%e\t%e\n", cur_step, u1[i][j][k], v1[i][j][k], w1[i][j][k]);
          fflush(fchk);
        }
      }
    }
    time_un += gethrtime();
  }

  if (rank == 0) {
    fprintf(fchk, "END\n");
    fclose(fchk);
  }

  cudaStreamDestroy(stream_1);
  cudaStreamDestroy(stream_2);
  cudaStreamDestroy(stream_i);
  cudaFreeHost(SL_vel);
  cudaFreeHost(SR_vel);
  cudaFreeHost(RL_vel);
  cudaFreeHost(RR_vel);
  cudaFreeHost(SF_vel);
  cudaFreeHost(SB_vel);
  cudaFreeHost(RF_vel);
  cudaFreeHost(RB_vel);

  float GFLOPS = 1.0;
  GFLOPS = GFLOPS * 307.0 * (xre - xls) * (yre - yls) * nzt;
  GFLOPS = GFLOPS / (1000 * 1000 * 1000);
  time_un = time_un / cur_step;
  GFLOPS = GFLOPS / time_un;
  printf("GPU benchmark size NX=%d, NY=%d, NZ=%d, ReadStep=%d\n", NX, NY, NZ, READ_STEP);
  printf("GPU computing flops = %1.6f GFLOPS, time = %1.6f secs per timestep\n", GFLOPS, time_un);

  // Clean up
  cudaDestroyTextureObject(d_vx1_tex);
  cudaDestroyTextureObject(d_vx2_tex);
  Delloc3D(u1);
  Delloc3D(v1);
  Delloc3D(w1);
  Delloc3D(xx);
  Delloc3D(yy);
  Delloc3D(zz);
  Delloc3D(xy);
  Delloc3D(yz);
  Delloc3D(xz);
  Delloc3D(vx1);
  Delloc3D(vx2);

  cudaFree(d_u1);
  cudaFree(d_v1);
  cudaFree(d_w1);
  cudaFree(d_f_u1);
  cudaFree(d_f_v1);
  cudaFree(d_f_w1);
  cudaFree(d_b_u1);
  cudaFree(d_b_v1);
  cudaFree(d_b_w1);
  cudaFree(d_xx);
  cudaFree(d_yy);
  cudaFree(d_zz);
  cudaFree(d_xy);
  cudaFree(d_yz);
  cudaFree(d_xz);
  cudaFree(d_vx1);
  cudaFree(d_vx2);

  if (NVE == 1) {
    Delloc3D(r1);
    Delloc3D(r2);
    Delloc3D(r3);
    Delloc3D(r4);
    Delloc3D(r5);
    Delloc3D(r6);
    cudaFree(d_r1);
    cudaFree(d_r2);
    cudaFree(d_r3);
    cudaFree(d_r4);
    cudaFree(d_r5);
    cudaFree(d_r6);

    Delloc3D(qp);
    Delloc3D(qs);
    cudaFree(d_qp);
    cudaFree(d_qs);
  }

  if (NPC == 0) {
    Delloc1D(dcrjx);
    Delloc1D(dcrjy);
    Delloc1D(dcrjz);
    cudaFree(d_dcrjx);
    cudaFree(d_dcrjy);
    cudaFree(d_dcrjz);
  }

  Delloc3D(d1);
  Delloc3D(mu);
  Delloc3D(lam);
  Delloc3D(lam_mu);
  cudaFree(d_d1);
  cudaFree(d_mu);
  cudaFree(d_lam);
  cudaFree(d_lam_mu);

  if (rank == srcproc) {
    Delloc1D(taxx);
    Delloc1D(tayy);
    Delloc1D(tazz);
    Delloc1D(taxz);
    Delloc1D(tayz);
    Delloc1D(taxy);
    cudaFree(d_taxx);
    cudaFree(d_tayy);
    cudaFree(d_tazz);
    cudaFree(d_taxz);
    cudaFree(d_tayz);
    cudaFree(d_taxy);
    Delloc1P(tpsrc);
    cudaFree(d_tpsrc);
  }

  // TODO: Remove
  MPI_Comm_free(&MC1);
  MPI_Finalize();
  return 0;
}
