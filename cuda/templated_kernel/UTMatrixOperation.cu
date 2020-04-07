/*
   This is an upper-triangularization operation on a 'nearly upper triangular' matrix.
   In order to place rME and rMI in registers for optimal performance, the number of
   non-zero values below the subdiagonal must be known at compile time.  The only
   known way to do this is templating the function. See line 31.

   This is a simplied version that just solves a single matrix of ngr*ngr size.
   The kernel should be launched with # of threads = ngr.
*/
   template<unsigned char NSUBDIAG>
__global__ void kernelDecomposeRegister ( int ngr,
                                          const int * __restrict__ ODL, // array size = ngr
                                          double * __restrict__ MatrixArrayRM, // RM = in row-major order
                                          double *rhs
                                        )

{
   extern __shared__ double sMEM[];

   // require 19 * ngr SMEM storage  (19 kB for ngr = 128)
   // ... one for current row
   // ... one for column[row] of descendants
   // ... one for nAL for each row

   double * __restrict__ sNormal        = sMEM;
   double * __restrict__ sCurrentRow    = sMEM+1;
   double * __restrict__ sCurrentColumn = sCurrentRow + ngr;  // this really only needs to be size 16 (blockDim.x / ngr)
   double * __restrict__ sRhs           = sCurrentColumn + ngr;
   int    * __restrict__ sNAL           = (int *) (sRhs + ngr);

   // require a compile-time known number of descendants to be placed in registers
   double       rME[NSUBDIAG];                // [r]egister [M]atrix [E]lement
   unsigned int rMI[NSUBDIAG];                // [r]egister [M]atrix [I]ndex

   unsigned int tid = threadIdx.x;
   unsigned int tidongr = tid/ngr;
   unsigned int tidmngr = tid%ngr;

   // load first blockDim.x/ngr lines of dense matrix                 // this assumes ngr >= 16   // is this comment correct??   Seems out of date.
   for ( int ireg=0; ireg<NSUBDIAG; ireg++ ){                    // rows per thread is explicit
      rME[ireg] = MatrixArrayRM[tid + ireg*blockDim.x];
      rMI[ireg] = tidongr + ireg;
   }

   // the zeroth column for each rows places column[0] in SMEM        // place the first column into shared memory
   if ( tidmngr == 0 ) {                                              // this is what will be used to compute the row operations
      for ( int ireg=0; ireg<NSUBDIAG; ireg++ ) {
         sCurrentColumn[tidongr + ireg] = rME[ireg];
      }
   }

   // compute nAL for each row from ODL and place in SMEM  // nAL = number of elements below a given diagona.
   if ( tid < ngr-1 ) {   // this can be made an else clause from the previous if to avoid divergence.
      sNAL[tid] = ODL[tid+1] - ODL[tid];
   }

   // place the RHS in SMEM
   if ( tid < ngr ) {
      sRhs[tid] = rhs[tid];
   }

   __syncthreads();

   // normalize first line of dense matrix - just first ngr threads
   if ( tid == 0 ) {
      sRhs[0] /= sCurrentColumn[0];
      MatrixArrayRM[0] = 1.0;
   }
   else if ( tid < ngr ) {
      rME[0] /= sCurrentColumn[0];
      // place result in SMEM
      sCurrentRow[tid] = rME[0];
      // write to GMEM
      MatrixArrayRM[tid] = rME[0];
   } 

   __syncthreads();

   for ( int irow=0; irow<ngr-1 ; irow++ ) {
      for ( int ireg=0; ireg<NSUBDIAG; ireg++ ) {
         if ( rMI[ireg] == irow ) {                                       // if this row is the diagonal, 
            rMI[ireg] += NSUBDIAG;
            if ( rMI[ireg] < ngr ) {                                       //   if we haven't progressed past the end of the matrix                
               rME[ireg] = MatrixArrayRM[(tidmngr) + rMI[ireg]*ngr];	 //     then read in the next row.
            }
         }

         if ( rMI[ireg] - irow <= sNAL[irow] ) {                          // check if this row is within the number of descendants of the current diagonal
            // if so, then subtract irow 
            if ( tidmngr >= irow ) {
               rME[ireg] -= sCurrentColumn[rMI[ireg]] * sCurrentRow[tidmngr];
            }
            // update RHS
            if ( tidmngr == 0 ) {
               sRhs[rMI[ireg]] -= sCurrentColumn[rMI[ireg]] * sRhs[irow];
            }
         }

         if ( rMI[ireg] == irow+1 && tidmngr == irow+1 ) {               // if this will be the next eliminated (i.e. next iteration this will be the diagonal row)
            *sNormal = 1.0/rME[ireg];                                     //   then cache diagonal value so the row can be normalized
         }

      }
      __syncthreads();

      for ( int ireg=0; ireg < NSUBDIAG; ireg++ )
      {
         if ( rMI[ireg] == irow+1 ) {                                    // if this thread has row irow+1 cached (i.e. the next row to eliminate)
            // write out
            if ( tidmngr > irow ) {                                       // if the column is in the upper triangular
               // normalize
               rME[ireg] *= *sNormal;                                      // normalize
               sCurrentRow[tidmngr] = rME[ireg];                           // write to shared
               MatrixArrayRM[(tidmngr)+(irow+1)*ngr] = rME[ireg];          // also write to GMEM (as it is the fully upper-triangularized result)
            }
            if ( tidmngr == 0 ) {                                         // use the zeroth thread in the row to normalize the RHS
               sRhs[rMI[ireg]] *= *sNormal;
            }
         }

         if ( tidmngr == irow+1 ) {
            if (rMI[ireg] < ngr) {
               sCurrentColumn[rMI[ireg]] = rME[ireg];
            }
         }

      }

      // syncthreads ------------------------------------------------
      __syncthreads();    

   }

   // output updated rhs
   if ( tid < ngr ) {
      rhs[tid] = sRhs[tid];
   }
}
