#include "mpi.h"
#include <stdio.h>

//Modify these to change timing scenario

#define TRIALS 10
#define STEPS 15
#define MAX_MSGSIZE (1<<STEPS) //2^STEPS
#define REPS 1000
#define MAXPOINTS 10000

char sbuff[MAX_MSGSIZE];
char rbuff[MAX_MSGSIZE];
int msgsizes[MAXPOINTS];
double results[MAXPOINTS];

int main (int argc ,char *argv[] )
{

   int numtasks, rank, tag =999;
   int n, i, j, k;
   double mbytes, tbytes, ttime, t1, t2;
   MPI_Status stats[2];
   MPI_Request reqs[2];
   MPI_Init (&argc, &argv);

   char *sbuff_ptr = &sbuff[0];
   char *rbuff_ptr = &rbuff[0];

   MPI_Comm_size(MPI_COMM_WORLD, &numtasks );
   MPI_Comm_rank(MPI_COMM_WORLD, &rank );

#pragma omp target enter data map(alloc: rbuff_ptr[0:MAX_MSGSIZE], sbuff_ptr[0:MAX_MSGSIZE])

   //task  0
   if(rank == 0)
   {
	   //Greeting
   	printf("\nPersistent Communications\n");
      printf(" Trials=      %8d\n" ,TRIALS);
      printf(" Reps/trial=  %8d\n" ,REPS);
      printf(" Message Size   Bandwidth (mbytes/sec)\n");

      //Initializations
      n=1;
      for( i =0; i<=STEPS ; i++)
      {
   	   msgsizes[i] = n;
         results[i] = 0.0;
         n=n*2;
      }
      for( i =0; i<MAX_MSGSIZE; i++)
      {
         sbuff[i] = 'x';
      }

      //Begin timings
      for( k=0; k<TRIALS; k++)
      {
         n=1;
         for( j =0; j<=STEPS; j++)
         {
   	      //Setup persistent requests for both the send and receive
#pragma omp target data use_device_ptr(rbuff_ptr, sbuff_ptr)
{
	         MPI_Recv_init(rbuff_ptr, n, MPI_CHAR, 1, tag ,MPI_COMM_WORLD, reqs );
	         MPI_Send_init(sbuff_ptr, n ,MPI_CHAR, 1, tag ,MPI_COMM_WORLD, reqs +1);
}

            t1= MPI_Wtime();
            for( i =1; i<=REPS; i++)
            {
               MPI_Startall(2, reqs );
               MPI_Waitall(2, reqs, stats);
            }
            t2 = MPI_Wtime();

            //Compute bandwidth and save best result over all TRIALS
            ttime= t2 - t1;
            tbytes= sizeof(char)*n*2.0*(double)REPS;
            mbytes= tbytes /ttime;
            if(results[j] < mbytes )
            {
               results[j] = mbytes;
            }

            //Free persistent requests
            MPI_Request_free(reqs);
            MPI_Request_free(reqs+1);
            n=n*2;

         } // end j loop
      } // end k loop

      //Print results
      for(j=0; j<=STEPS; j++)
      {
         printf("%9d %16zu\n", msgsizes[j], (size_t)results[j]/1024/1024 );
      }
   } //end of task 0

   // task  1
   if(rank == 1)
   {
      //Begin timing test
      for( k=0; k<TRIALS; k++)
      {
         n=1;
         for(j=0; j<=STEPS; j++)
         {
            //Setup  persistent requests for both the send and receive
#pragma omp target data use_device_ptr(rbuff_ptr, sbuff_ptr)
{
            MPI_Recv_init(rbuff_ptr, n, MPI_CHAR, 0, tag ,MPI_COMM_WORLD, &reqs[0]);
            MPI_Send_init(sbuff_ptr, n, MPI_CHAR, 0, tag ,MPI_COMM_WORLD, &reqs[1]);
}
            for(i=1; i<=REPS; i++)
            {
               MPI_Startall(2, reqs);
               MPI_Waitall(2, reqs, stats);
            }

            //Free persistent requests
            MPI_Request_free(&reqs[0]);
            MPI_Request_free(&reqs[1]);
            n=n*2;
         } //end j loop
      }//end k loop
   } // end task 1

   MPI_Finalize();

   return 0;
} // end of main
