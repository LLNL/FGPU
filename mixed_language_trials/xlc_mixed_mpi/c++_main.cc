#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

extern "C" void fhello(int *);
int main(int argc, char *argv[])
{
    int  numtasks, rank, rc;
    MPI_Comm comm;
    MPI_Fint fcomm;


       cout << "Howdy from C++!" << endl;
    rc = MPI_Init(&argc,&argv);
    if (rc != MPI_SUCCESS) {
      printf ("Error starting MPI program. Terminating.\n");
      MPI_Abort(MPI_COMM_WORLD, rc);
      }

    comm = MPI_COMM_WORLD;
    fcomm = MPI_Comm_c2f(comm);
    printf ("Size of comm %i size of fcomm %i\n", sizeof(comm), sizeof(fcomm));

    cerr << "Howdy after" << endl;
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

       cout << "Calling fortran hello!" << endl;
       fhello(&fcomm); 
       cout << "Trying to reproduce bad thing without conversion" << endl;
       fhello((int *)(&comm)); 
       cout << "Back in C++!" << endl;

    MPI_Finalize();
	return (0);
}
