//
//  main.cpp
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//

// compile: mpic++  -o comm_trial comm_trial.cpp -fopenmp
// running: mpirun -n 3 ./comm_trial 

#include <iostream>
#include <cstdio>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <unistd.h>

#define C_Node 0
#define Num 9
#define size 3

int nprocs;
MPI_Status status;

inline void center_node(){
    double dbuf[size]={4,5,6};
    double dbuf_r[size];
    int std;
    // for (int i=0; i < Num; i++){
    //   printf("0: %d \n", i);
    //   for (int j=1;j<nprocs;j++){
    //     MPI_Recv(dbuf_r, size, MPI_DOUBLE, j, j, MPI_COMM_WORLD, &status);
    //     printf("     recieve from process %d  \n",  j);
    //     MPI_Send(dbuf, size, MPI_DOUBLE, j, j+nprocs*2, MPI_COMM_WORLD);
    //     printf("     send to process %d \n",  j);
    //   }
    // }
    #pragma omp parallel private(dbuf,dbuf_r,std)
    {
      std=omp_get_thread_num();
      printf("process 0 : thread: %d \n", std);
      int j;
      j=std+1;
      for (int i=0; i < Num; i++)
      {
        MPI_Recv(dbuf_r, size, MPI_DOUBLE, j, j, MPI_COMM_WORLD, &status);
        printf("     recieve from process %d  \n",  j);
        MPI_Send(dbuf, size, MPI_DOUBLE, j, j+nprocs*2, MPI_COMM_WORLD);
        printf("     send to process %d \n",  j);
      }
    }
}


inline void distributed_node(int myid){
    double dbuf[size]={1,2,3};
    double dbuf_r[size];
    printf("myid: %d \n",myid);
    for (int i=0; i < Num; i++){
        printf("%d: %d \n", myid, i);
        MPI_Send(dbuf, size, MPI_DOUBLE, C_Node, myid, MPI_COMM_WORLD);
        printf("send from process %d \n",  myid);
        MPI_Recv(dbuf_r, size, MPI_DOUBLE, C_Node, myid+nprocs*2, MPI_COMM_WORLD, &status);
        printf(" process %d recieve \n",  myid);
    }
}


int main(int argc, char**argv)
{

  int myid;
  int n;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  
  if (myid == 0) {
    printf("There are %d processes running in this MPI program\n", nprocs);
    center_node();
  }
  else {
    distributed_node(myid);
  }
  MPI_Finalize();
  return 0;
}