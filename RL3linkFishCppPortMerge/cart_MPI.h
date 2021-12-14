#include "mpi.h"
#include <iostream>
#include <fstream>
#define NNnode 0
// #define invalidaction 999999999
#define INVALIDACTION std::numeric_limits<float>::max()
MPI_Status status;
MPI_Request request;

int nprocs;

const int control_vars = 1; // force along x
const int obs_vars = 6; // x, vel, angvel, angle, cosine, sine

// double dbufa[control_vars];
// double dbufsrt[obs_vars+2]; // the dbuf is sending these things: state, reward, whether an episode terminate or done  or start or not (if terminate 1, if timeup 2, if start 3, else 0) (terminate means fail or success; timeup means bootstrapping is needed, start means no need to store reward and env status)


const int Nepisodes=10000;
const int N_timestep=200;//Maximum time step in one episode
const int Max_timestep=1000000;//Maximum training timestep (either define this or define Nepisodes)
int updateTimestep = 5400;


// const int Nepisodes=90000;
// const int N_timestep=100;//Maximum time step in one episode
// const int Max_timestep=9000000;//Maximum training timestep (either define this or define Nepisodes)
// const int updateTimestep = 1900;
