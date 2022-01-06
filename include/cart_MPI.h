#include <fstream>
#include <iostream>

#include "mpi.h"
#define NNnode 0
// #define invalidaction 999999999
#define INVALIDACTION std::numeric_limits<float>::max()
MPI_Status status;
MPI_Request request;

int nprocs;

int control_vars;  // force along x
int obs_vars;      // x, vel, angvel, angle, cosine, sine

// double dbufa[control_vars];
// double dbufsrt[obs_vars+2]; // the dbuf is sending these things: state,
// reward, whether an episode terminate or done  or start or not (if terminate
// 1, if timeup 2, if start 3, else 0) (terminate means fail or success; timeup
// means bootstrapping is needed, start means no need to store reward and env
// status)

int Nepisodes;
int N_timestep;    // Maximum time step in one episode
int Max_timestep;  // Maximum training timestep (either define this or define
                   // Nepisodes)
int updateTimestep;
int update_pre_timestep;

// const int Nepisodes=90000;
// const int N_timestep=100;//Maximum time step in one episode
// const int Max_timestep=9000000;//Maximum training timestep (either define
// this or define Nepisodes) const int updateTimestep = 1900;
