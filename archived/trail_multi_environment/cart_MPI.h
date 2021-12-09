#include "mpi.h"
#define NNnode 0
// #define invalidaction 999999999
#define invalidaction std::numeric_limits<double>::max()
MPI_Status status;
MPI_Request request;

int nprocs;

const int control_vars = 1; // force along x
const int obs_vars = 6; // x, vel, angvel, angle, cosine, sine

// double dbufa[control_vars];
// double dbufsrt[obs_vars+2]; // the dbuf is sending these things: state, reward, whether an episode terminate or done  or start or not (if terminate 1, if done 2, if start 3, else 0) (terminate means fail; done means reach maximum time step in a period, thus need to add value function, start means the corresponding reward is 0, which is no meaning and will not be stored)
const int Nepisodes=9;
const int N_timestep=9;//Maximum time step in one episode
const int Max_timestep=9;//Maximum training timestep (either define this or define Nepisodes)
