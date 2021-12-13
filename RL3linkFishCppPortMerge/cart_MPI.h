#include "mpi.h"
#include <iostream>
#include <fstream>
#define NNnode 0
// #define invalidaction 999999999
#define INVALIDACTION std::numeric_limits<float>::max()
MPI_Status status;
MPI_Request request;
enum TRAIN_STATUS {NORMAL = 0, TERMINATE, DONE, START};

int nprocs;

const int control_vars = 1; // force along x
const int obs_vars = 6; // x, vel, angvel, angle, cosine, sine

// double dbufa[control_vars];
// double dbufsrt[obs_vars+2]; // the dbuf is sending these things: state, reward, whether an episode terminate or done  or start or not (if terminate 1, if done 2, if start 3, else 0) (terminate means fail; done means reach maximum time step in a period, thus need to add value function, start means the corresponding reward is 0, which is no meaning and will not be stored)
// const int Nepisodes=10000;
// const int N_timestep=5;//Maximum time step in one episode
// const int Max_timestep=1000000;//Maximum training timestep (either define this or define Nepisodes)
// int updateTimestep = 20;


const int Nepisodes=90000;
const int N_timestep=5;//Maximum time step in one episode
const int Max_timestep=9000000;//Maximum training timestep (either define this or define Nepisodes)
const int updateTimestep = 20;
std::ofstream myfile;
