#include "mpi.h"

MPI_Status status;
// MPI_Comm comm_action;
// MPI_Comm comm_observation;
// MPI_Comm comm_reward;
MPI_Request request;


const int control_vars = 1; // force along x
const int state_vars = 6; // x, vel, angvel, angle, cosine, sine
double dbuf[state_vars];
double dbufa[control_vars];
double dbufr[1];
bool 