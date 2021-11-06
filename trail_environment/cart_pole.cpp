//
//  main.cpp
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//

// compile: mpic++  -o cart cart_pole.cpp -fopenmp
// running: mpirun ./cart

#include "cart_pole.h"
#include "cart_MPI.h"
#include "custom_action.h"
#include <iostream>
#include <cstdio>
#include "mpi.h"

// inline void app_main(int argc, char**argv)
inline void app_main()
{
  
//   comm->setStateActionDims(state_vars, control_vars);

  //OPTIONAL: action bounds
  bool bounded = true;
  std::vector<double> upper_action_bound{10}, lower_action_bound{-10};
//   comm->setActionScales(upper_action_bound, lower_action_bound, bounded);

  /*
    // ALTERNATIVE for discrete actions:
    vector<int> n_options = vector<int>{2};
    comm->set_action_options(n_options);
    // will receive either 0 or 1, app chooses resulting outcome
  */

  //OPTIONAL: hide state variables. e.g. show cosine/sine but not angle
  // std::vector<bool> b_observable = {true, true, true, false, true, true};
  //std::vector<bool> b_observable = {true, false, false, false, true, true};
//   comm->setStateObservable(b_observable);
  //comm->setIsPartiallyObservable();

  CartPole env;

  // while(true) //train loop
  for (int i=0;i<Nepisodes;i++)
  {
    env.reset(); // prng with different seed on each process
    // comm->sendInitState(env.getState()); //send initial state
    std::vector<double> state = env.getState();
    for (int i=0;i<state_vars;i++) dbuf[i]=state[i];
    MPI_Request request;
    MPI_Isend(dbuf, state_vars, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD, &request);
    
    printf("send state = %f %f %f %f %f %f \n", state[0] ,state[1], state[2] , state[3] ,state[4], state[5]);
    MPI_Wait(&request, MPI_STATUSES_IGNORE);

    // while (true) //simulation loop
    for (int j=0; j <N_timestep; j++)
    {
      // MPI_Irecv(dbufa, control_vars, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD, &request);
      // MPI_Wait( &request, &status);
      MPI_Request request_action;
      MPI_Irecv(dbufa, control_vars, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD, &request_action); // receive action
      MPI_Status status_action;
      MPI_Wait(&request_action, &status_action);

      std::vector<double> action(control_vars);
      for (int i=0;i<control_vars;i++) action[i]=dbufa[i];
      printf("receive action = %f  \n", action[0]);

      // if(comm->terminateTraining()) return; // exit program

      bool poleFallen = env.advance(action); //advance the simulation:

      std::vector<double> state = env.getState(); 
      double reward = env.getReward();
      // send new observation, reward, and whether terminate or not, if terminate send 1, if not send 0
      for (int i=0;i<state_vars;i++) dbufsrt[i]=state[i];
      dbufsrt[state_vars]=reward;
      MPI_Request request_ob_rew_term;
      if(poleFallen){
        dbufsrt[state_vars+1]=1;
        MPI_Isend(dbufsrt, state_vars+2, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD, &request_ob_rew_term);
        break;
      }else{
        dbufsrt[state_vars+1]=0;
        MPI_Isend(dbufsrt, state_vars+2, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD, &request_ob_rew_term);
      }

      MPI_Wait(&request_ob_rew_term, MPI_STATUSES_IGNORE);
    }  //end of simulation loop
  }// end of train loop
}

int main(int argc, char**argv)
{
  
  int myid;
  int n;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (myid == 0) {
    // n = 777;
    // printf("sender n = %d\n", n);
    // MPI_Send(&n, 1, MPI_INT, 1, 10, MPI_COMM_WORLD);
    app_main();
    
  }
  else {
    // MPI_Recv(&n, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
    // printf("receiver n = %d\n", n);
    // while(true)
    for (int i=0;i<Nepisodes;i++)
    {
      std::vector<double> state(state_vars);
      // receive initial state
      MPI_Request request;
      MPI_Irecv(dbuf, state_vars, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &request);
      MPI_Wait( &request, &status);
      // MPI_Irecv(dbuf, state_vars, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &request);
      // MPI_Wait( &request, &status);
      for (int i=0;i<state_vars;i++) state[i]=dbuf[i];
      printf("receive state = %f %f %f %f %f %f \n", state[0] ,state[1], state[2] , state[3] ,state[4], state[5]);

      for (int j=0; j <N_timestep; j++)
      { 

        std::vector<double> action =getAction(state,control_vars);
        for (int i=0;i<control_vars;i++) dbufa[i]=action[i];
        MPI_Request request_action;
        MPI_Isend(dbufa, control_vars, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &request_action); // send action
        printf("send action = %f  \n", action[0]);
        MPI_Wait(&request_action, MPI_STATUSES_IGNORE);

        MPI_Request request_ob_rew_term;
        MPI_Irecv(dbufsrt, state_vars+2, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &request_ob_rew_term); // receive state, reward, termination
        MPI_Wait(&request_ob_rew_term, &status);
        // MPI_Irecv(dbufsrt, state_vars+2, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &request);
        // MPI_Wait( &request, &status);
        for (int i=0;i<state_vars;i++) state[i]=dbufsrt[i];
        float reward=dbufsrt[state_vars];
        bool terminal = true;
        if (abs(dbufsrt[state_vars+1]) < 1E-3){
          terminal=false;
        }
        if (terminal) break;
      }  //end of simulation loop
    }// end of train loop
  }
  MPI_Finalize();
  return 0;
}