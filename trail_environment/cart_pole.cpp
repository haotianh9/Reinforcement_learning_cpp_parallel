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
    MPI_Send(dbuf, state_vars, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD);
    
    printf("send state = %f %f %f %f %f %f \n", state[0] ,state[1], state[2] , state[3] ,state[4], state[5]);
    

    // while (true) //simulation loop
    for (int j=0; j <N_timestep; j++)
    {
      // MPI_Irecv(dbufa, control_vars, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD, &request);
      // MPI_Wait( &request, &status);
      MPI_Recv(dbufa, control_vars, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD, &status); // recieve action
      std::vector<double> action(control_vars);
      for (int i=0;i<control_vars;i++) action[i]=dbufa[i];
      printf("recieve action = %f  \n", action[0]);
      // if(comm->terminateTraining()) return; // exit program

      bool poleFallen = env.advance(action); //advance the simulation:

      std::vector<double> state = env.getState(); 
      double reward = env.getReward();
      // send new observation, reward, and whether terminate or not, if terminate send 1, if not send 0
      for (int i=0;i<state_vars;i++) dbufsrt[i]=state[i];
      dbufsrt[state_vars]=reward;
      if(poleFallen){
        dbufsrt[state_vars+1]=1;
        MPI_Send(dbufsrt, state_vars+2, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD);  
        break;
      }else{
        dbufsrt[state_vars+1]=0;
        MPI_Send(dbufsrt, state_vars+2, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD);  
      }
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
    // printf("reciever n = %d\n", n);
    // while(true)
    for (int i=0;i<Nepisodes;i++)
    {
      std::vector<double> state(state_vars);
      // recieve intial state
      MPI_Recv(dbuf, state_vars, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &status);
      // MPI_Irecv(dbuf, state_vars, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &request);
      // MPI_Wait( &request, &status);
      for (int i=0;i<state_vars;i++) state[i]=dbuf[i];
      printf("recieve state = %f %f %f %f %f %f \n", state[0] ,state[1], state[2] , state[3] ,state[4], state[5]);
      for (int j=0; j <N_timestep; j++)
      { 

        std::vector<double> action =getAction(state,control_vars);
        for (int i=0;i<control_vars;i++) dbufa[i]=action[i];
        MPI_Send(dbufa, control_vars, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD); // send action
        printf("send action = %f  \n", action[0]);
        MPI_Recv(dbufsrt, state_vars+2, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &status); // recieve state, reward, termination
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