//
//  main.cpp
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//


#include "cart-pole.h"
#include "custom_action.h"
#include <iostream>
#include <cstdio>

inline void app_main(smarties::Communicator*const comm, int argc, char**argv)
{
  const int control_vars = 1; // force along x
  const int state_vars = 6; // x, vel, angvel, angle, cosine, sine
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
  std::vector<bool> b_observable = {true, true, true, false, true, true};
  //std::vector<bool> b_observable = {true, false, false, false, true, true};
//   comm->setStateObservable(b_observable);
  //comm->setIsPartiallyObservable();

  CartPole env;

  while(true) //train loop
  {
    env.reset(comm->getPRNG()); // prng with different seed on each process
    comm->sendInitState(env.getState()); //send initial state

    while (true) //simulation loop
    {
      std::vector<double> action = comm->recvAction();
      if(comm->terminateTraining()) return; // exit program

      bool poleFallen = env.advance(action); //advance the simulation:

      std::vector<double> state = env.getState();
      double reward = env.getReward();

      if(poleFallen) { //tell smarties that this is a terminal state
        comm->sendTermState(state, reward);
        break;
      } else comm->sendState(state, reward);
    }
  }
}

int main(int argc, char**argv)
{
  
}