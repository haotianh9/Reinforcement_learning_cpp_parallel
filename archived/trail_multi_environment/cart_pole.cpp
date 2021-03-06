//
//  main.cpp
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//

// compile: mpic++  -o cart cart_pole.cpp -fopenmp
// running: mpirun -n N ./cart if don't change other parameters, using N=4 is fine now

#include "cart_pole.h"
#include "cart_MPI.h"
#include "custom_action.h"
#include "Memory.h"
#include <iostream>
#include <cstdio>
#include "mpi.h"
#include <unistd.h>
inline void env_run(int myid)
{
  
  printf("environment running on process myid: %d \n", myid);
  //OPTIONAL: action bounds
  bool bounded = true;
  std::vector<double> upper_action_bound{10}, lower_action_bound{-10};

  CartPole env;

  // while(true) //train loop
  for (int i=0;i<(int)(Nepisodes/(nprocs-1));i++)
  {
    printf("myid: %d episoe: %d \n", myid,i);
    env.reset(); // prng with different seed on each process
    //send initial obs
    std::vector<double> obs = env.getobs();
    // for (int i=0;i<obs_vars;i++) dbuf[i]=obs[i];
    std::copy(obs.begin(), obs.end(), dbufsrt);
    dbufsrt[obs_vars]=0;
    dbufsrt[obs_vars+1]=3;
    MPI_Send(dbufsrt, obs_vars+2, MPI_DOUBLE, NNnode, myid, MPI_COMM_WORLD);
    printf("%d send obs = %f %f %f %f %f %f \n",myid, obs[0] ,obs[1], obs[2] , obs[3] ,obs[4], obs[5]);
    

    // while (true) //simulation loop
    for (int j=0; j <N_timestep; j++)
    {
      MPI_Recv(dbufa, control_vars, MPI_DOUBLE, NNnode, myid+nprocs*2, MPI_COMM_WORLD, &status); // receive action
      std::vector<double> action(dbufa,dbufa+control_vars);
      printf("%d receive action = %f  \n", myid, action[0]);
      if (action[0] == invalidaction){
        printf("environment node %d done \n", myid);
        return;
      }
      bool terminate = env.advance(action); //advance the simulation:
      obs = env.getobs();
      double reward = env.getReward();
      // send new observation, reward, and whether terminate or not, if terminate send 1, if not send 0
      // for (int i=0;i<obs_vars;i++) dbufsrt[i]=obs[i];
      std::copy(obs.begin(), obs.end(), dbufsrt);
      dbufsrt[obs_vars+1]=reward;
      if(terminate){
        dbufsrt[obs_vars+1]=1;
        MPI_Send(dbufsrt, obs_vars+2, MPI_DOUBLE, NNnode, myid, MPI_COMM_WORLD);  
        break;
      }else if (j == (N_timestep-1)){
        dbufsrt[obs_vars+1]=2;
        MPI_Send(dbufsrt, obs_vars+2, MPI_DOUBLE, NNnode, myid, MPI_COMM_WORLD);  
      }else{
        dbufsrt[obs_vars+1]=0;
        MPI_Send(dbufsrt, obs_vars+2, MPI_DOUBLE, NNnode, myid, MPI_COMM_WORLD);  
      }
    }  //end of simulation loop
  }// end of train loop
  printf("environment node %d done \n", myid);
  return;
}

inline void respond_action(int envid, Memory& mem, bool end, int& n_ep, double dbufsrt[]){
  std::vector<double> obs(dbufsrt,dbufsrt+obs_vars);
  auto [action,logprobs] =getAction(obs,control_vars); // here is a C++17 functionality https://stackoverflow.com/questions/321068/returning-multiple-values-from-a-c-function
  mem.push_obs_act(obs,action,logprobs);
  std::copy(action.begin(), action.end(), dbufa);
  if (end){
    dbufa[0]=invalidaction;
  }
  MPI_Send(dbufa, control_vars, MPI_DOUBLE, envid, envid+nprocs*2, MPI_COMM_WORLD); // send action
  printf("send action to %d = %f  \n", envid , action[0]);
  float reward=dbufsrt[obs_vars];
  bool terminate = false;
  bool done =false;
  bool start =false;
  if (std::abs(dbufsrt[obs_vars+1]-1) < 1E-3){
    terminate=true;
    n_ep++;
  }if (std::abs(dbufsrt[obs_vars+1]-2) < 1E-3){
    done=true;
    n_ep++;
  }if (std::abs(dbufsrt[obs_vars+1]-3) < 1E-3){
    start=true;
  }
  if (!start){
    mem.push_reward(reward,terminate,done);
  }

}

inline void NN_run(){
  int n_ep=0;
  int n_timestep=0;
  bool end=false;
  Memory mem[nprocs-1]; // an array of memorys, each memory object for each environment process
  // MPI_Request reqs[nprocs-1];
  while(true){

    for (int i=1;i<=nprocs-1;i++){
      MPI_Recv(dbufsrt, obs_vars+2, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
      respond_action(i,mem[i-1],end,n_ep,dbufsrt);
      n_timestep++;
    }
    printf("total: Nepisodes: %d ; Ntimestep: %d \n" ,n_ep, n_timestep);

    // for (int i=1;i<=nprocs-1;i++){
    //   MPI_Irecv(dbufsrt, obs_vars+2, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &reqs[i-1]);  
    // }
    // int completed[nprocs-1];
    // int comp_num=0;
    // while(true){
    //   for (int i=1;i<=nprocs-1;i++){
    //     if (~completed[i-1]){
    //       MPI_Test(&reqs[i-1],&completed[i-1],MPI_STATUS_IGNORE);
    //       if(completed[i-1]){
    //         respond_action(i,mem[i-1],end,n_ep,dbufsrt);
    //         n_timestep++;
    //         comp_num++;
    //       }
    //     }
    //     // usleep(1);
    //   }
    //   if (comp_num == nprocs-1) break;
    // }
    // printf("total: Nepisodes: %d ; Ntimestep: %d \n" ,n_ep, n_timestep);

    if (end or (n_ep >= Nepisodes)){
      printf("NN node exit");
      printf("check: memory length of 1's environment is %ld", mem[0].action_list.size());
      break;
    }
    if (n_timestep >= Max_timestep){
      end=true;
    }
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
    NN_run();
  }
  else {
    env_run(myid);
  }
  MPI_Finalize();
  return 0;
}