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
#include "Memory.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include "mpi.h"
#include <unistd.h>
#include "ppo_nn.h"
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
    double episode_reward=0.0;
    printf("myid: %d episoe: %d \n", myid,i);
    env.reset(); // prng with different seed on each process
    //send initial obs
    auto obs_raw = env.getobs();
    // make sure the data type is float
    // cout << "ORIGINAL" << obs_raw << typeid(obs_raw[0]).name() << endl;
    vector<float> obs(obs_raw.begin(), obs_raw.end());
    // cout << "NEW" << obs << typeid(obs[0]).name() << endl;
    // for (int i=0;i<obs_vars;i++) dbuf[i]=obs[i];
    // std::copy(obs.begin(), obs.end(), dbufsrt);
    // dbufsrt[obs_vars]=0;
    // dbufsrt[obs_vars+1]=3;
    obs.push_back(0); obs.push_back(START);

    MPI_Send(obs.data(), obs_vars+2, MPI_FLOAT, NNnode, myid, MPI_COMM_WORLD);
    printf("%d send obs = %f %f %f %f %f %f \n",myid, obs[0] ,obs[1], obs[2] , obs[3] ,obs[4], obs[5]);

    // while (true) //simulation loop
    for (int j=0; j <N_timestep; j++)
    {
      printf("% d recieved action",myid);
      std::vector<float> action(control_vars);
      MPI_Recv(action.data(), control_vars, MPI_FLOAT, NNnode, myid+nprocs*2, MPI_COMM_WORLD, &status); // recieve action
      if (isnan(action[0])){
        printf("nan bug!!!"); 
        exit(1);
      }
      action[0]*=10;
      // std::vector<double> action(action,action+control_vars);
      printf("%d recieve action = %f  \n", myid, action[0]);
      if (action[0] == INVALIDACTION){
        printf("environment node %d done \n", myid);
        return;
      }
      //convert to double before putting into environment
      std::vector<double> action_double(action.begin(), action.end() );

      bool terminate = env.advance(action_double); //advance the simulation:
      obs_raw = env.getobs();
      double reward = env.getReward();
      episode_reward+=reward;
      // cout << terminate << endl;
      // send new observation, reward, and whether terminate or not, if terminate send 1, if not send 0
      // for (int i=0;i<obs_vars;i++) dbufsrt[i]=obs[i];
      vector<float> obs(obs_raw.begin(), obs_raw.end());
      obs.push_back(reward); 
      if(terminate){
        obs.push_back(TERMINATE);
        MPI_Send(obs.data(), obs_vars+2, MPI_FLOAT, NNnode, myid, MPI_COMM_WORLD); 
        printf("myid: %d episoe: %d terminate !!! \n", myid,i); 
        break;
      }else if (j == (N_timestep-1)){
        obs.push_back(DONE);
        MPI_Send(obs.data(), obs_vars+2, MPI_FLOAT, NNnode, myid, MPI_COMM_WORLD);  
      }else{
        obs.push_back(NORMAL);
        MPI_Send(obs.data(), obs_vars+2, MPI_FLOAT, NNnode, myid, MPI_COMM_WORLD);  
      }
    }  //end of simulation loop
    printf("myid: %d episoe: %d reward: %f\n", myid,i,episode_reward);
    myfile <<"myid: " << myid;
    myfile << "\t episode:"<< i; 
    myfile <<  "\t reward:" ;
    myfile << std::fixed << std::setprecision(8) << episode_reward << endl;

  }// end of train loop
  printf("environment node %d done \n", myid);
  return;
}

// inline void respond_action(int envid, Memory& mem, MemoryNN& memNN, PPO ppo, bool end, int& n_ep, float dbufsrt[]){
inline void respond_action(int envid, MemoryNN& memNN, PPO ppo, bool end, int& n_ep, std::vector<float> obs_and_more){
  cout << "Observation: " << obs_and_more << endl;
  std::vector<float> observation(obs_and_more.begin(), obs_and_more.end() - 2);
  auto [action,logprobs] = getAction(observation,control_vars, ppo, memNN); // here is a C++17 functionality https://stackoverflow.com/questions/321068/returning-multiple-values-from-a-c-function

  // TODO generalize learner and memory (class learner as the base class for ppo etc.)
  // ppo.update_memory()
  
   if (end){
    action[0]=INVALIDACTION;
  }
  MPI_Send(action.data(), control_vars, MPI_FLOAT, envid, envid+nprocs*2, MPI_COMM_WORLD); // send action
  printf("send action to %d = %f  \n", envid , action[0]);
  float reward = obs_and_more[obs_vars];
  bool terminate = false;
  bool done =false;
  
  // cout << "Reward is: " << reward << endl;

  // TODO: unify the expressions of training state, both here and in memory, using int instead of bool
  if (std::abs(obs_and_more[obs_vars+1]-TERMINATE) < 1E-3){
    terminate=true;
    n_ep++;
  }
  if (std::abs(obs_and_more[obs_vars+1]-DONE) < 1E-3){
    done=true;
    n_ep++;
  }
  if (!std::abs(obs_and_more[obs_vars+1]-START) < 1E-3){
    memNN.push_reward(reward, terminate, done);
  }
  // cout << "Obs vars are:  " << dbufsrt[obs_vars+1]-3 << " " << "Start is: " << start << " " << done << " " << " " << terminate <<  endl;

}

inline void NN_run(){
  int n_ep=0;
  int n_timestep=0;
  bool end=false;
  // Memory mem[nprocs-1]; // an array of memorys, each memory object for each environment process
  // MPI_Request reqs[nprocs-1];
  
  auto action_std = 0.2;            // constant std for action distribution (Multivariate Normal)
  auto K_epochs = 80;            // update policy for K epochs
  auto eps_clip = 0.2;            // clip parameter for PPO
  auto gamma = 0.99;            // discount factor
  
  auto lr = 0.0003;                 // parameters for Adam optimizer
  auto betas = make_tuple(0.9, 0.999);
  PPO ppo = PPO(obs_vars, control_vars, action_std, lr, betas, gamma, K_epochs, eps_clip);
  //TODO: merge with Memory?
  MemoryNN memNN[nprocs-1];
  auto updateTimestep = 1900;
  std::vector<float> obs_and_more(obs_vars+2);
  while(true){
    for (int i=1;i<=nprocs-1;i++){
      MPI_Recv(obs_and_more.data(), obs_vars+2, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
      printf("recieve observation from %d",i);
      // respond_action(i,mem[i-1],memNN[i-1], ppo, end,n_ep,dbufsrt);
      respond_action(i,memNN[i-1], ppo, end,n_ep,obs_and_more);
      n_timestep++;
      cout << "Timestep " << n_timestep << endl;
      if(n_timestep%updateTimestep==0){
        cout << "Updating " << n_timestep << endl;
        MemoryNN mergedMemory;
        for(int proc = 0; proc < nprocs-1; proc++){
          cout << "States" << memNN[proc].states.size() << endl;
          mergedMemory.merge(memNN[proc]);
          // memNN[proc].clear();
        }
        ppo.update(mergedMemory);
        n_timestep = 0;
      }  
      
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
      // printf("check: memory length of 1's environment is %ld", mem[0].action_list.size());
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
  myfile.open ("./log.txt");
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
  myfile.close();
  return 0;
}