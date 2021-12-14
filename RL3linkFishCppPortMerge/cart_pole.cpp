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
  std::ofstream myfile;
  myfile.open ("./proc" + std::to_string(nprocs)+"_log.txt");
  bool bounded = true;
  std::vector<double> upper_action_bound{10}, lower_action_bound{-10};

  CartPole env;

  // while(true) //train loop
  for (int i=0;i<(int)(Nepisodes/(nprocs-1));i++)
  {
    cout << "*****NEW EPISODE!*****" 
        << "myid: " << myid << "episode: " << i << endl;
    double episode_reward=0.0;
    env.reset(); // prng with different seed on each process
    //send initial obs
    auto obs_raw = env.getobs();
    // make sure the data type is float
    // cout << "ORIGINAL" << obs_raw << typeid(obs_raw[0]).name() << endl;
    vector<float> obs(obs_raw.begin(), obs_raw.end());
    // cout << "NEW" << obs << typeid(obs[0]).name() << endl;

    obs.push_back(0); obs.push_back(START);
    MPI_Send(obs.data(), obs_vars+2, MPI_FLOAT, NNnode, myid, MPI_COMM_WORLD);
    // printf("Env %d send obs = %f %f %f %f %f %f \n",myid, obs[0] ,obs[1], obs[2] , obs[3] ,obs[4], obs[5]);
    
    // while (true) //simulation loop
    for (int j=0; j <N_timestep; j++)
    {
      std::vector<float> action(control_vars);
      MPI_Recv(action.data(), control_vars, MPI_FLOAT, NNnode, myid+nprocs*2, MPI_COMM_WORLD, &status); // receive action
      printf("Env %d received action = %f  \n", myid, action[0]);
      if (isnan(action[0])){
        printf("nan bug!!!"); 
        cout << "Observation that led to problem is:" << ' ' << obs << endl;
        exit(1);
      }
      // cout << "action before scaling:" << ' ' << action << endl;
      for (int k = 0; k < action.size(); k++) 
        action[k] = (action[k]+1)*(upper_action_bound[k] - lower_action_bound[k])/2 + lower_action_bound[k];
      cout << "$$$action after scaling:" << ' ' << action << endl;
      // std::vector<double> action(action,action+control_vars);

      if (action[0] == INVALIDACTION){
        printf("environment node %d done \n", myid);
        return;
      }
      //convert to double before putting into environment
      std::vector<double> action_double(action.begin(), action.end() );

      bool terminate = env.advance(action_double); //advance the simulation:
      obs_raw = env.getobs();
      double reward = env.getReward();
      episode_reward += reward;
      // cout << terminate << endl;
      // send new observation, reward, and whether terminate or not, if terminate send 1, if not send 0
      // for (int i=0;i<obs_vars;i++) dbufsrt[i]=obs[i];
      vector<float> obs(obs_raw.begin(), obs_raw.end());
      obs.push_back(reward); 
      if(terminate){
        obs.push_back(TERMINATE);
        MPI_Send(obs.data(), obs_vars+2, MPI_FLOAT, NNnode, myid, MPI_COMM_WORLD); 
        printf("myid: %d episode: %d TERMINATED!! \n", myid,i); 
        break;
      }else if (j == (N_timestep-1)){
        obs.push_back(TIMEUP);
        MPI_Send(obs.data(), obs_vars+2, MPI_FLOAT, NNnode, myid, MPI_COMM_WORLD);  
      }else{
        obs.push_back(NORMAL);
        MPI_Send(obs.data(), obs_vars+2, MPI_FLOAT, NNnode, myid, MPI_COMM_WORLD);  
      }
    }  //end of simulation loop
    printf("------myid: %d episode: %d total reward: %f\n", myid,i,episode_reward);
    myfile <<"myid: " << myid;
    myfile << "\t episode:"<< i; 
    myfile <<  "\t reward:" ;
    myfile << std::fixed << std::setprecision(8) << episode_reward << endl;

  }// end of train loop
  printf("environment node %d done \n", myid);
  myfile.close();
  return;
}

// inline void respond_action(int envid, Memory& mem, MemoryNN& memNN, PPO ppo, bool end, int& n_ep, float dbufsrt[]){
inline void respond_to_env(int envid, MemoryNN& memNN, PPO ppo, bool end, std::vector<float> obs_and_more){
  
  std::vector<float> observation(obs_and_more.begin(), obs_and_more.end() - 2);
  cout << "$$$Observation: " << observation << ' '
        << "reward:" << obs_and_more[obs_and_more.size() - 2] << ' '
        << "train_status:" << obs_and_more.back() << endl;
  auto [action,logprobs] = getAction(observation,control_vars, ppo, memNN); // here is a C++17 functionality https://stackoverflow.com/questions/321068/returning-multiple-values-from-a-c-function
  cout << "Direct action and logprob: " << action << ' ' << logprobs << endl;
  // TODO generalize learner and memory (class learner as the base class for ppo etc.)
  // ppo.update_memory()
  if (end) action[0]=INVALIDACTION;
  
  cout << "sending action " << action << ' '
       << "to" << ' ' << envid << endl;
  MPI_Send(action.data(), control_vars, MPI_FLOAT, envid, envid+nprocs*2, MPI_COMM_WORLD); // send action
  
}

inline void NN_run(){
  int n_ep = 0;
  int n_timestep = 0;
  bool end = false;
  // MPI_Request reqs[nprocs-1];
  
  auto action_std = 0.5;            // constant std for action distribution (Multivariate Normal)
  auto K_epochs = 80;            // update policy for K epochs
  auto eps_clip = 0.2;            // clip parameter for PPO
  auto gamma = 0.99;            // discount factor
  
  auto lr = 0.0003;                 // parameters for Adam optimizer
  auto betas = make_tuple(0.9, 0.999);

  PPO ppo = PPO(obs_vars, control_vars, action_std, lr, betas, gamma, K_epochs, eps_clip);
  
  MemoryNN memNN[nprocs-1];
  TRAIN_STATUS env_status;
  
  std::vector<float> obs_and_more(obs_vars+2);
  while(true){
    for (int i = 1; i <= nprocs-1; i++){
      MPI_Recv(obs_and_more.data(), obs_vars+2, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
      printf("received observations and more from %d \n",i);
    

      float reward = obs_and_more[obs_vars];
      env_status = static_cast<TRAIN_STATUS>(int(obs_and_more[obs_vars+1]+1E-3));
      // TODO: unify the expressions of training state, both here and in memory, using int instead of bool
      if (env_status != START){
        cout << "PUSHPUSHPUSH!!!" <<endl;
        memNN[i-1].push_reward(reward, (env_status == TERMINATE), (env_status == TIMEUP));
        n_timestep++;
        cout << "Timestep " << n_timestep << endl;
      }
      if (env_status == TERMINATE || env_status == TIMEUP)
      {
        n_ep++;
        if(n_timestep >= updateTimestep){
          cout << "UPDATING " << n_timestep << endl;
          MemoryNN mergedMemory;
          for(int proc = 1; proc < nprocs; proc++){
            cout << "proc" << ' ' << proc << ' ' << "States in memory:" << memNN[proc-1].states << '\n'
                << "proc" << ' ' << proc << ' ' << "Actions in memory:" << memNN[proc-1].actions << '\n'
                << "proc" << ' ' << proc << ' ' << "Rewards in memory:" << memNN[proc-1].rewards << endl;
            mergedMemory.merge(memNN[proc-1]);
            memNN[proc-1].clear();
          }
          ppo.update(mergedMemory);
          n_timestep = 0;
        }
      }
      else respond_to_env(i,memNN[i-1], ppo, end, obs_and_more);

      
      cout << "##########################################################################################" << endl;
      // cout << "After respond action, the memory is: " << memNN[i-1].states << endl;

      
      
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
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  // myfile.open ("./proc" + std::to_string(nprocs)+"_log.txt");
  if (myid == 0) {
    printf("There are %d processes running in this MPI program\n", nprocs);
    NN_run();
  }
  else {
    env_run(myid);
  }
  MPI_Finalize();
  //myfile.close();
  return 0;
}