#include <mpi.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "cart_MPI.h"
#include "cart_pole.h"
#include "ppo_nn.h"
inline void env_run(int myid, MPI_Comm& env_comm) {
  printf("environment running on process myid: %d \n", myid);
  // OPTIONAL: action bounds
  std::ofstream myfile;
  myfile.open("./proc" + std::to_string(myid) + "_log.txt");
  bool bounded = true;
  std::vector<double> upper_action_bound{10}, lower_action_bound{-10};
  int timestep = 0;
  CartPole env;

  // while(true) //train loop
  for (int i = 0; i < (int)(Nepisodes / (nprocs - 1)); i++) {
    std::cout << "*****NEW EPISODE!*****"
              << "myid: " << myid << "episode: " << i << std::endl;
    double episode_reward = 0.0;
    env.reset();  // prng with different seed on each process
    // send initial obs
    auto obs_raw = env.getobs();
    std::cout << "OBS RAW SIZE" << obs_raw.size() << std::endl;
    // make sure the data type is float
    // std::cout << "ORIGINAL" << obs_raw << typeid(obs_raw[0]).name() <<
    // std::endl;
    std::vector<float> obs(obs_raw.begin(), obs_raw.end());
    // std::cout << "NEW" << obs << typeid(obs[0]).name() << std::endl;

    obs.push_back(0);
    obs.push_back(START);
    MPI_Send(obs.data(), obs_vars + 2, MPI_FLOAT, NNnode, myid, MPI_COMM_WORLD);
    printf("Env %d send obs = %f %f %f %f %f %f \n", myid, obs[0], obs[1],
           obs[2], obs[3], obs[4], obs[5]);

    // while (true) //simulation loop
    for (int j = 0; j < N_timestep; j++) {
      timestep++;
      std::vector<float> action(control_vars);
      MPI_Recv(action.data(), control_vars, MPI_FLOAT, NNnode,
               myid + nprocs * 2, MPI_COMM_WORLD, &status);  // receive action
      printf("Env %d received action = %f  \n", myid, action[0]);
      if (isnan(action[0])) {
        printf("nan bug!!!");
        std::cout << "Observation that led to problem is:" << ' ' << obs
                  << std::endl;
        exit(1);
      }
      // std::cout << "action before scaling:" << ' ' << action << std::endl;
      for (int k = 0; k < action.size(); k++)
        action[k] = (action[k] + 1) *
                        (upper_action_bound[k] - lower_action_bound[k]) / 2 +
                    lower_action_bound[k];
      std::cout << "$$$action after scaling:" << ' ' << action << std::endl;
      // std::vector<double> action(action,action+control_vars);

      if (action[0] == INVALIDACTION) {
        printf("environment node %d done \n", myid);
        return;
      }
      // convert to double before putting into environment
      std::vector<double> action_double(action.begin(), action.end());

      bool terminate = env.advance(action_double);  // advance the simulation:
      obs_raw = env.getobs();
      double reward = env.getReward();
      episode_reward += reward;
      // std::cout << terminate << std::endl;
      // send new observation, reward, and whether terminate or not, if
      // terminate send 1, if not send 0 for (int i=0;i<obs_vars;i++)
      // dbufsrt[i]=obs[i];
      std::vector<float> obs(obs_raw.begin(), obs_raw.end());
      obs.push_back(reward);
      if (terminate) {
        obs.push_back(TERMINATE);

        printf("myid: %d episode: %d TERMINATED!! \n", myid, i);
        // break;
      } else if (j == (N_timestep - 1)) {
        std::cout << "TIMEUP" << std::endl;
        obs.push_back(TIMEUP);
        // MPI_Send(obs.data(), obs_vars+2, MPI_FLOAT, NNnode, myid,
        // MPI_COMM_WORLD);
      } else {
        obs.push_back(NORMAL);
        // MPI_Send(obs.data(), obs_vars+2, MPI_FLOAT, NNnode, myid,
        // MPI_COMM_WORLD);
      }
      printf("Env %d send obs = %f %f %f %f %f %f \n", myid, obs[0], obs[1],
             obs[2], obs[3], obs[4], obs[5]);
      // std::cout << timestep << std::endl;
      if (timestep >= update_pre_timestep) {
        std::cout << "begin barrier my process :" << myid
                  << "timestep: " << timestep << std::endl;

        if (myid != 1) {
          MPI_Send(obs.data(), obs_vars + 2, MPI_FLOAT, NNnode, myid,
                   MPI_COMM_WORLD);
          if (terminate || (j == (N_timestep - 1))) {
            MPI_Barrier(env_comm);
            std::cout << "After barrierA1" << std::endl;
          }

          if (terminate || (j == (N_timestep - 1))) {
            MPI_Barrier(env_comm);
            std::cout << "After barrierA2" << std::endl;
            timestep = 0;
          }
        } else {
          if (terminate || (j == (N_timestep - 1))) {
            MPI_Barrier(env_comm);
            std::cout << "After barrierB1" << std::endl;
          }
          // MPI_Barrier(env_comm);
          MPI_Send(obs.data(), obs_vars + 2, MPI_FLOAT, NNnode, myid,
                   MPI_COMM_WORLD);
          if (terminate || (j == (N_timestep - 1))) {
            MPI_Barrier(env_comm);
            std::cout << "After barrierB2" << std::endl;
            timestep = 0;
          }
        }
      } else {
        MPI_Send(obs.data(), obs_vars + 2, MPI_FLOAT, NNnode, myid,
                 MPI_COMM_WORLD);
      }

      if (terminate) break;
    }  // end of simulation loop
    printf("------myid: %d episode: %d total reward: %f\n", myid, i,
           episode_reward);
    myfile << "myid: " << myid;
    myfile << "\t episode:" << i;
    myfile << "\t reward:";
    myfile << std::fixed << std::setprecision(8) << episode_reward << std::endl;

  }  // end of train loop
  printf("environment node %d done \n", myid);
  myfile.close();
  return;
}

std::string status_to_string(TRAIN_STATUS status) {
  /*enum TRAIN_STATUS {NORMAL = 0, TERMINATE, TIMEUP, START};*/
  if (status == TERMINATE) return "TERMINATE";
  if (status == NORMAL) return "NORMAL";
  if (status == TIMEUP) return "TIMEUP";
  if (status == START)
    return "START";
  else {
    std::cout << "INVALID STATUS: " << (int)status << std::endl;
    return "INVALID";
  }
}

// inline void respond_action(int envid, Memory& mem, MemoryNN& memNN, PPO ppo,
// bool end, int& n_ep, float dbufsrt[]){
inline void respond_to_env(int envid, MemoryNN& memNN, PPO ppo, bool end,
                           std::vector<float> obs_and_more,
                           std::vector<float>& envAction) {
  std::vector<float> observation(obs_and_more.begin(), obs_and_more.end() - 2);
  std::cout << "$$$Observation: " << observation << ' '
            << "reward:" << obs_and_more[obs_and_more.size() - 2] << ' '
            << "train_status:" << obs_and_more.back() << std::endl;
  auto [action, logprobs] = getAction(
      observation, control_vars, ppo,
      memNN);  // here is a C++17 functionality
               // https://stackoverflow.com/questions/321068/returning-multiple-values-from-a-c-function
  std::cout << "Direct action and logprob: " << action << ' ' << logprobs
            << std::endl;
  // TODO generalize learner and memory (class learner as the base class for ppo
  // etc.) ppo.update_memory()
  if (end) action[0] = INVALIDACTION;

  std::cout << "sending action " << action << ' ' << "to" << ' ' << envid
            << std::endl;
  envAction = action;
  MPI_Request req;
  MPI_Isend(envAction.data(), control_vars, MPI_FLOAT, envid,
            envid + nprocs * 2, MPI_COMM_WORLD, &req);  // send action
}

inline void NN_run() {
  int n_ep = 0;
  int n_timestep = 0;
  bool end = false;
  // MPI_Request reqs[nprocs-1];

  auto action_std =
      0.5;  // constant std for action distribution (Multivariate Normal)
  auto K_epochs = 80;   // update policy for K epochs
  auto eps_clip = 0.2;  // clip parameter for PPO
  auto gamma = 0.99;    // discount factor

  auto lr = 0.0003;  // parameters for Adam optimizer
  auto betas = std::make_tuple(0.9, 0.999);

  PPO ppo = PPO(obs_vars, control_vars, action_std, lr, betas, gamma, K_epochs,
                eps_clip);

  MemoryNN memNN[nprocs - 1];
  TRAIN_STATUS env_status;

  std::vector<std::vector<float>> obs_and_more(
      nprocs, std::vector<float>(obs_vars + 2));
  std::vector<std::vector<float>> env_actions(nprocs, std::vector<float>());
  MPI_Request recv[nprocs];

  for (int i = 0; i < nprocs; i++) recv[i] = MPI_REQUEST_NULL;
  while (true) {
    //  std::cout << "Iteration" << std::endl;
    //  for (int i = 1; i <= nprocs-1; i++){
    //     MPI_Irecv(obs_and_more[i].data(), obs_vars+2, MPI_FLOAT, i, i,
    //     MPI_COMM_WORLD, &recv[i]);
    //  }

    int nProcessed = 0;

    // std::cout << nProcessed << std::endl;
    for (int i = 1; i <= nprocs - 1; i++) {
      int flag = -1;
      if (recv[i] == MPI_REQUEST_NULL) {
        MPI_Irecv(obs_and_more[i].data(), obs_vars + 2, MPI_FLOAT, i, i,
                  MPI_COMM_WORLD, &recv[i]);
        continue;
      }
      MPI_Test(&recv[i], &flag, &status);

      // MPI_Status status;
      if (flag) {
        std::cout << "REQUEST IS: " << (int)(recv[i] == MPI_REQUEST_NULL)
                  << std::endl;
        // MPI_Test(&r[i], &flag, &status);

        std::cout << "Flag is: " << flag << std::endl;
        // MPI_Recv(obs_and_more.data(), obs_vars+2, MPI_FLOAT, i, i,
        // MPI_COMM_WORLD, &status);
        nProcessed++;
        printf("received observations and more from %d \n", i);

        float reward = obs_and_more[i][obs_vars];
        env_status = static_cast<TRAIN_STATUS>(
            int(obs_and_more[i][obs_vars + 1] + 1E-3));
        std::cout << "Env status is: " << status_to_string(env_status)
                  << std::endl;
        std::cout << "EQ: " << (int)(env_status == TERMINATE) << std::endl;
        // TODO: unify the expressions of training state, both here and in
        // memory, using int instead of bool
        if (env_status != START) {
          std::cout << "PUSHPUSHPUSH!!!" << std::endl;
          memNN[i - 1].push_reward(reward, (env_status == TERMINATE),
                                   (env_status == TIMEUP));
          n_timestep++;
          std::cout << "Timestep " << n_timestep << std::endl;
        }

        if (env_status == TERMINATE || env_status == TIMEUP) {
          n_ep++;
          std::cout << "In terminate condition " << n_timestep << " "
                    << updateTimestep << std::endl;
          if (n_timestep >= updateTimestep) {
            std::cout << "UPDATING " << n_timestep << std::endl;
            MemoryNN mergedMemory;
            bool shouldUpdate = true;
            for (int proc = 1; proc < nprocs; proc++) {
              std::cout << "proc" << ' ' << proc << ' '
                        << "States in memory:" << memNN[proc - 1].states.size()
                        << '\n'
                        << "proc" << ' ' << proc << ' ' << "Actions in memory:"
                        << memNN[proc - 1].actions.size() << '\n'
                        << "proc" << ' ' << proc << ' ' << "Rewards in memory:"
                        << memNN[proc - 1].rewards.size() << std::endl;
              std::cout << "begining merging memory from " << proc << std::endl;
              if (memNN[proc - 1].states.size() !=
                      memNN[proc - 1].actions.size() ||
                  memNN[proc - 1].states.size() !=
                      memNN[proc - 1].rewards.size()) {
                shouldUpdate = false;
              }
            }
            std::cout << "Should update: " << shouldUpdate << std::endl;
            if (shouldUpdate) {
              for (int proc = 1; proc < nprocs; proc++) {
                std::cout << "proc" << ' ' << proc << ' ' << "States in memory:"
                          << memNN[proc - 1].states.size() << '\n'
                          << "proc" << ' ' << proc << ' '
                          << "Actions in memory:"
                          << memNN[proc - 1].actions.size() << '\n'
                          << "proc" << ' ' << proc << ' '
                          << "Rewards in memory:"
                          << memNN[proc - 1].rewards.size() << std::endl;
                std::cout << "begining merging memory from " << proc
                          << std::endl;
                mergedMemory.merge(memNN[proc - 1]);
                memNN[proc - 1].clear();
              }
              ppo.update(mergedMemory);
              n_timestep = 0;
            }
          }
        } else
          respond_to_env(i, memNN[i - 1], ppo, end, obs_and_more[i],
                         env_actions[i]);

        std::cout << "#########################################################"
                     "#################################"
                  << std::endl;
        // std::cout << "After respond action, the memory is: " <<
        // memNN[i-1].states << std::endl;
      }
    }

    // printf("total: Nepisodes: %d ; Ntimestep: %d \n" ,n_ep, n_timestep);

    // for (int i=1;i<=nprocs-1;i++){
    //   MPI_Irecv(dbufsrt, obs_vars+2, MPI_DOUBLE, i, i, MPI_COMM_WORLD,
    //   &reqs[i-1]);
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

    if (end or (n_ep >= Nepisodes)) {
      printf("NN node exit");
      // printf("check: memory length of 1's environment is %ld",
      // mem[0].action_list.size());
      break;
    }
    if (n_timestep >= Max_timestep) {
      end = true;
    }
  }
}

inline void read_parameters(std::string path) {
  std::ifstream config_file;
  config_file.open(path);
  if (config_file) {
    std::cout << "reading config file" << std::endl;
    std::string line;
    int number;
    std::string::size_type sz;
    std::vector<int> numbers;
    while (getline(config_file, line)) {
      std::cout << line << std::endl;

      number = std::stoi(line, &sz);
      std::cout << number << std::endl;
      numbers.push_back(number);
    }
    control_vars = numbers[0];
    obs_vars = numbers[1];
    Nepisodes = numbers[2];
    N_timestep = numbers[3];
    Max_timestep = numbers[4];
    updateTimestep = numbers[5];

  } else {
    std::cout << "config file not found" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  int myid;
  int n;
  std::string config_file_path;
  // std::cout << "argc:   " << argc << std::endl;
  // std::cout << "argv[0]:   " << argv[0] << std::endl;
  if (argc == 1) {
    config_file_path = "../config";
  } else if (argc == 2) {
    std::cout << "config file directory:   " << argv[1] << std::endl;
    config_file_path = argv[1];
  } else {
    std::cout
        << "Warning: too many arguments, redundent arguments are not used "
        << std::endl;
  }

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (nprocs < 2){
    std::cout << "You need to assign this program at least two processes" << std::endl;
  }
  std::cout << "update_pre_timestep: " << update_pre_timestep << std::endl;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  // if (myid == 0) {
  read_parameters(config_file_path);
  update_pre_timestep = (updateTimestep - N_timestep) / (nprocs - 1);
  std::cout << "control_vars: " << control_vars << std::endl;
  std::cout << "obs_vars: " << obs_vars << std::endl;
  std::cout << "Nepisodes: " << Nepisodes << std::endl;
  std::cout << "N_timestep: " << N_timestep << std::endl;
  std::cout << "Max_timestep: " << Max_timestep << std::endl;
  std::cout << "updateTimestep: " << updateTimestep << std::endl;
  std::cout << "update_pre_timestep: " << update_pre_timestep << std::endl;
  // }
  // MPI_Barrier(MPI_COMM_WORLD);

  MPI_Comm env_comm;
  // myid == 0 for NN update, otherwise for (myid)th environment
  MPI_Comm_split(MPI_COMM_WORLD, myid == 0, myid, &env_comm);
  int sub_procs;
  MPI_Comm_size(env_comm, &sub_procs);
  std::cout << "Sub procs: " << myid << "/" << sub_procs << std::endl;

  if (myid == 0) {
    printf("There are %d processes running in this MPI program\n", nprocs);
    NN_run();
  } else {
    env_run(myid, env_comm);
  }
  MPI_Finalize();
  return 0;
}