// #pragma once
#ifndef ppo_nn
#define ppo_nn
#include <torch/torch.h>

#include <iostream>
// #include "eigenmvn.h"
#include <math.h>
#include <time.h>

#include <random>
#include <vector>
// using namespace std;

enum TRAIN_STATUS { NORMAL = 0, TERMINATE, TIMEUP, START };
// std::random_device rd_NN;
// std::mt19937_64 eng_NN(rd_NN());
// std::uniform_int_distribution<unsigned long> distr_NN;

class MemoryNN {
 public:
  std::vector<torch::Tensor> actions, states, logprobs;
  std::vector<double> rewards;
  std::vector<bool> is_terminals;
  std::vector<bool> is_timeups;
  void push_reward(double reward, bool terminate, bool timeup);
  void merge(MemoryNN& r);
  void clear();
};

// template<typename Scalar>
// torch::Tensor eigenToTensor(Eigen::Matrix<Scalar, Eigen::Dynamic, -1> mat){
//     torch::Tensor out = torch::randn({mat.rows(), mat.cols()});
//     int i = 0, j = 0;
//     for(auto row: mat.rowwise()){
//         j = 0;
//         for(auto elem: row){
//             out[i][j] = elem;
//             j++;
//         }
//         i++;
//     }
//     return out;
//     // return torch::Tensor();
// }

// // template<typename Scalar>
// Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
// tensorToMatrix(torch::Tensor& in){
//     // std::cout << "Tensor to matrix begin" << std::endl;
//     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> out;
//     int d1 = in.sizes()[0];
//     int d2 = (in.sizes().size()>1?in.sizes()[1]:1);
//     out.resize(d1, d2);

//     for(int i = 0; i < out.rows(); i++){
//         for(int j = 0; j < out.cols(); j++){
//             out(i, j) = in.index({i, j}).item().toDouble();
//         }
//     }
//     // std::cout << "Tensor to matrix end" << std::endl;
//     return out;
// }
// // template<typename Scalar>
// Eigen::Matrix<double, Eigen::Dynamic, 1> tensorToVector(torch::Tensor& in){
//     // std::cout << "Tensor to vector begin" << std::endl;
//     Eigen::Matrix<double, Eigen::Dynamic, 1> out;
//     int d1 = in.sizes()[0];

//     out.resize(d1, 1);
//     for(int i = 0; i < out.rows(); i++){
//             out(i,0) = in.index({i}).item().toDouble();
//     }
//     // std::cout << "Tensor to vector end" << std::endl;
//     return out;

// }

// TODO: make it gpu
struct ActorCritic : torch::nn::Module {
  ActorCritic() {}
  ActorCritic(int state_dim, int action_dim, double action_std)
      : actor(register_module(
            "actor", torch::nn::Sequential(
                         {{"linear1", torch::nn::Linear(state_dim, 64)},
                          {
                              "tanh1",
                              torch::nn::Tanh(),
                          },
                          {
                              "linear2",
                              torch::nn::Linear(64, 32),
                          },
                          {
                              "tanh2",
                              torch::nn::Tanh(),
                          },
                          {
                              "linear3",
                              torch::nn::Linear(32, action_dim),
                          },
                          {"tanh3", torch::nn::Tanh()}}))),
        critic(register_module(
            "critic", torch::nn::Sequential({

                          {"linear1", torch::nn::Linear(state_dim, 64)},
                          {
                              "tanh1",
                              torch::nn::Tanh(),
                          },
                          {
                              "linear2",
                              torch::nn::Linear(64, 32),
                          },
                          {
                              "tanh2",
                              torch::nn::Tanh(),
                          },
                          {
                              "linear3",
                              torch::nn::Linear(32, 1),
                          },
                      }))) {
    int64_t dims[] = {action_dim};
    action_var = torch::full(torch::IntArrayRef(dims, 1),
                             torch::Scalar(action_std * action_std));
    // std::cout << "action_var: " << action_var.grad_fn()->name() << std::endl;
    this->action_std = action_std;
  }

  auto act(torch::Tensor state, MemoryNN& MemoryNN);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate(
      torch::Tensor state, torch::Tensor action);

  torch::nn::Sequential actor;
  torch::nn::Sequential critic;
  torch::Tensor action_var;
  double action_std;
};

class PPO {
  // Dummy constructor for initializing

 public:
  PPO(int64_t state_dim, int64_t action_dim, double action_std, double lr,
      std::tuple<double, double> betas, double gamma, int64_t K_epochs,
      double eps_clip)
      : lr(lr),
        betas(betas),
        gamma(gamma),
        eps_clip(eps_clip),
        K_epochs(K_epochs) {
    // this->lr = lr;
    // this->betas = betas;
    // this->gamma = gamma;
    // this->eps_clip = eps_clip;
    // this->K_epochs = K_epochs;
    std::cout << "lr: " << this->lr << std::endl;
    // std::cout << "betas: " << this->betas << std::endl;
    std::cout << "gamma: " << this->gamma << std::endl;
    std::cout << "eps_clip: " << this->eps_clip << std::endl;
    std::cout << "K_epochs: " << this->K_epochs << std::endl;

    this->policy = ActorCritic(state_dim, action_dim, action_std);
    auto adamOptions = torch::optim::AdamOptions(this->lr);
    // adamOptions.betas()
    adamOptions.betas(betas);
    this->optimizer =
        new torch::optim::Adam(this->policy.parameters(), adamOptions);

    this->MseLoss = torch::nn::MSELoss();
  }

  auto select_action(torch::Tensor state, MemoryNN& MemoryNN);
  void update(MemoryNN MemoryNN);

  double lr, gamma, eps_clip;
  std::tuple<double, double> betas;
  int64_t K_epochs;
  // ActorCritic policy, policy_old;
  ActorCritic policy;
  torch::optim::Adam* optimizer;
  torch::nn::MSELoss MseLoss;
};
std::tuple<std::vector<float>, float> getAction(std::vector<float> observation,
                                                int dim, PPO ppo,
                                                MemoryNN& memoryNN);
#endif
