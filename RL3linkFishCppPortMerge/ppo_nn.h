#ifndef ppo_nn
#define ppo_nn
#include <torch/torch.h>
#include <iostream>
// #include "eigenmvn.h"
#include <vector>
#include <math.h>
#include <time.h>
#include <random>
// using namespace std;

enum TRAIN_STATUS {NORMAL = 0, TERMINATE, TIMEUP, START};
std::random_device rd_NN;
std::mt19937_64 eng_NN(rd_NN());
std::uniform_int_distribution<unsigned long> distr_NN;

void printSizes(torch::Tensor& a){
    std::cout << a.sizes()[0] << " " << a.sizes()[1] << " " << a.sizes()[2] << std::endl;
}
class MemoryNN {
    public:
    std::vector<torch::Tensor> actions, states, logprobs;
    std::vector<double> rewards;
    std::vector<bool> is_terminals;
    std::vector<bool> is_timeups;
    auto push_reward(double reward, bool terminate, bool timeup){
        std::cout << "rewards now: " << rewards << std::endl;
        rewards.push_back(reward);
        is_terminals.push_back(terminate);
        is_timeups.push_back(timeup);
    }
    void merge(MemoryNN& r);
    void clear();
};

void MemoryNN::merge(MemoryNN& r){

    std::cout << "States size: " << r.states.size() << " Rewards size: " << r.rewards.size() << " " << std::endl;



    this->actions.insert(this->actions.end(), r.actions.begin(), r.actions.end());
    this->states.insert(this->states.end(), r.states.begin(), r.states.end());
    this->logprobs.insert(this->logprobs.end(), r.logprobs.begin(), r.logprobs.end());
    this->rewards.insert(this->rewards.end(), r.rewards.begin(), r.rewards.end());
    this->is_terminals.insert(this->is_terminals.end(), r.is_terminals.begin(), r.is_terminals.end());
    this->is_timeups.insert(this->is_timeups.end(), r.is_timeups.begin(), r.is_timeups.end());
    std::cout << "Merge successful" << std::endl;
}

void MemoryNN::clear(){
    this->actions.clear();
    this->states.clear();
    this->logprobs.clear();
    this->rewards.clear();
    this->is_terminals.clear();
    this->is_timeups.clear();
}

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

auto multivariateLogProb(torch::Tensor& action_mean, torch::Tensor& covar, torch::Tensor& action){
    // std::cout << covar << std::endl;
    // std::cout << covar.inverse() << std::endl;

    // std::cout << "Action is: " << action << std::endl;
    // std::cout << "Action mean is: " << action_mean << std::endl;
    // std::cout << "covar is: " << covar << std::endl;
    action=action.unsqueeze(1);
    // printSizes(action);
    // printSizes(action_mean);
    // printSizes(covar);
    auto diff = (action - action_mean).unsqueeze(2);
    // printSizes(diff);
    // std::cout << "diff is: " << diff << std::endl;

    auto covarInverse = covar.inverse();
    // printSizes(covarInverse);

    // auto a=covarInverse.matmul( diff);
    // std::cout << "a" << std::endl;
    // printSizes(a);
    auto numerator = -0.5 * (diff.transpose(1,2).matmul(covarInverse.matmul( diff)));
    // printSizes(numerator);
    // numerator = torch::exp(numerator);
    // std::cout << "numerator " << numerator << std::endl;
    // auto a=torch::logdet(covar);
    // printSizes(a);
    // float b= log(2 * M_PI) * action_mean.sizes()[1];
    // std::cout << b << std::endl;
    printSizes(action_mean);
    std::cout << "action_mean.sizes()[1]:    " << action_mean.sizes()[1]  << std::endl; 
    auto denominator =   torch::logdet(covar).add(log(2 * M_PI) * action_mean.sizes()[1]);
    denominator = denominator/2;
    // std::cout << "numerator " << std::endl;
    // printSizes(numerator);
    // std::cout << "denominator " << std::endl;
    // printSizes(denominator);
    numerator = numerator.squeeze() - denominator;
    // std::cout << "output " << std::endl;
    // printSizes(numerator);
  

    return numerator;

}
auto multivariateEntropy(int k, torch::Tensor& covar){
    float v1 = pow(2*M_PI*M_E, k);
    // std::cout << covar << std::endl;
    std::cout << "multivariateEntropy" << std::endl;
    //std::cout  << v1 << std::endl;
    // auto a= torch::logdet(covar);
    // printSizes(a);
    // std::cout << a << std::endl;
    
    return  0.5*torch::logdet(covar).add(v1);
}

// // template<typename Scalar>
// Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> tensorToMatrix(torch::Tensor& in){
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

//TODO: make it gpu
struct ActorCritic: torch::nn::Module {
    ActorCritic(){}
    ActorCritic(int state_dim, int action_dim, double action_std) : actor(register_module("actor", torch::nn::Sequential({
            {"linear1", torch::nn::Linear(state_dim, 64)},
            {"tanh1", torch::nn::Tanh(),},
            {"linear2", torch::nn::Linear(64, 32),},
            {"tanh2", torch::nn::Tanh(),},
            {"linear3", torch::nn::Linear(32, action_dim),},
            {"tanh3", torch::nn::Tanh()}
        }))), critic(register_module("critic", torch::nn::Sequential({

            {"linear1", torch::nn::Linear(state_dim, 64)},
            {"tanh1", torch::nn::Tanh(),},
            {"linear2", torch::nn::Linear(64, 32),},
            {"tanh2", torch::nn::Tanh(),},
            {"linear3", torch::nn::Linear(32, 1),},            
        }))){
    
        int64_t dims[] = {action_dim};
        action_var =torch::full(
                torch::IntArrayRef(dims, 1),
                torch::Scalar(action_std*action_std));
        // std::cout << "action_var: " << action_var.grad_fn()->name() << std::endl;
        this -> action_std = action_std;
        //why register?
    }

    auto act(torch::Tensor state, MemoryNN& MemoryNN){
        
        torch::Tensor action_mean = actor->forward(state);
        
        torch::Tensor cov_mat = torch::diag(action_var);
        // torch::Tensor action = 0.0;
        // std::cout << "AAAAAAAAAAAAAAAAAAAAAAA" << action_mean.item<float>() << std::endl;
        // TODO: case of multiple action variables in the following line
        torch::Tensor action = torch::normal(action_mean.item<double>(), action_std, {action_mean.size(0)});
        std::cout << "COME ON!!! mean: " << action_mean.item<float>() << '\n'
             << "std: " << action_std << '\n'
             << "action: " << action.item<float>() << std::endl;

        // TODO: transform to real

        auto log_prob = multivariateLogProb(action_mean, cov_mat, action);
        // std::cout << log_prob << std::endl;
        // std::cout << "UNTIL HERE OBS IS: " << state << std::endl;
        MemoryNN.states.push_back(state);
        // std::cout << "OBS in MEMORY IS: " << MemoryNN.states << std::endl;
        // std::cout << "OBS in MEMORY has size: " << MemoryNN.states.size() << std::endl;
        MemoryNN.actions.push_back(action);
        MemoryNN.logprobs.push_back(log_prob);
        return std::make_tuple(action.detach(), log_prob);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate(torch::Tensor state, torch::Tensor action){
        // std::cout << "EVALUATE ";

        auto action_mean = actor->forward(state);
        
        auto action_var_expanded = action_var.expand_as(action_mean);
        
        auto cov_mat = torch::diag_embed(action_var_expanded);
        // std::cout << "cov_mat: " << cov_mat.grad_fn()->name() << std::endl;
        // std::cout << "%%%%%%%%%%%%%%%BUNCH OF STUFFS" << '\n' 
        //     << action_mean << action_var_expanded << cov_mat << std::endl;

        std::cout << "begin multivariateLogProb" << std::endl;
        auto action_logprobs = multivariateLogProb(action_mean, cov_mat, action);
        std::cout << "action_logprobs: " << action_logprobs.grad_fn()->name() << std::endl;
        printSizes(action_logprobs);
        std::cout << "action_mean.sizes()[1]: " << action_mean.sizes()[1] << std::endl;
        auto dist_entropy = multivariateEntropy(action_mean.sizes()[1], cov_mat);
        // std::cout << "dist_entropy: " << dist_entropy.grad_fn()->name() << std::endl;
        printSizes(dist_entropy);

        // auto action_logprobs = torch::randn({state.sizes()[0]});
        // auto dist_entropy = torch::randn({state.sizes()[0]});
        // for(int sample = 0; sample < state.sizes()[0]; sample++){
        //     auto sampleActionMean = action_mean.index({sample}).reshape({action_mean.sizes()[1], action.sizes()[2]?action.sizes()[2]:1});;

        //     // Eigen::Matrix<double, Eigen::Dynamic, 1> eigen_mean = tensorToVector(sampleActionMean);
        //     auto sampleCovar = cov_mat.index({sample});
            
        //     // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_covar = tensorToMatrix(sampleCovar);
        //     // Eigen::EigenMultivariateNormal<double> normalSolver(eigen_mean, eigen_covar,true,distr_NN(eng_NN));
        //     // auto squeezedAction = torch::squeeze(action);
        //     // PRINT_SIZES(squeezedAction.sizes());
        //     // std::cout << "Action sizes" << " " << action.sizes()[1] << " " << action.sizes()[2] << std::endl;
        //     auto sampleAction = action.index({sample}).reshape({action.sizes()[1]?action.sizes()[1]:1, action.sizes()[2]?action.sizes()[2]:1});


        //     std::cout << "sampleActionMean: " << sampleActionMean.grad_fn()->name() << std::endl;
        //     std::cout << "sampleCovar: " << sampleCovar.grad_fn()->name() << std::endl;
        //     // std::cout << "sampleAction: " << sampleAction.grad_fn()->name() << std::endl;
        //     auto action_logprob = multivariateLogProb(sampleActionMean, sampleCovar, sampleAction);
        //     std::cout << "action_logprob: " << action_logprob.grad_fn()->name() << std::endl;
        //     // std::cout << "Evaluation action_logprob " << action_logprob << std::endl;
        //     //TODO: make it general
        //     action_logprobs.index({sample}) = action_logprob.squeeze();
        //     auto sample_dist_entropy = multivariateEntropy(sampleActionMean.sizes()[0], sampleCovar);
        //     dist_entropy.index({sample}) = sample_dist_entropy.squeeze();
        // }
        
        



        
        //Why forward needed
        auto state_value = critic->forward(state);


        return std::make_tuple(action_logprobs, torch::squeeze(state_value), dist_entropy);
        // auto dist_entropy 
    }

    torch::nn::Sequential actor;
    torch::nn::Sequential critic;
    torch::Tensor action_var;
    double action_std;
};

class PPO {
    //Dummy constructor for initializing
    
    public:
    PPO(int64_t state_dim, int64_t action_dim, double action_std, double lr, std::tuple<double, double> betas, double gamma, int64_t K_epochs, double eps_clip){
        this->lr = lr;
        this->betas = betas;
        this->gamma = gamma;
        this->eps_clip = eps_clip;
        this->K_epochs = K_epochs;
        
        this->policy = ActorCritic(state_dim, action_dim, action_std);
        auto adamOptions = torch::optim::AdamOptions(this->lr);
        // adamOptions.betas()
        adamOptions.betas(betas);
        this->optimizer = new torch::optim::Adam(this->policy.parameters(), adamOptions);


        this->MseLoss = torch::nn::MSELoss();
    }

    auto select_action(torch::Tensor state, MemoryNN& MemoryNN){
        //TODO: check
        state = state.reshape({1, -1});
        auto [action, logProb] = policy.act(state, MemoryNN);
        
        action = action.cpu().flatten();
        
        return std::make_tuple(action, logProb);
    }
    auto update(MemoryNN MemoryNN){
        auto MemoryNNRewards = MemoryNN.rewards;
        auto MemoryNNIsTerminals = MemoryNN.is_terminals;
        auto MemoryNNIsTimeups = MemoryNN.is_timeups;
        auto MemoryNNStates = MemoryNN.states;
        std::reverse(MemoryNNRewards.begin(), MemoryNNRewards.end());
        std::reverse(MemoryNNIsTerminals.begin(), MemoryNNIsTerminals.end());
        std::reverse(MemoryNNIsTimeups.begin(), MemoryNNIsTimeups.end());
        std::reverse(MemoryNNStates.begin(), MemoryNNStates.end());
        torch::Tensor discounted_reward = torch::tensor({0.0});
        std::vector<torch::Tensor> discounted_rewards;

        std::cout << "MemoryNNIsTimeups: " << MemoryNNIsTimeups << std::endl;
        
        
        for(int index = 0; index < MemoryNNRewards.size(); index++){
            auto reward = MemoryNNRewards[index];
            auto is_terminal = MemoryNNIsTerminals[index];
            auto is_timeup = MemoryNNIsTimeups[index];
            auto MemoryNNState = MemoryNNStates[index];
            
            if (is_timeup){
                auto value = policy.critic->forward(MemoryNNState.squeeze());
                discounted_reward = value;
            }
            else if (is_terminal) discounted_reward = torch::tensor({0.0});            
            discounted_reward = reward + (gamma * discounted_reward);
            discounted_rewards.insert(discounted_rewards.begin(), discounted_reward);
        }
        std::cout << "rewards: " << MemoryNNRewards << std::endl;
        torch::Tensor Rewards = torch::cat(discounted_rewards);
        Rewards=Rewards.detach();
        std::cout << "Rewards: \n" << Rewards.requires_grad() << std::endl;
        // std::cout << "Merged Rewards: \n" << Rewards << std::endl;

        // rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5);
        auto old_states = torch::squeeze(torch::stack(MemoryNN.states)).detach();
        auto old_actions = torch::squeeze(torch::stack(MemoryNN.actions)).detach();
        auto old_logprobs = torch::squeeze(torch::stack(MemoryNN.logprobs)).detach();
        
        for(int index = 0; index < K_epochs; index++){
            
            std::cout << "BEGIN EVALUATION" << std::endl;
            auto res = policy.evaluate(old_states, old_actions);
            auto logprobs = std::get<0>(res);
            auto state_values = std::get<1>(res);
            auto dist_entropy = std::get<2>(res);

            // std::cout << "value: \n" << state_values.requires_grad() << std::endl;
            // std::cout << "value: \n" << state_values.grad_fn()->name() << std::endl;
            


            std::cout << "logprobs: \n" << logprobs.requires_grad() << std::endl;
            std::cout << "logprobs: \n" << logprobs.grad_fn()->name() << std::endl;
            
            std::cout << "dist_entropy: \n" << dist_entropy.requires_grad() << std::endl;
            // std::cout << "dist_entropy: \n" << dist_entropy << std::endl;
            // std::cout << "Log probs sizes" << logprobs.sizes()[0] << " " << old_logprobs.sizes()[0] << std::endl;
            auto ratios = torch::exp(logprobs - old_logprobs.detach());
            // std::cout << "ratios: \n" << ratios << std::endl;
            std::cout << "ratios: \n" << ratios.requires_grad() << std::endl;
            std::cout << "ratios: \n" << ratios.grad_fn()->name() << std::endl;
            
            // # Finding Surrogate Loss:
            printSizes(Rewards);
            printSizes(state_values);

            
            // std::cout << "state_values: " << state_values << std::endl;
            // std::cout << "value: \n" << state_values << std::endl;
            auto advantages = Rewards - state_values.detach();
            std::cout << "state_values: \n" << state_values.requires_grad() << std::endl;
            std::cout << "state_values: \n" << state_values.grad_fn()->name() << std::endl;
            // std::cout << "advantages: \n" << advantages << std::endl;
            std::cout << "advantages: \n" << advantages.requires_grad() << std::endl;
            // std::cout << "advantages: \n" << advantages.grad_fn()->name() << std::endl;
            
            auto surr1 = ratios * advantages;
            auto surr2 = torch::clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages;


            auto loss1 = -torch::min(surr1, surr2) ;
            auto loss2 = 0.5*MseLoss->forward(state_values, Rewards); 
            auto loss3 = - 0.01*dist_entropy;
            auto loss = loss1+ loss2 + loss3;


            std::cout << "LOSS1 is: " << loss1.mean() << std::endl;
            std::cout << "LOSS2 is: " << loss2.mean() << std::endl;
            std::cout << "LOSS3 is: " << loss3.mean() << std::endl;
            // auto loss = -torch::min(surr1, surr2) + 0.5*MseLoss->forward(state_values, Rewards) - 0.01*dist_entropy;
            std::cout << "LOSS is: " << loss.mean() << std::endl;
            std::cout << "LOSS is: " << loss.requires_grad() << std::endl;
            std::cout << "LOSS is: " << loss.grad_fn()->name() << std::endl;
            // # take gradient step
            optimizer->zero_grad();
            loss.mean().backward();
            optimizer->step();

            std::cout << "finish " << index <<" epoch" << std::endl; 
        }
        
    }

    double lr, gamma, eps_clip;
    std::tuple<double, double> betas;
    int64_t K_epochs;
    // ActorCritic policy, policy_old;
    ActorCritic policy;
    torch::optim::Adam* optimizer;
    torch::nn::MSELoss MseLoss;
};

std::tuple<std::vector<float>, float> getAction(std::vector<float> observation,  int dim, PPO ppo, MemoryNN& memoryNN)
{
    // std::vector<double> action(dim);
    //should we return both the action and the log prob here?
    
    torch::Tensor observationTensor = torch::from_blob(observation.data(), {(long int)observation.size()}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    // std::cout << "HERE IS WHAT YOU SHOULD LOOK AT" << observation << '\n' 
                // << observation.data() << '\n' << observationTensor << std::endl;
    auto [actionTensor, logProbTensor] = ppo.select_action(observationTensor, memoryNN);
    actionTensor = actionTensor.contiguous();


    std::vector<float> actionVec(actionTensor.data_ptr<float>(), actionTensor.data_ptr<float>() + actionTensor.numel());


    auto logProb = logProbTensor.item<float>();

    return {actionVec, logProb};
}
#endif
