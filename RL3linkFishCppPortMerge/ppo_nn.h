#ifndef ppo_nn
#define ppo_nn
#include <torch/torch.h>
#include <iostream>
#include "eigenmvn.h"
#include <vector>
#include <math.h>
#include <time.h>
#include <random>
using namespace std;

enum TRAIN_STATUS {NORMAL = 0, TERMINATE, TIMEUP, START};
std::random_device rd_NN;
std::mt19937_64 eng_NN(rd_NN());
std::uniform_int_distribution<unsigned long> distr_NN;

void printSizes(torch::Tensor& a){
    cout << a.sizes()[0] << " " << a.sizes()[1] << endl;
}
class MemoryNN {
    public:
    vector<torch::Tensor> actions, states, logprobs;
    vector<double> rewards;
    vector<bool> is_terminals;
    vector<bool> is_timeups;
    auto push_reward(double reward, bool terminate, bool timeup){
        cout << "rewards now: " << rewards << endl;
        rewards.push_back(reward);
        is_terminals.push_back(terminate);
        is_timeups.push_back(timeup);
    }
    void merge(MemoryNN& r);
    void clear();
};

void MemoryNN::merge(MemoryNN& r){

    cout << "States size: " << r.states.size() << " Rewards size: " << r.rewards.size() << " " << endl;



    this->actions.insert(this->actions.end(), r.actions.begin(), r.actions.end());
    this->states.insert(this->states.end(), r.states.begin(), r.states.end());
    this->logprobs.insert(this->logprobs.end(), r.logprobs.begin(), r.logprobs.end());
    this->rewards.insert(this->rewards.end(), r.rewards.begin(), r.rewards.end());
    this->is_terminals.insert(this->is_terminals.end(), r.is_terminals.begin(), r.is_terminals.end());
    this->is_timeups.insert(this->is_timeups.end(), r.is_timeups.begin(), r.is_timeups.end());
    cout << "Merge successful" << endl;
}

void MemoryNN::clear(){
    this->actions.clear();
    this->states.clear();
    this->logprobs.clear();
    this->rewards.clear();
    this->is_terminals.clear();
    this->is_timeups.clear();
}

template<typename Scalar>
torch::Tensor eigenToTensor(Eigen::Matrix<Scalar, Eigen::Dynamic, -1> mat){
    torch::Tensor out = torch::randn({mat.rows(), mat.cols()});
    int i = 0, j = 0;
    for(auto row: mat.rowwise()){
        j = 0;
        for(auto elem: row){
            out[i][j] = elem;
            j++;            
        }
        i++;
    }
    return out;
    // return torch::Tensor();
}

auto multivariateLogProb(torch::Tensor& action_mean, torch::Tensor& covar, torch::Tensor& action){
    // cout << covar << endl;
    // cout << covar.inverse() << endl;

    // cout << "Action is: " << action << endl;
    // cout << "Action mean is: " << action_mean << endl;
    // cout << "covar is: " << covar << endl;
    auto diff = action - action_mean;
  
    auto covarInverse = covar.inverse();
  
    auto numerator = -0.5 * (diff.transpose(0, 1).matmul(covarInverse.matmul( diff)));
    numerator = torch::exp(numerator);
    // cout << "numerator " << numerator << endl;
    auto denominator = pow(2 * M_PI, action_mean.sizes()[0]) * torch::det(covar).reshape({1, 1});
    denominator = torch::sqrt(denominator);
    // cout << "denominator " << denominator << endl;
    numerator = numerator / denominator;
    numerator =torch::log(numerator);
  

    return numerator;

}
auto multivariateEntropy(int k, torch::Tensor& covar){
    double v1 = pow(2*M_PI*M_E, k);
    // HH: I haven't figure out where the problem is, but the calculated value is always half of the correct value, so I removed the 0.5*
    // Also if the standard deviation is a function of trainable variables in neural network, by doing todouble and tensor will remove it from the gradient chain, which is wrong. But not it's fine
    return  torch::log(torch::tensor({v1 * torch::det(covar).item().toDouble()}));
}

// template<typename Scalar>
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> tensorToMatrix(torch::Tensor& in){
    // cout << "Tensor to matrix begin" << endl;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> out;
    int d1 = in.sizes()[0];
    int d2 = (in.sizes().size()>1?in.sizes()[1]:1);
    out.resize(d1, d2);
    
    for(int i = 0; i < out.rows(); i++){
        for(int j = 0; j < out.cols(); j++){
            out(i, j) = in.index({i, j}).item().toDouble();
        }
    }
    // cout << "Tensor to matrix end" << endl;
    return out;
}
// template<typename Scalar>
Eigen::Matrix<double, Eigen::Dynamic, 1> tensorToVector(torch::Tensor& in){
    // cout << "Tensor to vector begin" << endl;
    Eigen::Matrix<double, Eigen::Dynamic, 1> out;
    int d1 = in.sizes()[0];

    out.resize(d1, 1);
    for(int i = 0; i < out.rows(); i++){
            out(i,0) = in.index({i}).item().toDouble();
    }
    // cout << "Tensor to vector end" << endl;
    return out;

}

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
        action_var = register_parameter(
            "action_var",
            torch::full(
                torch::IntArrayRef(dims, 1),
                torch::Scalar(action_std*action_std)
            )
        );
        this -> action_std = action_std;
        //why register?
    }

    auto act(torch::Tensor state, MemoryNN& MemoryNN){
        
        torch::Tensor action_mean = actor->forward(state);
        
        torch::Tensor cov_mat = torch::diag(action_var);

        torch::Tensor action = torch::normal(0, action_std, {action_mean.size(0)});
        cout << "COME ON!!!" << action_std << action << endl;

        // TODO: transform to real

        auto log_prob = multivariateLogProb(action_mean, cov_mat, action);
        // cout << log_prob << endl;
        // cout << "UNTIL HERE OBS IS: " << state << endl;
        MemoryNN.states.push_back(state);
        // cout << "OBS in MEMORY IS: " << MemoryNN.states << endl;
        // cout << "OBS in MEMORY has size: " << MemoryNN.states.size() << endl;
        MemoryNN.actions.push_back(action);
        MemoryNN.logprobs.push_back(log_prob);
        return make_tuple(action.detach(), log_prob);
    }

    tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate(torch::Tensor state, torch::Tensor action){
        // cout << "EVALUATE ";

        auto action_mean = actor->forward(state);
        
        auto action_var_expanded = action_var.expand_as(action_mean);
        
        auto cov_mat = torch::diag_embed(action_var_expanded);
        
        // cout << "%%%%%%%%%%%%%%%BUNCH OF STUFFS" << '\n' 
        //     << action_mean << action_var_expanded << cov_mat << endl;
        auto action_logprobs = torch::randn({state.sizes()[0]});
        auto dist_entropy = torch::randn({state.sizes()[0]});
        for(int sample = 0; sample < state.sizes()[0]; sample++){
            auto sampleActionMean = action_mean.index({sample}).reshape({action_mean.sizes()[1], action.sizes()[2]?action.sizes()[2]:1});;

            // Eigen::Matrix<double, Eigen::Dynamic, 1> eigen_mean = tensorToVector(sampleActionMean);
            auto sampleCovar = cov_mat.index({sample});
            
            // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_covar = tensorToMatrix(sampleCovar);
            // Eigen::EigenMultivariateNormal<double> normalSolver(eigen_mean, eigen_covar,true,distr_NN(eng_NN));
            // auto squeezedAction = torch::squeeze(action);
            // PRINT_SIZES(squeezedAction.sizes());
            // cout << "Action sizes" << " " << action.sizes()[1] << " " << action.sizes()[2] << endl;
            auto sampleAction = action.index({sample}).reshape({action.sizes()[1]?action.sizes()[1]:1, action.sizes()[2]?action.sizes()[2]:1});
            auto action_logprob = multivariateLogProb(sampleActionMean, sampleCovar, sampleAction);
            // cout << "Evaluation action_logprob " << action_logprob << endl;
            //TODO: make it general
            action_logprobs.index({sample}) = action_logprob.squeeze();
            auto sample_dist_entropy = multivariateEntropy(sampleActionMean.sizes()[0], sampleCovar);
            dist_entropy.index({sample}) = sample_dist_entropy.squeeze();
        }
        
        
        //Why forward needed
        auto state_value = critic->forward(state);


        return make_tuple(action_logprobs, torch::squeeze(state_value), dist_entropy);
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
    PPO(int64_t state_dim, int64_t action_dim, double action_std, double lr, tuple<double, double> betas, double gamma, int64_t K_epochs, double eps_clip){
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
        
        return make_tuple(action, logProb);
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
        vector<torch::Tensor> discounted_rewards;

        cout << "MemoryNNIsTimeups: " << MemoryNNIsTimeups << endl;
        
        
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
        cout << "rewards: " << MemoryNNRewards << endl;
        torch::Tensor Rewards = torch::cat(discounted_rewards);
        Rewards=Rewards.detach();
        cout << "Rewards: \n" << Rewards.requires_grad() << endl;
        cout << "Merged Rewards: \n" << Rewards << endl;

        // rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5);
        auto old_states = torch::squeeze(torch::stack(MemoryNN.states)).detach();
        auto old_actions = torch::squeeze(torch::stack(MemoryNN.actions)).detach();
        auto old_logprobs = torch::squeeze(torch::stack(MemoryNN.logprobs)).detach();
        
        for(int index = 0; index < K_epochs; index++){
            
            cout << "BEGIN EVALUATION" << endl;
            auto res = policy.evaluate(old_states, old_actions);
            auto logprobs = std::get<0>(res);
            auto state_values = std::get<1>(res);
            auto dist_entropy = std::get<2>(res);

            cout << "value: \n" << state_values.requires_grad() << endl;
            cout << "value: \n" << state_values.grad_fn()->name() << endl;
            


            cout << "logprobs: \n" << logprobs.requires_grad() << endl;
            cout << "logprobs: \n" << logprobs.grad_fn()->name() << endl;
            
            cout << "dist_entropy: \n" << dist_entropy.requires_grad() << endl;
            // cout << "dist_entropy: \n" << dist_entropy << endl;
            // cout << "Log probs sizes" << logprobs.sizes()[0] << " " << old_logprobs.sizes()[0] << endl;
            auto ratios = torch::exp(logprobs - old_logprobs.detach());
            // cout << "ratios: \n" << ratios << endl;
            cout << "ratios: \n" << ratios.requires_grad() << endl;
            cout << "ratios: \n" << ratios.grad_fn()->name() << endl;
            
            // # Finding Surrogate Loss:
            printSizes(Rewards);
            printSizes(state_values);

            
            cout << "state_values: " << state_values << endl;
            // cout << "value: \n" << state_values << endl;
            auto advantages = Rewards - state_values.detach();
            // cout << "advantages: \n" << advantages << endl;
            cout << "advantages: \n" << advantages.requires_grad() << endl;
            // cout << "advantages: \n" << advantages.grad_fn()->name() << endl;
            
            auto surr1 = ratios * advantages;
            auto surr2 = torch::clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages;


            auto loss1 = -torch::min(surr1, surr2) ;
            auto loss2 = 0.5*MseLoss->forward(state_values, Rewards); 
            auto loss3 = - 0.01*dist_entropy;
            auto loss = loss1+ loss2 + loss3;


            cout << "LOSS1 is: " << loss1.mean() << endl;
            cout << "LOSS2 is: " << loss2.mean() << endl;
            cout << "LOSS3 is: " << loss3.mean() << endl;
            // auto loss = -torch::min(surr1, surr2) + 0.5*MseLoss->forward(state_values, Rewards) - 0.01*dist_entropy;
            cout << "LOSS is: " << loss.mean() << endl;
            cout << "LOSS is: " << loss.requires_grad() << endl;
            cout << "LOSS is: " << loss.grad_fn()->name() << endl;
            // # take gradient step
            optimizer->zero_grad();
            // loss.mean().backward(1);
            optimizer->step();

            cout << "finish " << index <<" epoch" << endl; 
        }
        
    }

    double lr, gamma, eps_clip;
    tuple<double, double> betas;
    int64_t K_epochs;
    // ActorCritic policy, policy_old;
    ActorCritic policy;
    torch::optim::Adam* optimizer;
    torch::nn::MSELoss MseLoss;
};

tuple<vector<float>, float> getAction(vector<float> observation,  int dim, PPO ppo, MemoryNN& memoryNN)
{
    // std::vector<double> action(dim);
    //should we return both the action and the log prob here?
    
    torch::Tensor observationTensor = torch::from_blob(observation.data(), {(long int)observation.size()}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    // cout << "HERE IS WHAT YOU SHOULD LOOK AT" << observation << '\n' 
                // << observation.data() << '\n' << observationTensor << endl;
    auto [actionTensor, logProbTensor] = ppo.select_action(observationTensor, memoryNN);
    actionTensor = actionTensor.contiguous();


    vector<float> actionVec(actionTensor.data_ptr<float>(), actionTensor.data_ptr<float>() + actionTensor.numel());


    auto logProb = logProbTensor.item<float>();

    return {actionVec, logProb};
}
#endif
