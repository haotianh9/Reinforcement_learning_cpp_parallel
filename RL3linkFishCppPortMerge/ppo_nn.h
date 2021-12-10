#ifndef ppo_nn
#define ppo_nn
#include <torch/torch.h>
#include <iostream>
#include "eigenmvn.h"
#include<vector>
#include<math.h>

#define PRINT_SIZES(a) cout << a[0] <<" " << a[1] << endl
using namespace std;


void printSizes(torch::Tensor& a){
    cout << a.sizes()[0] << " " << a.sizes()[1] << endl;
}
class MemoryNN {
    public:
    vector<torch::Tensor> actions, states, logprobs;
    vector<double> rewards;
    vector<bool> is_terminals;
    vector<bool> is_dones;
    auto push_reward(double reward, bool terminate, bool done){
        rewards.push_back(reward);
        is_terminals.push_back(terminate);
        is_dones.push_back(done);
    }
    void merge(MemoryNN& r);
    void clear();



    //TODO: handle done

};

void MemoryNN::merge(MemoryNN& r){
    // cout << "Rewards: " << r.states.size() << " " << r.rewards.size() << " " << endl;
    auto rewardsDiff = r.rewards.end() - r.rewards.begin();
    this->actions.insert(this->actions.end(), r.actions.begin(), r.actions.begin()+rewardsDiff);
    r.actions.erase(r.actions.begin(), r.actions.begin()+rewardsDiff);
    this->states.insert(this->states.end(), r.states.begin(), r.states.begin() + rewardsDiff);
    r.states.erase(r.states.begin(), r.states.begin() + rewardsDiff);
    this->logprobs.insert(this->logprobs.end(), r.logprobs.begin(), r.logprobs.begin() + rewardsDiff);
    r.logprobs.erase(r.logprobs.begin(), r.logprobs.begin() + rewardsDiff);
    this->rewards.insert(this->rewards.end(), r.rewards.begin(), r.rewards.begin() + rewardsDiff);
    r.rewards.erase(r.rewards.begin(), r.rewards.begin() + rewardsDiff);
    this->is_terminals.insert(this->is_terminals.end(), r.is_terminals.begin(), r.is_terminals.begin() + rewardsDiff);
    r.is_terminals.erase(r.is_terminals.begin(), r.is_terminals.begin() + rewardsDiff);
    // cout << "Merge successful" << endl;

}

void MemoryNN::clear(){
    this->actions.clear();
    this->states.clear();
    this->logprobs.clear();
    this->rewards.clear();
    this->is_terminals.clear();
    this->is_dones.clear();
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
    // cout << "Diff sizes " << diff << endl;
    
    // cout << "diff " << diff << endl;
    auto covarInverse = covar.inverse();
    // cout << "Multivariate log prob" << endl;
    // printSizes(action_mean);
    // printSizes(covar);
    // printSizes(diff);    

    auto numerator = -0.5 * (diff.transpose(0, 1).matmul(covarInverse.matmul( diff)));
    numerator = torch::exp(numerator);
    // cout << "numerator " << numerator << endl;
    auto denominator = pow(2 * M_PI, action_mean.sizes()[0]) * torch::det(covar).reshape({1, 1});
    denominator = torch::sqrt(denominator);
    // cout << "denominator " << denominator << endl;
    numerator = numerator / denominator;
    numerator =torch::log(numerator);
    // cout << action_mean.sizes() << endl;
    // cout << action_mean.sizes()[0] << endl;
    // cout << typeid(action_mean.sizes()[0]).name() << endl;


    // TODO: rewrite the formular to save computational power

    // auto numerator = -0.5 * (diff.transpose(0, 1).matmul(covarInverse.matmul( diff)));
    // // numerator = torch::exp(numerator);
    // cout << "numerator " << numerator << endl;
    // cout << action_mean.sizes() << endl;
    // auto denominator = torch::log(torch::Tensor(2 * (float)M_PI) )* (float)action_mean.sizes()[0] + torch::log(torch::det(covar));
    // denominator = denominator / 2;
    // cout << "denominator " << denominator << endl;
    // numerator = numerator - denominator;


    return numerator;

}

auto multivariateEntropy(int k, torch::Tensor& covar){
    double v1 = pow(2*M_PI*M_E, k);
    return 0.5 * torch::log(torch::tensor({v1 * torch::det(covar).item().toDouble()}));
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
            //TODO: make it state_dim
            {"linear1", torch::nn::Linear(state_dim, 64)},
            {"tanh1", torch::nn::Tanh(),},
            {"linear2", torch::nn::Linear(64, 32),},
            {"tanh2", torch::nn::Tanh(),},
            {"linear3", torch::nn::Linear(32, action_dim),},
            {"tanh3", torch::nn::Tanh()}
        }))), critic(register_module("critic", torch::nn::Sequential({
            //TODO: make it state_dim
            {"linear1", torch::nn::Linear(state_dim, 64)},
            {"tanh1", torch::nn::Tanh(),},
            {"linear2", torch::nn::Linear(64, 32),},
            {"tanh2", torch::nn::Tanh(),},
            {"linear3", torch::nn::Linear(32, 1),},            
        }))){
        // actor = register_module("actor", torch::nn::Sequential({
        //     //TODO: make it state_dim
        //     {"linear1", torch::nn::Linear(state_dim, 64)},
        //     {"tanh1", torch::nn::Tanh(),},
        //     {"linear2", torch::nn::Linear(64, 32),},
        //     {"tanh2", torch::nn::Tanh(),},
        //     {"linear3", torch::nn::Linear(32, action_dim),},
        //     {"tanh3", torch::nn::Tanh()}
        // }));

        // critic = register_module("critic", torch::nn::Sequential({
        //     //TODO: make it state_dim
        //     {"linear1", torch::nn::Linear(state_dim, 64)},
        //     {"tanh1", torch::nn::Tanh(),},
        //     {"linear2", torch::nn::Linear(64, 32),},
        //     {"tanh2", torch::nn::Tanh(),},
        //     {"linear3", torch::nn::Linear(32, 1),},            
        // }));
        int64_t dims[] = {action_dim};
        action_var = register_parameter(
            "action_var",
            torch::full(
                torch::IntArrayRef(dims, 1),
                torch::Scalar(action_std*action_std)
            )
        );
        //why register?
    }

    auto act(torch::Tensor state, MemoryNN& MemoryNN){
        // torch::Tensor test = linear(state);
        // cout << "ACT ";
        // PRINT_SIZES(state.sizes());
        // cout << "CLEAR one" << endl;
        torch::Tensor action_mean = actor->forward(state);
        
        torch::Tensor cov_mat = torch::diag(action_var);
        // cout << "CLEAR two" << endl;
        // cout << "Action " << action_mean << endl;
        // cout << "Cov Mat" << cov_mat << endl;
        // PRINT_SIZES(action_mean.sizes());
        // cov_mat = torch.diag(self.action_var).to(device)
        //TODO:NEED To convert to Pytorch
        // Eigen::Vector2d eigen_mean = tensorToVector2d(action_mean);
        // Eigen::Matrix2d eigen_covar = tensorToMatrix2d(cov_mat);
        Eigen::Matrix<double, Eigen::Dynamic, 1> eigen_mean = tensorToMatrix(action_mean);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_covar = tensorToMatrix(cov_mat);
        // cout << "CLEAR three" << endl;
        Eigen::EigenMultivariateNormal<double> normalSolver(eigen_mean, eigen_covar);
        auto sampledAction = eigenToTensor(normalSolver.samples(1));
        // cout <<  normalSolver.samples(1) << endl;
        // cout << "CLEAR FOUR" << endl;
        auto sampledActionLogProb = multivariateLogProb(action_mean, cov_mat, sampledAction);
        // auto dist = torch::MultivariateNormal(action_mean, cov_mat);
        // cout << "CLEAR five" << endl;
        torch::Tensor action = sampledAction, log_prob = sampledActionLogProb;
        // cout << log_prob << endl;
        MemoryNN.states.push_back(state);
        MemoryNN.actions.push_back(action);
        MemoryNN.logprobs.push_back(log_prob);
        return make_tuple(action.detach(), log_prob);
        // dist = MultivariateNormal(action_mean, cov_mat)
        // #Sampling from distribution of probabilities
        // action = dist.sample()
        // action_logprob = dist.log_prob(action)
        
        // MemoryNN.states.append(state)
        // MemoryNN.actions.append(action)
        // MemoryNN.logprobs.append(action_logprob)
        
        // return action.detach()
        
    }

    tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate(torch::Tensor state, torch::Tensor action){
        // cout << "EVALUATE ";
        // PRINT_SIZES(state.sizes());
        auto action_mean = actor->forward(state);
        // cout << action_mean.sizes()[0] << " " << action_var.sizes()[0] << endl;
        auto action_var_expanded = action_var.expand_as(action_mean);
        auto cov_mat = torch::diag_embed(action_var_expanded);

        // PRINT_SIZES(action_mean.sizes());
        // for(auto s: action_mean.sizes()){
        //     cout << s << " ";
        // }
        // for(auto s: cov_mat.sizes()){
        //     cout << s << " ";
        // }
        // cout << endl;
        // PRINT_SIZES(cov_mat.sizes();
        auto action_logprobs = torch::randn({state.sizes()[0]});
        auto dist_entropy = torch::randn({state.sizes()[0]});
        for(int sample = 0; sample < state.sizes()[0]; sample++){
            
            auto sampleActionMean = action_mean.index({sample}).reshape({action_mean.sizes()[1], action.sizes()[2]?action.sizes()[2]:1});;
            Eigen::Matrix<double, Eigen::Dynamic, 1> eigen_mean = tensorToVector(sampleActionMean);
            auto sampleCovar = cov_mat.index({sample});
            // for(auto s: sampleCovar.sizes()){
            //     cout << s << " ";
            // }
            // cout << endl;
            
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_covar = tensorToMatrix(sampleCovar);
            Eigen::EigenMultivariateNormal<double> normalSolver(eigen_mean, eigen_covar);
            // auto squeezedAction = torch::squeeze(action);
            // PRINT_SIZES(squeezedAction.sizes());
            // cout << "Action sizes" << " " << action.sizes()[1] << " " << action.sizes()[2] << endl;
            auto sampleAction = action.index({sample}).reshape({action.sizes()[1]?action.sizes()[1]:1, action.sizes()[2]?action.sizes()[2]:1});
            auto action_logprob = multivariateLogProb(sampleActionMean, sampleCovar, sampleAction);
            cout << "Evaluation action_logprob " << action_logprob << endl;
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
        
        this->policy_old = ActorCritic(state_dim, action_dim, action_std);
        std::stringstream in;
        
        // torch::nn::ModuleHolder<torch::nn::Module> policyModuleHolder(std::make_shared<torch::nn::Module>(&this->policy));
        // torch::nn::ModuleHolder<torch::nn::Module> policyOldModuleHolder(std::make_shared<torch::nn::Module>(&this->policy_old));
        
        auto sharedPtrPolicy = std::make_shared<torch::nn::Module>(this->policy);
        auto sharedPtrPolicyOld = std::make_shared<torch::nn::Module>(this->policy_old);
        torch::save(sharedPtrPolicy, in);
        // outputArchive->save_to(in);
        torch::load(sharedPtrPolicyOld, in);
        // this->policy_old.load_state_dict(this->policy.state_dict());
   
        this->MseLoss = torch::nn::MSELoss();
    }

    auto select_action(torch::Tensor state, MemoryNN& MemoryNN){
        //TODO: check
        state = state.reshape({1, -1});
        auto [action, logProb] = policy_old.act(state, MemoryNN);
        // PRINT_SIZES(action.sizes()) << endl;
        action = action.cpu().flatten();
        // PRINT_SIZES(action.sizes()) << endl;
        return make_tuple(action, logProb);

    }

    auto update(MemoryNN MemoryNN){
        auto MemoryNNRewards = MemoryNN.rewards;
        auto MemoryNNIsTerminals = MemoryNN.is_terminals;
        auto MemoryNNStates = MemoryNN.states;
        std::reverse(MemoryNNRewards.begin(), MemoryNNRewards.end());
        std::reverse(MemoryNNIsTerminals.begin(), MemoryNNIsTerminals.end());
        std::reverse(MemoryNNStates.begin(), MemoryNNStates.end());
        torch::Tensor discounted_reward = torch::tensor({0.0});
        vector<torch::Tensor> rewards;
        // cout << MemoryNNRewards.size() << " " << MemoryNNIsTerminals.size() << " " << MemoryNNStates.size() << endl;
        for(int index = 0; index < MemoryNNRewards.size(); index++){
            auto reward = MemoryNNRewards[index];
            auto is_terminal = MemoryNNIsTerminals[index];
            auto MemoryNNState = MemoryNNStates[index];
            if(is_terminal){
                auto value = policy.critic->forward(MemoryNNState.squeeze());
                discounted_reward = value;
                
            }
            discounted_reward = reward + (gamma * discounted_reward);
            rewards.insert(rewards.begin(), discounted_reward);
        }
        cout << "Discounted reward: " << discounted_reward << endl;
        vector<double> newRewards;
        for(auto r: rewards) newRewards.push_back(r.item().toDouble());
        auto newRewardsT = torch::tensor(newRewards);
        // auto tensorRewards = torch::tensor(rewards);
        // rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5);
        auto old_states = torch::squeeze(torch::stack(MemoryNN.states));
        auto old_actions = torch::squeeze(torch::stack(MemoryNN.actions));
        auto old_logprobs = torch::squeeze(torch::stack(MemoryNN.logprobs));
        // cout << "Memory actions" << MemoryNN.actions.size() << endl;
        for(int index = 0; index < K_epochs; index++){
            cout << "begin evaluation !!!!!!!!!" << endl;
            auto res = policy.evaluate(old_states, old_actions);
            auto logprobs = std::get<0>(res);
            auto state_values = std::get<1>(res);
            auto dist_entropy = std::get<2>(res);
            // cout << "Log probs sizes" << logprobs.sizes()[0] << " " << old_logprobs.sizes()[0] << endl;
            auto ratios = torch::exp(logprobs - old_logprobs.detach());

            // # Finding Surrogate Loss:
            // cout << "Rewards Size and State size" << newRewardsT.sizes()[0] << " " << state_values.sizes()[0] << endl;
            auto advantages = newRewardsT - state_values.detach();
            auto surr1 = ratios * advantages;
            auto surr2 = torch::clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages;
            auto loss = -torch::min(surr1, surr2) + 0.5*MseLoss->forward(state_values, newRewardsT) - 0.01*dist_entropy;
            // cout << "LOSS is: " << loss << endl;
            // # take gradient step
            optimizer->zero_grad();
            loss.mean().backward();
            optimizer->step();
        }
        
        std::stringstream in;
        auto sharedPtrPolicy = std::make_shared<torch::nn::Module>(this->policy);
        auto sharedPtrPolicyOld = std::make_shared<torch::nn::Module>(this->policy_old);
        torch::save(sharedPtrPolicy, in);
        // outputArchive->save_to(in);
        torch::load(sharedPtrPolicyOld, in);
        
         //load_state_dict(self.policy.state_dict()
        
        // # Optimize policy for K epochs:
        // for _ in range(self.K_epochs):
        //     # Evaluating old actions and values :
        //     logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
        //     # Finding the ratio (pi_theta / pi_theta__old):
        //     ratios = torch.exp(logprobs - old_logprobs.detach())

        //     # Finding Surrogate Loss:
        //     advantages = rewards - state_values.detach()   
        //     surr1 = ratios * advantages
        //     surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
        //     loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
        //     # take gradient step
        //     self.optimizer.zero_grad()
        //     loss.mean().backward()
        //     self.optimizer.step()
            
        // # Copy new weights into old policy:
        // self.policy_old.load_state_dict(self.policy.state_dict())
    }

    double lr, gamma, eps_clip;
    tuple<double, double> betas;
    int64_t K_epochs;
    ActorCritic policy, policy_old;
    torch::optim::Adam* optimizer;
    torch::nn::MSELoss MseLoss;
};

tuple<vector<float>, float> getAction(vector<float> observation,  int dim, PPO ppo, MemoryNN& memoryNN)
{
    // std::vector<double> action(dim);
    //should we return both the action and the log prob here?
    
    torch::Tensor observationTensor = torch::from_blob(observation.data(), {(long int)observation.size()}, torch::TensorOptions().dtype(torch::kFloat32));
    // cout << "HERE IS WHAT YOU SHOULD LOOK AT" << observation << '\n' 
                // << observation.data() << '\n' << observationTensor << endl;
    auto [actionTensor, logProbTensor] = ppo.select_action(observationTensor, memoryNN);
    actionTensor = actionTensor.contiguous();
    // action[0]=observation[0]+observation[1];
    // float logprob;
    // logprob=0.2;
    // PRINT_SIZES(actionTensor.sizes());
    vector<float> actionVec(actionTensor.data_ptr<float>(), actionTensor.data_ptr<float>() + actionTensor.numel());
    // for (int i = 0; i < actionTensor.sizes()[0]; i++) 
    //     actionVec.push_back(actionTensor.index({i}).item());

    auto logProb = logProbTensor.item<float>();
    return {actionVec, logProb};
}

#endif
