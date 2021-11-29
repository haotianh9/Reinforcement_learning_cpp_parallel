#include <torch/torch.h>
#include <iostream>
#include "eigenmvn.h"
#include<vector>
#include<math.h>

using namespace std;
class MemoryNN {
    public:
    vector<torch::Tensor> actions, states, logprobs;
    vector<double> rewards;
    vector<bool> is_terminals;
    auto push_reward(double reward, bool terminate, bool done){
        rewards.push_back(reward);
        is_terminals.push_back(terminate);

        //TODO: handle done
    }
};

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
    cout << "Action is: " << action << endl;
    auto diff = action - action_mean;
    // cout << "Diff sizes " << diff.sizes() << endl;
    // cout << diff.transpose(0, 1) << endl;
    auto covarInverse = covar.inverse();
    auto numerator = -0.5 * (diff.transpose(0, 1).matmul(covarInverse.matmul( diff)));
    numerator = torch::exp(numerator);
    auto denominator = pow(2 * M_PI, action_mean.sizes()[0]) * torch::tensor({torch::det(covar).item().toDouble()}).reshape({1, 1});
    denominator = torch::sqrt(denominator);
    // cout << numerator << endl;
    numerator = numerator / denominator;
    return numerator;

}

auto multivariateEntropy(int k, torch::Tensor& covar){
    double v1 = pow(2*M_PI*M_E, k);
    return 0.5 * torch::log(torch::tensor({v1 * torch::det(covar).item().toDouble()}));
}

auto tensorToMatrix2d(torch::Tensor& in){
    Eigen::Matrix2d out;
    for(int i = 0; i < out.rows(); i++){
        for(int j = 0; j < out.cols(); j++){
            out(i, j) = in.index({i, j}).item().toDouble();
        }
    }
    return out;
}

auto tensorToVector2d(torch::Tensor& in){
    Eigen::Vector2d out;
    for(int i = 0; i < out.rows(); i++){
        for(int j = 0; j < out.cols(); j++){
            out(i, j) = in.index({i, j}).item().toDouble();
        }
    }
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

    auto act(torch::Tensor state, MemoryNN MemoryNN){
        // torch::Tensor test = linear(state);
        torch::Tensor action_mean = actor->forward(state);
        torch::Tensor cov_mat = torch::diag(action_var);
        // cov_mat = torch.diag(self.action_var).to(device)
        //TODO:NEED To convert to Pytorch
        Eigen::Vector2d eigen_mean = tensorToVector2d(action_mean);
        Eigen::Matrix2d eigen_covar = tensorToMatrix2d(cov_mat);
        Eigen::EigenMultivariateNormal<double> normalSolver(eigen_mean, eigen_covar);
        auto sampledAction = eigenToTensor(normalSolver.samples(1));
        // cout <<  normalSolver.samples(1) << endl;
        auto sampledActionLogProb = multivariateLogProb(action_mean, cov_mat, sampledAction);
        // auto dist = torch::MultivariateNormal(action_mean, cov_mat);
        torch::Tensor action = sampledAction, log_prob = sampledActionLogProb;
        
        MemoryNN.states.push_back(state);
        MemoryNN.actions.push_back(action);
        MemoryNN.logprobs.push_back(log_prob);
        return {action.detach(), log_prob};
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
        auto action_mean = torch::squeeze(actor->forward(state));
        auto cov_mat = torch::diag_embed(action_var);
        Eigen::Vector2d eigen_mean = tensorToVector2d(action_mean);
        Eigen::Matrix2d eigen_covar = tensorToMatrix2d(cov_mat);
        Eigen::EigenMultivariateNormal<double> normalSolver(eigen_mean, eigen_covar);
        auto squeezedAction = torch::squeeze(action);
        auto action_logprobs = multivariateLogProb(action_mean, cov_mat, squeezedAction);
        auto entropy = multivariateEntropy(action_mean.sizes()[0], cov_mat);
        //Why forward needed
        auto state_value = critic->forward(state);


        return make_tuple(action_mean, action_mean, action_mean);
        // auto dist_entropy 
    }

    torch::nn::Sequential actor;
    torch::nn::Sequential critic;
    torch::Tensor action_var; 
};

class PPO {
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

    auto select_action(torch::Tensor state, MemoryNN MemoryNN){
        //TODO: check
        state = state.reshape({1, -1});
        return policy_old.act(state, MemoryNN).cpu().flatten();

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

        vector<double> newRewards;
        for(auto r: rewards) newRewards.push_back(r.item().toDouble());
        auto newRewardsT = torch::tensor(newRewards);
        // auto tensorRewards = torch::tensor(rewards);
        // rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5);
        auto old_states = torch::squeeze(torch::stack(MemoryNN.states));
        auto old_actions = torch::squeeze(torch::stack(MemoryNN.actions));
        auto old_logprobs = torch::squeeze(torch::stack(MemoryNN.logprobs));
        for(int index = 0; index < K_epochs; index++){
            auto res = policy.evaluate(old_states, old_actions);
            auto logprobs = std::get<0>(res);
            auto state_values = std::get<1>(res);
            auto dist_entropy = std::get<2>(res);
            auto ratios = torch::exp(logprobs - old_logprobs.detach());

            // # Finding Surrogate Loss:
            auto advantages = newRewardsT - state_values.detach();
            auto surr1 = ratios * advantages;
            auto surr2 = torch::clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages;
            auto loss = -torch::min(surr1, surr2) + 0.5*MseLoss->forward(state_values, newRewardsT) - 0.01*dist_entropy;
            
            // # take gradient step
            optimizer->zero_grad();
            loss.mean().backward();
            optimizer->step();
        }
        
        
        
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

Eigen::Matrix2d genCovar(double v0,double v1,double theta)
{
  Eigen::Matrix2d rot = Eigen::Rotation2Dd(theta).matrix();
  return rot*Eigen::DiagonalMatrix<double,2,2>(v0,v1)*rot.transpose();
}


// int main() {
//   torch::Tensor tensor = torch::rand({2, 3});
//   std::cout << tensor << std::endl;

//   Eigen::Vector2d mean;
//   Eigen::Matrix2d covar;
//   mean << -1,0.5; // Set the mean
// //   cout << eigenToTensor(mean) << endl;
//   // Create a covariance matrix
//   // Much wider than it is tall
//   // and rotated clockwise by a bit
//   covar = genCovar(3,0.1,M_PI/5.0);
//   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> newCovar = covar;
//   cout << newCovar << endl;
//   cout << eigenToTensor(newCovar) << endl;
//   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> newMean = mean;

//   torch::Tensor in = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
//   cout << tensorToMatrix2d(in) << endl;
//   auto newMeanT = eigenToTensor(newMean);
//   auto newCovarT = eigenToTensor(newCovar);
//   auto newActionT = torch::tensor({-1,-1}).reshape({2, 1});
//   cout << multivariateLogProb(newMeanT, newCovarT, newActionT);
// //   cout << eigenToTensor(covar) << endl;
// //   cout <

//   // Create a bivariate gaussian distribution of doubles.
//   // with our chosen mean and covariance
//   const int dim = 1;
//   Eigen::EigenMultivariateNormal<double> normX_solver(mean,covar);
  
  
// //   cout << normX_solver.samples(1)  << endl;

// }