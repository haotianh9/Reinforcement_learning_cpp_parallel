#include "ppo_nn.h"
std::tuple<std::vector<double>, double> getAction(std::vector<double> observation,  int dim, PPO ppo, MemoryNN& memoryNN)
{
    std::vector<double> action(dim);
    //should we return both the action and the log prob here?
    
    //TODO: probably inefficient conversion
    
    torch::Tensor observationTensor = torch::randn({observation.size()});
    for(int i = 0; i < observation.size(); i++){
        observationTensor[{i}] = observation[i];
    }
    //torch::from_blob(observation.data(), {observation.size()}, torch::TensorOptions().dtype(torch::kFloat32));
    cout << "Observation " << observationTensor << endl;
    auto [actionTensor, logProbTensor] = ppo.select_action(observationTensor, memoryNN);
    // action[0]=observation[0]+observation[1];
    // float logprob;
    // logprob=0.2;
    // PRINT_SIZES(actionTensor.sizes());
    vector<double> actionTensorDouble;
    
    for(int i = 0; i < actionTensor.sizes()[0]; i++){
            actionTensorDouble.push_back(actionTensor.index({i}).item().toDouble());
    }
    // cout << actionTensorDouble.size() << endl;
    // for(auto a: actionTensorDouble){
    //     cout << a << " ";
    // }
    // cout << endl;
    auto logProb = logProbTensor.item().toDouble();
    return {actionTensorDouble, logProb};
}
