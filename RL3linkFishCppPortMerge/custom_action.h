#include "ppo_nn.h"
std::tuple<std::vector<double>, double> getAction(std::vector<double> observation,  int dim, PPO ppo, MemoryNN& memoryNN)
{
    std::vector<double> action(dim);
    //should we return both the action and the log prob here?
    
    torch::Tensor observationTensor = torch::from_blob(observation.data(), {(long int)observation.size()}, torch::TensorOptions().dtype(torch::kFloat64));
    std::cout << "HERE IS WHAT YOU SHOULD LOOK AT" << observation << '\n' 
                << observation.data() << '\n' << observationTensor << endl;

    auto [actionTensor, logProbTensor] = ppo.select_action(observationTensor, memoryNN);
    // action[0]=observation[0]+observation[1];
    // float logprob;
    // logprob=0.2;
    // PRINT_SIZES(actionTensor.sizes());
    vector<double> actionTensorDouble;
    cout << "CLEAR 1" << endl;
    for(int i = 0; i < actionTensor.sizes()[0]; i++){
            actionTensorDouble.push_back(actionTensor.index({i}).item().toDouble());
    }
    cout << "CLEAR 2" << endl;
    // cout << actionTensorDouble.size() << endl;
    // for(auto a: actionTensorDouble){
    //     cout << a << " ";
    // }
    // cout << endl;
    auto logProb = logProbTensor.item().toDouble();
    cout << "CLEAR 3" << endl;
    return {actionTensorDouble, logProb};
}
