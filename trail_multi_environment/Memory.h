#include <vector>
using namespace std;
class Memory
{   
    private:
        

    public:
        std::vector<std::vector<double>> observation_list;
        std::vector<std::vector<double>> action_list;
        std::vector<double>   logprobs_list;
        std::vector<double>   reward_list;
        std::vector<bool>   terminate_list;
        std::vector<bool>   done_list;
        void push_obs_act(std::vector<double>,std::vector<double>,double);
        void push_reward(double,bool,bool);
        void clear_memory();
        // std::tuple<std::vector<double>,td::vector<double>,double, double,bool,bool> get_memory(int);
};

// Don'y need to define constructor, use default constructor
// Memory::Memory(){

// }

void Memory::push_obs_act(std::vector<double> obs,std::vector<double> act,double logprob){
    observation_list.push_back(obs);
    action_list.push_back(act);
    logprobs_list.push_back(logprob);
}

void Memory::push_reward(double reward,bool terminate,bool done){
    reward_list.push_back(reward);
    terminate_list.push_back(terminate);
    done_list.push_back(done);
}

void Memory::clear_memory(){
    observation_list.clear();
    action_list.clear();
    logprobs_list.clear();
    reward_list.clear();
    terminate_list.clear();
    done_list.clear();
}

// std::tuple<std::vector<double>,td::vector<double>,double, double,bool,bool> get_memory(int i){

// }