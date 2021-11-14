#include <vector>

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

// std::tuple<std::vector<double>,td::vector<double>,double, double,bool,bool> get_memory(int i){

// }