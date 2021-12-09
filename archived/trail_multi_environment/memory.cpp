//
// Created by qiongao on 2021/11/14.
//
#include "Memory.h"

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