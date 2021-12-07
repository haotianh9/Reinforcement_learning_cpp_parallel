#include "Memory.h"
#include <iostream>
#include <vector>
using namespace std;


int main ()
{
  Memory mem;  
  std::vector<double> myvector;
  myvector.push_back (19);
  myvector.push_back (2.09);
  myvector.push_back (4.07);
  std::vector<double> myvector2;
  myvector2.push_back (9);
  myvector2.push_back (19);


  mem.push_obs_act(myvector,myvector2,0.1);
  mem.push_reward(9,false,false);

  mem.push_obs_act(myvector,myvector2,0.9);
  mem.push_reward(19,false,true);
  int i=0;
  printf("observation: %f %f %f \t action: %f %f \t logprob: %f  \n reward: %f \t terminate: %s \t done: %s  \n", mem.observation_list[i][0], mem.observation_list[i][1], mem.observation_list[i][2], mem.action_list[i][0], mem.action_list[i][1], mem.reward_list[i], mem.terminate_list[i]?"true":"false", mem.done_list[i]?"true":"false");
  i++;
  printf("observation: %f %f %f \t action: %f %f \t logprob: %f  \n reward: %f \t terminate: %s \t done: %s  \n", mem.observation_list[i][0], mem.observation_list[i][1], mem.observation_list[i][2], mem.action_list[i][0], mem.action_list[i][1], mem.reward_list[i], mem.terminate_list[i]?"true":"false", mem.done_list[i]?"true":"false");



  mem.clear_memory();
  return 0;
}