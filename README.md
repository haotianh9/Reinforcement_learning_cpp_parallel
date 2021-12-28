# Reinforcement_learning_cpp_parallel
Name is to be determined.

A parallel reinforcement learning framework written in C++

Goal: build a scalable, reliable and well-maintained RL framework in C++ that can be used for engineering research.

see documentation [here](https://github.com/haotianh9/Reinforcement_learning_cpp_parallel/tree/main/doc/main.pdf)
# alpha 0.1
Version alpha 0.1 implements Policy Proximal Optimization ([PPO](https://arxiv.org/pdf/1707.06347.pdf)) learning algorithm only. Basic MPI functions are used for communications between simulation nodes and the learning node. [Libtorch](https://pytorch.org/cppdocs/) is used for neural network training and inference. Our code can be compiled and run on USC CARC. 

Using pybind11 to make our code callable in python, using GPU for training and inference, and implement other algorithms are our furture goals. 
<!-- We also tried using [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), but finally found that we don't actually need it.  -->

Developed by Kishore Ganesh, Qiongao Liu, Yusheng Jiao, Chenchen Huang, Haotian Hang as USC CSCI596 course final project

We used our code to trian on a cartpole environment. The learning curve using only one environment node is shown as follows.

![learning curve](https://github.com/haotianh9/Reinforcement_learning_cpp_parallel/blob/main/results/carpole1node/learning%20curve.jpg)

The learning curve using  three environments running simutaneously is showns as follows.

![learning curve](https://github.com/haotianh9/Reinforcement_learning_cpp_parallel/blob/main/results/carpole3node/learning_curve.jpg)

In the learning curves above, the dots shows the reward of each episode, and the line shows the average reward over 100 episodes.
