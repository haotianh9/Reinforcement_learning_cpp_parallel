# Reinforcement_learning_cpp_parallel
Name is to be determined.

A parallel reinforcement learning framework written in C++

Goal: build a scalable, reliable and well-maintained RL framework in C++ that can be used for engineering research.

see documentation [here](https://github.com/haotianh9/Reinforcement_learning_cpp_parallel/tree/main/doc/main.pdf)

For dependence, building and running, see [here](https://github.com/haotianh9/Reinforcement_learning_cpp_parallel/blob/main/Build%20and%20use%20note)

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

# Build

1. Download dependence:
[libtorch](https://pytorch.org/cppdocs/installing.html) and add libtorch to your environment path:
"
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
export Torch_DIR=</absolute/path/to/libtorch>
echo "export Torch_DIR=</absolute/path/to/libtorch>" >> ~/.bashrc
"
clang-format is not reqiured, but if you want to use "make format", you need to have it installed
on debian system:
"
sudo apt install clang-format
"
2. build our project:
First way(using Makefile):
"
make all 
"
or 
make build
or 
make debug

Second way(using cmake):
mkdir build
cd build
cmake  ..
cmake --build . --config Release
or
cmake --build . --config Debug

2.1 build on USC CARC 
module load gcc/8.3.0 
module load openmpi/4.0.2
module load cmake
export LD_PRELOAD=/spack/apps/gcc/8.3.0/lib64/libstdc++.so.6

build:
cmake  -DCMAKE_C_COMPILER=gcc   -DCMAKE_CXX_COMPILER=g++   ..

3. running:
mpirun -n 4 ./cpp-rl-training <path/to/config/file> > out   
or
mpirun -n 4 ./cpp-rl-training  > out   
in which 4 is the amount of nodes you want to use 
if config file is not specified, the default config file will be used (../config)
(> out) means everything which are supposed to be printed in terminal is instead printed in a file called out
  
3.1 running on CARC
use .sl file 
