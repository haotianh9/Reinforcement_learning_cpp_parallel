#include <torch/torch.h>
#include <Eigen/Dense>
#include "Network.h"
// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 10));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

class Network
// A parent class for nueral network(can be actor or critic)
// We will finish this and an implementation of fully connected nueral network
{
private:
    /* data */
public:
    Network(/* args */);
    ~Network();
};

Network::Network(/* args */)
{
}

Network::~Network()
{
}

Netowrk::evaluate()
// Input a sample data X, output a reqiured output y
{

}


int main() {
    return 0;
}