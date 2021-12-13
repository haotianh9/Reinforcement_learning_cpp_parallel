#include <pybind11/pybind11.h>
#include "../RL3linkFishCppPortMerge/ppo_nn.h"
#include <iostream>
#include<vector>
#include<math.h>
#include <torch/torch.h>

namespace py = pybind11;


PYBIND11_MODULE(ppo_fish, m) {
    py::class_<MemoryNN>(m, "MemoryNN")
        .def(py::init<> ())
        .def("push_reward", &MemoryNN::push_reward,
            py::arg("reward"), py::arg("terminate"), py::arg("done"))
        .def("merge", &MemoryNN::merge,
            py::arg("r"))
        .def("clear", &MemoryNN::clear);

    py::class_<PPO>(m, "PPO")
        .def(py::init<int64_t, int64_t, double, double, std::tuple<double, double>, double, int64_t, double> ())
        .def("select_action", &PPO::select_action,
            py::arg("state"), py::arg("MemoryNN"))
        .def("update", &PPO::update,
            py::arg("MemoryNN"));
}