//
// Created by Qiongao Liu on 2021/11/14.
//

// cmake .
// make

#include <pybind11/pybind11.h>
namespace py = pybind11;

int add(int i, int j)
{
    return i + j;
}

PYBIND11_MODULE(get_cpp, m)
{
    // optional module docstring
    m.doc() = "pybind11 example plugin";
    // expose add function, and add keyword arguments and default arguments
    m.def("add", &add, "A function which adds two numbers", py::arg("i")=1, py::arg("j")=2);

    // exporting variables
    m.attr("the_answer") = 42;
    py::object world = py::cast("World");
    m.attr("what") = world;
}