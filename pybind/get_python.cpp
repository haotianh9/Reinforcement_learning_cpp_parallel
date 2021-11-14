//
// Created by Qiongao Liu on 2021/11/14.
//

#include <pybind11/embed.h>
#include <iostream>

namespace py = pybind11;

int main() {
    py::scoped_interpreter python;

    //py::module sys = py::module::import("sys");
    //py::print(sys.attr("path"));

    py::module t = py::module::import("get_python");
    t.attr("add")(1, 2);
    return 0;
}