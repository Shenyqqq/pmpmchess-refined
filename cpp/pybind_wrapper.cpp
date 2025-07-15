#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <tuple>
#include <stdexcept>
#include "GameInterface.h"
#include "MCTS.h"

namespace py = pybind11;

PYBIND11_MODULE(katago_cpp_core, m) {
    m.doc() = "C++ core module for Katago-like project";

    // 定义一个健壮的类型别名，用于处理来自 Python 的 NumPy 浮点数数组
    using py_array_float = py::array_t<float, py::array::c_style | py::array::forcecast>;

    py::class_<GameInterface>(m, "Game")
        .def(py::init<int, int>(), py::arg("n") = 9, py::arg("max_rounds") = 50)
        .def("getInitialBoard", &GameInterface::getInitialBoard)
        .def("getBoardSize", &GameInterface::getBoardSize)
        .def("getActionSize", &GameInterface::getActionSize)

        .def("getNextState", [](const GameInterface& g, py_array_float b, int p, int a) {
        return g.getNextState({ b.data(), b.data() + b.size() }, p, a);
            })
        .def("getValidMoves", [](const GameInterface& g, py_array_float b, int p) {

        return py::array(py::cast(g.getValidMoves({ b.data(), b.data() + b.size() }, p)));
            })
        .def("getGameEnded", [](const GameInterface& g, py_array_float b, int p) {
        return g.getGameEnded({ b.data(), b.data() + b.size() }, p);
            })
        .def("getCanonicalForm", [](const GameInterface& g, py_array_float b, uint64_t h, int p) {
        return g.getCanonicalForm({ b.data(), b.data() + b.size() }, h, p);
            })
        .def("getScore", [](const GameInterface& g, py_array_float b, int p) {
        return g.getScore({ b.data(), b.data() + b.size() }, p);
            })
        .def("getSymmetries", [](const GameInterface& g, py_array_float b, py_array_float pi) {
        return g.getSymmetries({ b.data(), b.data() + b.size() }, { pi.data(), pi.data() + pi.size() });
            })
        .def_static("setLogging", &GameInterface::setLogging, "Enable/disable C++ side logging for Arena");


    py::class_<MCTSArgs>(m, "MCTSArgs")
        .def(py::init<>())
        .def_readwrite("numMCTSSims", &MCTSArgs::numMCTSSims)
        .def_readwrite("cpuct", &MCTSArgs::cpuct)
        .def_readwrite("dirichletAlpha", &MCTSArgs::dirichletAlpha)
        .def_readwrite("epsilon", &MCTSArgs::epsilon)
        .def_readwrite("factor_winloss", &MCTSArgs::factor_winloss);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<const GameInterface&, py::object, const MCTSArgs&>())
        .def("getActionProbs", &MCTS::getActionProbs,
            py::arg("canonicalBoards"), py::arg("hashes"), py::arg("seeds"), py::arg("temp") = 1.0f,
            py::return_value_policy::move);
}
