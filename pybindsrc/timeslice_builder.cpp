/**
 * @file timeslice_builder.cpp Python bindings for TriggerPrimitiveTimeSliceBuilder
 *
 * This is part of the DUNE DAQ Software Suite, copyright 2023.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#include "trgtools/TriggerPrimitiveTimeSliceBuilder.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dunedaq::trgtools::python {

void
register_timeslice_builder(py::module& m)
{
  py::class_<TriggerPrimitiveTimeSliceBuilder>(m, "TriggerPrimitiveTimeSliceBuilder")
    .def(py::init<daqdataformats::timeslice_number_t, daqdataformats::run_number_t, daqdataformats::SourceID>(),
         py::arg("ts_num"),
         py::arg("run_number"),
         py::arg("element_id"))
    .def("get_timeslice", &TriggerPrimitiveTimeSliceBuilder::get_timeslice,
         py::return_value_policy::reference_internal)
    .def("reset_timeslice", &TriggerPrimitiveTimeSliceBuilder::reset_timeslice)
    .def("add_fragment", &TriggerPrimitiveTimeSliceBuilder::add_fragment,
         py::arg("element_number"),
         py::arg("tps"))
    .def("add_fragment",
         [](TriggerPrimitiveTimeSliceBuilder& self,
            daqdataformats::SourceID::ID_t element_number,
            py::array_t<uint64_t> version,
            py::array_t<uint64_t> flag,
            py::array_t<uint64_t> detid,
            py::array_t<uint64_t> channel,
            py::array_t<uint64_t> samples_over_threshold,
            py::array_t<uint64_t> time_start,
            py::array_t<uint64_t> samples_to_peak,
            py::array_t<uint64_t> adc_integral,
            py::array_t<uint64_t> adc_peak) {
           auto r_version                = version.unchecked<1>();
           auto r_flag                   = flag.unchecked<1>();
           auto r_detid                  = detid.unchecked<1>();
           auto r_channel                = channel.unchecked<1>();
           auto r_samples_over_threshold = samples_over_threshold.unchecked<1>();
           auto r_time_start             = time_start.unchecked<1>();
           auto r_samples_to_peak        = samples_to_peak.unchecked<1>();
           auto r_adc_integral           = adc_integral.unchecked<1>();
           auto r_adc_peak               = adc_peak.unchecked<1>();

           const py::ssize_t n = time_start.size();
           std::vector<trgdataformats::TriggerPrimitive> tps(n);
           for (py::ssize_t i = 0; i < n; ++i) {
             tps[i].version                = r_version(i);
             tps[i].flag                   = r_flag(i);
             tps[i].detid                  = r_detid(i);
             tps[i].channel                = r_channel(i);
             tps[i].samples_over_threshold = r_samples_over_threshold(i);
             tps[i].time_start             = r_time_start(i);
             tps[i].samples_to_peak        = r_samples_to_peak(i);
             tps[i].adc_integral           = r_adc_integral(i);
             tps[i].adc_peak               = r_adc_peak(i);
           }
           self.add_fragment(element_number, tps);
         },
         py::arg("element_number"),
         py::arg("version"),
         py::arg("flag"),
         py::arg("detid"),
         py::arg("channel"),
         py::arg("samples_over_threshold"),
         py::arg("time_start"),
         py::arg("samples_to_peak"),
         py::arg("adc_integral"),
         py::arg("adc_peak"))
    ;
}

} // namespace dunedaq::trgtools::python
