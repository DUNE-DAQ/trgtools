

# This is a skeleton CMakeLists.txt file, auto-generated on
# Fri Dec 1 20:19:30 CET 2023.  The developer(s) of this package should delete
# this comment as well as adding dependent targets, packages,
# etc. specific to the package. For details on how to write a package,
# please see
# https://dune-daq-sw.readthedocs.io/en/latest/packages/daq-cmake/

cmake_minimum_required(VERSION 3.12)
project(trgtools VERSION 2.2.1)

find_package(daq-cmake REQUIRED)
find_package(CLI11 REQUIRED)
find_package(fmt REQUIRED)

find_package(nlohmann_json REQUIRED)
find_package(hdf5libs REQUIRED)
find_package(trgdataformats REQUIRED)
find_package(triggeralgs REQUIRED)
find_package(detchannelmaps REQUIRED)


daq_setup_environment()


find_package(Boost COMPONENTS unit_test_framework REQUIRED)

##############################################################################


# See https://dune-daq-sw.readthedocs.io/en/latest/packages/daq-cmake/#daq_add_library

daq_add_library(*.cpp LINK_LIBRARIES hdf5libs::hdf5libs trgdataformats::trgdataformats triggeralgs::triggeralgs detchannelmaps::detchannelmaps CLI11::CLI11 fmt::fmt nlohmann_json::nlohmann_json)

##############################################################################


# See https://dune-daq-sw.readthedocs.io/en/latest/packages/daq-cmake/#daq_add_python_bindings


daq_add_python_bindings(*.cpp) 


##############################################################################


# See https://dune-daq-sw.readthedocs.io/en/latest/packages/daq-cmake/#daq_add_application

daq_add_application(trgtools_tapipe tapipe.cxx LINK_LIBRARIES ${PROJECT_NAME})
daq_add_application(trgtools_copy_tpstream copy_tpstream.cxx LINK_LIBRARIES ${PROJECT_NAME})
daq_add_application(trgtools_process_tpstream process_tpstream.cxx LINK_LIBRARIES ${PROJECT_NAME})
daq_add_application(trgtools_emulate_from_tpstream emulate_from_tpstream.cxx LINK_LIBRARIES ${PROJECT_NAME})

##############################################################################


# See https://dune-daq-sw.readthedocs.io/en/latest/packages/daq-cmake/#daq_add_unit_test

# daq_add_unit_test(TPGtools_unittest LINK_LIBRARIES)  # Should be replaced with real unit tests

##############################################################################

daq_install()

