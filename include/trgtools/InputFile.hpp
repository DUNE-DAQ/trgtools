/**
 * @file InputFile.hpp Checks on the raw data files that trgtools applications read
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2020.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#ifndef TRGTOOLS_INCLUDE_TRGTOOLS_INPUTFILE_HPP_
#define TRGTOOLS_INCLUDE_TRGTOOLS_INPUTFILE_HPP_

#include "rawdatafilelibs/RawDataFileReader.hpp"

namespace dunedaq::trgtools {

/**
 * @brief Throws std::runtime_error, naming the file and the first record concerned, if the file
 * holds any FW_Trigger_Primitive fragment. That firmware TP format is deprecated and no longer
 * supported. Only the records' indices are looked at, not the fragment data.
 */
void
check_no_firmware_tps(const rawdatafilelibs::RawDataFileReader& reader);

} // namespace dunedaq::trgtools

#endif // TRGTOOLS_INCLUDE_TRGTOOLS_INPUTFILE_HPP_
