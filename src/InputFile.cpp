/**
 * @file InputFile.cpp
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2020.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#include "trgtools/InputFile.hpp"

#include "daqdataformats/FragmentHeader.hpp"
#include "daqdataformats/RecordIndex.hpp"

#include <stdexcept>
#include <string>

namespace dunedaq::trgtools {

void
check_no_firmware_tps(const rawdatafilelibs::RawDataFileReader& reader)
{
  daqdataformats::SourceIDSelector firmware_tps;
  firmware_tps.fragment_type = daqdataformats::FragmentType::kFW_TriggerPrimitive;
  for (auto const& record_id : reader.get_record_ids()) {
    if (!reader.get_fragment_source_ids(record_id, firmware_tps).empty()) {
      throw std::runtime_error("file " + reader.get_file_name() + ", record " + record_id.to_string() +
                               ": FW_Trigger_Primitive fragments (the deprecated firmware TP format) are no longer "
                               "supported");
    }
  }
}

} // namespace dunedaq::trgtools
