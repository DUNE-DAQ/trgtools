/**
 * @file InputFile_test.cxx Unit tests of the input-file checks of trgtools applications
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2020.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#include "trgtools/InputFile.hpp"

#include "rawdatafilelibs/InMemoryBackend.hpp"
#include "rawdatafilelibs/RawDataFileReader.hpp"
#include "rawdatafilelibs/RawDataFileWriter.hpp"

#include "daqdataformats/Fragment.hpp"
#include "daqdataformats/SourceIDGeoIDMap.hpp"
#include "daqdataformats/TimeSlice.hpp"

#define BOOST_TEST_MODULE InputFile_test // NOLINT

#include "boost/test/unit_test.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dunedaq;

namespace {

// A TimeSlice with one 32-byte fragment of the given type from the Trigger subsystem
daqdataformats::TimeSlice
make_timeslice(daqdataformats::timeslice_number_t number, daqdataformats::FragmentType fragment_type)
{
  std::vector<char> payload(32, 'x');
  auto fragment = std::make_unique<daqdataformats::Fragment>(payload.data(), payload.size());
  daqdataformats::FragmentHeader header = fragment->get_header();
  header.trigger_number = number;
  header.element_id = daqdataformats::SourceID(daqdataformats::SourceID::Subsystem::kTrigger, 0);
  header.fragment_type = static_cast<daqdataformats::fragment_type_t>(fragment_type);
  fragment->set_header_fields(header);

  daqdataformats::TimeSliceHeader tsh;
  tsh.timeslice_number = number;
  tsh.element_id = daqdataformats::SourceID(daqdataformats::SourceID::Subsystem::kTRBuilder, 0);
  daqdataformats::TimeSlice timeslice(tsh);
  timeslice.add_fragment(std::move(fragment));
  return timeslice;
}

// Writes TimeSlices with the given fragment types (one TimeSlice each, numbered from 1) to an in-memory file
std::unique_ptr<rawdatafilelibs::RawDataFileReader>
write_and_open(const std::string& name, const std::vector<daqdataformats::FragmentType>& fragment_types)
{
  rawdatafilelibs::FileMetadata metadata;
  metadata.run_number = 42;
  metadata.record_type = rawdatafilelibs::RecordType::kTimeSlice;
  rawdatafilelibs::RawDataFileWriter writer(std::make_unique<rawdatafilelibs::InMemoryWriterBackend>(),
                                            name,
                                            metadata,
                                            daqdataformats::SourceIDGeoIDMap());
  daqdataformats::timeslice_number_t number = 1;
  for (auto fragment_type : fragment_types) {
    writer.write(make_timeslice(number++, fragment_type));
  }
  writer.close();
  return std::make_unique<rawdatafilelibs::RawDataFileReader>(
    std::make_unique<rawdatafilelibs::InMemoryReaderBackend>(), name);
}

} // namespace

BOOST_AUTO_TEST_SUITE(InputFile_test)

BOOST_AUTO_TEST_CASE(TriggerPrimitivesAreFine)
{
  rawdatafilelibs::InMemoryStore::instance().clear();
  auto reader = write_and_open("tps.rdfmem",
                               { daqdataformats::FragmentType::kTriggerPrimitive,
                                 daqdataformats::FragmentType::kTriggerActivity });
  BOOST_REQUIRE_NO_THROW(trgtools::check_no_firmware_tps(*reader));
}

// The error names the file and the first record with firmware TPs
BOOST_AUTO_TEST_CASE(FirmwareTriggerPrimitivesAreRefused)
{
  rawdatafilelibs::InMemoryStore::instance().clear();
  auto reader = write_and_open("fw_tps.rdfmem",
                               { daqdataformats::FragmentType::kTriggerPrimitive,
                                 daqdataformats::FragmentType::kFW_TriggerPrimitive });
  try {
    trgtools::check_no_firmware_tps(*reader);
    BOOST_FAIL("no error for a file with FW_Trigger_Primitive fragments");
  } catch (const std::runtime_error& excpt) {
    const std::string message = excpt.what();
    BOOST_TEST_MESSAGE(message);
    BOOST_REQUIRE(message.find("fw_tps.rdfmem") != std::string::npos);
    BOOST_REQUIRE(message.find(daqdataformats::RecordID(2, 0).to_string()) != std::string::npos);
    BOOST_REQUIRE(message.find("FW_Trigger_Primitive") != std::string::npos);
    BOOST_REQUIRE(message.find("no longer supported") != std::string::npos);
  }
}

BOOST_AUTO_TEST_SUITE_END()
