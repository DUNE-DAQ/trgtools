/**
 * @file OutputFile_test.cxx Unit tests of the output-file helpers of trgtools applications
 *
 * Set TRGTOOLS_TEST_OUTPUT_DIR to keep a copy of the TP TimeSlice file written by
 * TimeSliceFileWithAGivenLayout there (e.g. as input for trying the applications).
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2020.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#include "trgtools/OutputFile.hpp"

#include "hdf5libs/HDF5Backend.hpp"
#include "rawdatafilelibs/BackendFactory.hpp"
#include "trgdataformats/TriggerPrimitive.hpp"

#include "daqdataformats/Fragment.hpp"
#include "daqdataformats/TimeSlice.hpp"

#define BOOST_TEST_MODULE OutputFile_test // NOLINT

#include "boost/test/unit_test.hpp"

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dunedaq;

namespace {

std::string
make_test_dir(const std::string& name)
{
  auto dir = std::filesystem::temp_directory_path() / ("trgtools_" + name + "_" + std::to_string(getpid()));
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir.string();
}

void
write_text(const std::string& path, const std::string& text)
{
  std::ofstream(path) << text;
}

std::string
read_text(const std::string& path)
{
  std::ifstream file(path);
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// A TimeSlice with one fragment of TPs from the Trigger subsystem, as a TP stream holds them
daqdataformats::TimeSlice
make_tp_timeslice(daqdataformats::timeslice_number_t number, int tp_count)
{
  std::vector<trgdataformats::TriggerPrimitive> tps(tp_count);
  for (int idx = 0; idx < tp_count; ++idx) {
    tps[idx].time_start = 1000 * number + idx;
    tps[idx].channel = idx;
  }
  auto fragment = std::make_unique<daqdataformats::Fragment>(tps.data(), tps.size() * sizeof(tps[0]));
  daqdataformats::FragmentHeader header = fragment->get_header();
  header.trigger_number = number;
  header.run_number = 42;
  header.element_id = daqdataformats::SourceID(daqdataformats::SourceID::Subsystem::kTrigger, 0); // as tapipe expects
  header.fragment_type = static_cast<daqdataformats::fragment_type_t>(daqdataformats::FragmentType::kTriggerPrimitive);
  fragment->set_header_fields(header);

  daqdataformats::TimeSliceHeader tsh;
  tsh.timeslice_number = number;
  tsh.run_number = 42;
  tsh.element_id = daqdataformats::SourceID(daqdataformats::SourceID::Subsystem::kTRBuilder, 0);
  daqdataformats::TimeSlice timeslice(tsh);
  timeslice.add_fragment(std::move(fragment));
  return timeslice;
}

} // namespace

BOOST_AUTO_TEST_SUITE(OutputFile_test)

BOOST_AUTO_TEST_CASE(NewOutputIsFine)
{
  const auto dir = make_test_dir("NewOutputIsFine");
  BOOST_REQUIRE_NO_THROW(trgtools::check_output_path(dir + "/out.hdf5", false));
  BOOST_REQUIRE_NO_THROW(trgtools::prepare_output_path(dir + "/out.hdf5", false));
  std::filesystem::remove_all(dir);
}

// An existing output is refused unless overwrite is set, in which case it is removed
BOOST_AUTO_TEST_CASE(ExistingOutputNeedsOverwrite)
{
  const auto dir = make_test_dir("ExistingOutputNeedsOverwrite");
  const std::string output = dir + "/out.hdf5";
  write_text(output, "old output");

  BOOST_REQUIRE_THROW(trgtools::check_output_path(output, false), std::runtime_error);
  BOOST_REQUIRE_THROW(trgtools::prepare_output_path(output, false), std::runtime_error);
  BOOST_REQUIRE_EQUAL(read_text(output), "old output");

  BOOST_REQUIRE_NO_THROW(trgtools::check_output_path(output, true));
  BOOST_REQUIRE(std::filesystem::exists(output)); // checking removes nothing
  BOOST_REQUIRE_NO_THROW(trgtools::prepare_output_path(output, true));
  BOOST_REQUIRE(!std::filesystem::exists(output));
  std::filesystem::remove_all(dir);
}

// An unfinished "<output>.writing" is never removed, even with overwrite, and nothing else is either
BOOST_AUTO_TEST_CASE(UnfinishedOutputIsNeverRemoved)
{
  const auto dir = make_test_dir("UnfinishedOutputIsNeverRemoved");
  const std::string output = dir + "/out.hdf5";
  write_text(output + ".writing", "unfinished");
  write_text(output, "old output");

  for (bool overwrite : { false, true }) {
    BOOST_REQUIRE_THROW(trgtools::check_output_path(output, overwrite), std::runtime_error);
    BOOST_REQUIRE_THROW(trgtools::prepare_output_path(output, overwrite), std::runtime_error);
  }
  BOOST_REQUIRE_EQUAL(read_text(output + ".writing"), "unfinished");
  BOOST_REQUIRE_EQUAL(read_text(output), "old output");
  std::filesystem::remove_all(dir);
}

// The HDF5 TimeSlice file gets the given layout and keeps the TPs byte for byte
BOOST_AUTO_TEST_CASE(TimeSliceFileWithAGivenLayout)
{
  const auto dir = make_test_dir("TimeSliceFileWithAGivenLayout");
  const std::string output = dir + "/tpstream.hdf5";

  hdf5libs::HDF5FileLayoutParameters layout =
    hdf5libs::default_file_layout_parameters(rawdatafilelibs::RecordType::kTimeSlice);
  layout.digits_for_record_number = 7; // something other than the default

  rawdatafilelibs::FileMetadata metadata;
  metadata.run_number = 42;
  metadata.file_index = 3;
  metadata.application_name = "OutputFile_test";
  daqdataformats::SourceIDGeoIDMap geo_id_map;
  {
    auto writer = trgtools::create_hdf5_timeslice_file(output, metadata, geo_id_map, layout);
    for (daqdataformats::timeslice_number_t number = 1; number <= 3; ++number) {
      writer->write(make_tp_timeslice(number, 5));
    }
    writer->close();
  }

  auto reader = rawdatafilelibs::open_raw_data_file(output);
  BOOST_REQUIRE(reader->get_record_type() == rawdatafilelibs::RecordType::kTimeSlice);
  BOOST_REQUIRE_EQUAL(reader->get_metadata().file_index, 3);
  BOOST_REQUIRE_EQUAL(reader->get_record_ids().size(), 3);
  auto stored_layout =
    dynamic_cast<hdf5libs::HDF5ReaderBackend&>(reader->get_backend()).get_file_layout().get_file_layout_params();
  BOOST_REQUIRE_EQUAL(stored_layout.to_json().dump(), layout.to_json().dump());

  auto timeslice = reader->get_timeslice(daqdataformats::RecordID{ 2 });
  auto expected = make_tp_timeslice(2, 5);
  BOOST_REQUIRE_EQUAL(timeslice.get_fragments_ref().size(), 1);
  auto const& fragment = *timeslice.get_fragments_ref().front();
  auto const& expected_fragment = *expected.get_fragments_ref().front();
  BOOST_REQUIRE_EQUAL(fragment.get_data_size(), expected_fragment.get_data_size());
  BOOST_REQUIRE(std::memcmp(fragment.get_data(), expected_fragment.get_data(), fragment.get_data_size()) == 0);
  reader->close();

  if (const char* keep_dir = std::getenv("TRGTOOLS_TEST_OUTPUT_DIR")) {
    // a plain byte copy: std::filesystem::copy_file can fail on shared/host-mounted directories
    std::ifstream source(output, std::ios::binary);
    std::ofstream copy(std::filesystem::path(keep_dir) / "OutputFile_test_tpstream.hdf5",
                       std::ios::binary | std::ios::trunc);
    copy << source.rdbuf();
    BOOST_REQUIRE(copy.good());
  }
  std::filesystem::remove_all(dir);
}

BOOST_AUTO_TEST_SUITE_END()
