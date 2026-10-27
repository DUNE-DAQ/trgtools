#include "CLI/App.hpp"
#include "CLI/Config.hpp"
#include "CLI/Formatter.hpp"

#include <fmt/core.h>
#include <fmt/format.h>

#include "trgtools/InputFile.hpp"
#include "trgtools/OutputFile.hpp"

#include "hdf5libs/HDF5Backend.hpp"
#include "rawdatafilelibs/BackendFactory.hpp"
#include "rawdatafilelibs/RawDataFileReader.hpp"

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

using namespace dunedaq;

int
main(int argc, char const* argv[])
{

  CLI::App app{ "tapipe" };
  // argv = app.ensure_utf8(argv);

  std::string input_file_path;
  app.add_option("-i", input_file_path, "Input TPStream file path")->required();
  std::string output_file_path;
  app.add_option("-o", output_file_path, "Output TPStream file path")->required();
  bool overwrite = false;
  app.add_flag("--overwrite", overwrite, "Replace the output file if it exists: it is deleted before the new one is written. An unfinished <output>.writing file is never replaced");
  bool verbose = false;
  app.add_flag("-v", verbose);
  CLI11_PARSE(app, argc, argv);

  fmt::print("TPStream file: {}\n", input_file_path);

  std::unique_ptr<rawdatafilelibs::RawDataFileReader> input_file;
  try {
    input_file = rawdatafilelibs::open_raw_data_file(input_file_path);
  } catch (const ers::Issue& e) {
    fmt::print("ERROR: failed to open input file '{}'\n", input_file_path);
    std::cerr << e.what() << '\n';
    exit(-1);
  }

  if (input_file->get_record_type() != rawdatafilelibs::RecordType::kTimeSlice) {
    fmt::print("ERROR: input file '{}' not of type 'TimeSlice'\n", input_file_path);
    exit(-1);
  }
  try {
    trgtools::check_no_firmware_tps(*input_file);
  } catch (const std::runtime_error& e) {
    fmt::print("ERROR: {}\n", e.what());
    exit(-1);
  }

  auto const& input_metadata = input_file->get_metadata();
  fmt::print("Run Number: {}\nFile Index: {}\nApp name: '{}'\n",
             input_metadata.run_number,
             input_metadata.file_index,
             input_metadata.application_name);

  // the copy keeps the input's metadata (but gets its own creation time) and, for HDF5, its file layout
  rawdatafilelibs::FileMetadata output_metadata = input_metadata;
  output_metadata.creation_timestamp_ms = 0;
  output_metadata.closing_timestamp_ms = 0;
  output_metadata.recovery_timestamp_ms = 0;
  output_metadata.stats = {};
  std::optional<hdf5libs::HDF5FileLayoutParameters> layout;
  if (auto* hdf5_input = dynamic_cast<hdf5libs::HDF5ReaderBackend*>(&input_file->get_backend())) {
    layout = hdf5_input->get_file_layout().get_file_layout_params();
  }

  std::unique_ptr<rawdatafilelibs::RawDataFileWriter> output_file;
  try {
    trgtools::prepare_output_path(output_file_path, overwrite);
    output_file =
      trgtools::create_hdf5_timeslice_file(output_file_path, output_metadata, input_file->get_geo_id_map(), layout);
  } catch (const std::exception& e) {
    std::cout << "ERROR: failed to open output file" << std::endl;
    std::cerr << e.what() << '\n';
    exit(-1);
  }

  for (const auto& rid : input_file->get_record_ids()) {
    auto tsl = input_file->get_timeslice(rid);
    output_file->write(tsl);
    // Just 1, for testing
    // break;
  }
  output_file->close();

  return 0;
}
