#include "CLI/App.hpp"
#include "CLI/Config.hpp"
#include "CLI/Formatter.hpp"

#include <fmt/core.h>
#include <fmt/format.h>

#include "hdf5libs/HDF5RawDataFile.hpp"

using namespace dunedaq;

int main(int argc, char const *argv[])
{

  CLI::App app{"tapipe"};
  // argv = app.ensure_utf8(argv);

  std::string input_file_path;
  app.add_option("-i", input_file_path, "Input TPStream file path")->required();
  std::string output_file_path;
  app.add_option("-o", output_file_path, "Output TPStream file path")->required();
  bool verbose = false;
  app.add_flag("-v", verbose);
  CLI11_PARSE(app, argc, argv);

  fmt::print("TPStream file: {}\n", input_file_path);

  // Pointer to DD hdf5 file
  std::unique_ptr<hdf5libs::HDF5RawDataFile> input_file, output_file;

  try {
    input_file = std::make_unique<hdf5libs::HDF5RawDataFile>(input_file_path);
  } catch(const hdf5libs::FileOpenFailed& e) {
    fmt::print("ERROR: failed to open input file '{}'\n", input_file_path);
    std::cerr << e.what() << '\n';
    exit(-1);
  }

  if (!input_file->is_timeslice_type()) {
    fmt::print("ERROR: input file '{}' not of type 'TimeSlice'\n", input_file_path);
    exit(-1);
  }

  auto run_number = input_file->get_attribute<daqdataformats::run_number_t>("run_number");
  auto file_index = input_file->get_attribute<size_t>("file_index");
  // auto creation_timestamp = input_file->get_attribute("creation_timestamp");
  auto application_name = input_file->get_attribute<std::string>("application_name");

  fmt::print("Run Number: {}\nFile Index: {}\nApp name: '{}'\n", run_number, file_index, application_name);

  try {
    output_file = std::make_unique<hdf5libs::HDF5RawDataFile>(
      output_file_path,
      input_file->get_attribute<daqdataformats::run_number_t>("run_number"),
      input_file->get_attribute<size_t>("file_index"),
      input_file->get_attribute<std::string>("application_name"),
      input_file->get_file_layout().get_file_layout_params(),
      input_file->get_srcid_geoid_map()
    );

  } catch(const hdf5libs::FileOpenFailed& e) {
    std::cout << "ERROR: failed to open output file" << std::endl;
    std::cerr << e.what() << '\n';
    exit(-1);
  }

  auto records = input_file->get_all_record_ids();

  for( const auto& rid : records ) {
    auto tsl = input_file->get_timeslice(rid);
    output_file->write(tsl);
    // Just 1, for testing
    // break;
  }


  /* code */
  return 0;
}
