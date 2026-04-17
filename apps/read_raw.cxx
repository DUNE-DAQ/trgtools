#include "trgtools/TCEmulationUnit.hpp"
#include "trgtools/TAEmulationWorker.hpp"

#include "CLI/App.hpp"
#include "CLI/Config.hpp"
#include "CLI/Formatter.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/chrono.h>
#include <filesystem>
#include <optional>

#include "hdf5libs/HDF5RawDataFile.hpp"
#include "hdf5libs/HDF5SourceIDHandler.hpp"
#include "daqdataformats/Fragment.hpp"
#include "fddetdataformats/WIBEthFrame.hpp"


#include "triggeralgs/TriggerCandidateFactory.hpp"

//-----------------------------------------------------------------------------

using namespace dunedaq;
using namespace trgtools;


std::shared_ptr<hdf5libs::HDF5RawDataFile>
get_file(const std::string& _filename)
{
  std::shared_ptr<hdf5libs::HDF5RawDataFile> file_hdf5;

  file_hdf5 = std::make_shared<hdf5libs::HDF5RawDataFile>(_filename);
  if (!file_hdf5->is_trigger_record_type()) {
    throw std::runtime_error(fmt::format("ERROR: input file '{}' not of type 'TriggerRecord'", _filename));
  }

  std::string application_name = file_hdf5->get_attribute<std::string>("application_name");
  std::cout << "File '" << _filename << "' has application name: " << application_name << std::endl;

  return file_hdf5;
};

std::pair<uint64_t, uint64_t>
get_available_slice_id_range(const std::shared_ptr<hdf5libs::HDF5RawDataFile>& _file)
{
  if (!_file) {
    throw std::runtime_error("No file provided");
  }

  uint64_t start = std::numeric_limits<uint64_t>::max();
  uint64_t end = std::numeric_limits<uint64_t>::min();

  // Find min / max record id for this application
  auto record_ids = _file->get_all_record_ids();
  if (record_ids.empty()) {
    throw std::runtime_error("File contains no records.");
  }
  start = std::min(start, record_ids.begin()->first);
  end = std::max(end, record_ids.rbegin()->first);

  std::cout << "Start: " << start << " end: " << end << std::endl;

  return {start, end};
}

/**
 * @brief Struct with available cli application options
 */
struct Options
{
  /// @brief vector of input filenames
  std::string input_file;
};

/**
 * @brief Adds options to our CLI application
 * 
 * @param _app CLI application
 * @param _opts Struct with the available options
 */
void parse_app(CLI::App& _app, Options& _opts)
{
  _app.add_option("-i,--input-file", _opts.input_file, "Input file (one!)")
    ->required()
    ->check(CLI::ExistingFile) // Validate that each file exists
    ->type_name("FILE")
    ->expected(1); // Expect exactly one input file
}

int main(int argc, char const *argv[])
{
  // Do all the CLI processing first
  CLI::App app{"Application that reads raw data file and measures input size"};
  Options opts{};

  parse_app(app, opts);

  try {
    app.parse(argc, argv);
  }
  catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  std::cout << "File to process:\n";
  std::cout << "  " << opts.input_file << "\n";

  // Get the file
  std::shared_ptr<hdf5libs::HDF5RawDataFile> file_hdf5 =
    get_file(opts.input_file);

  // Get the available record_id range
  std::pair<uint64_t, uint64_t> recordid_range = get_available_slice_id_range(file_hdf5);
  uint64_t recordid_to_process = recordid_range.first;

  uint64_t num_wibeth_frames_total = 0;
  for (const std::string& frag_dataset : file_hdf5->get_fragment_dataset_paths(recordid_to_process)) {
    std::cout << "Fragment dataset path: " << frag_dataset << std::endl;

    // Get the fragment and print its type
    std::unique_ptr<daqdataformats::Fragment> frag_ptr = file_hdf5->get_frag_ptr(frag_dataset);
    dunedaq::daqdataformats::FragmentType frag_type = frag_ptr->get_fragment_type();
    std::cout << "Fragment type: " << daqdataformats::fragment_type_to_string(frag_type) << std::endl;

    // Only process WIBEth frames
    if (frag_type != daqdataformats::FragmentType::kWIBEth) {
      std::cout << "Fragment is not of type WIBEthFrame, skipping.\n";
      continue;
    }
    
    // Get the number of WIBEth frames in the fragment
    int num_frames = (frag_ptr->get_size() - sizeof(daqdataformats::FragmentHeader)) / sizeof(dunedaq::fddetdataformats::WIBEthFrame);
    std::cout << "Number of WIBEthFrames in fragment: " << num_frames << std::endl;

    // Gets a pointer to the first WIBEth frame in the fragment
    fddetdataformats::WIBEthFrame* frame_ptr = reinterpret_cast<fddetdataformats::WIBEthFrame*>( 
      static_cast<char*>(frag_ptr->get_data()) + sizeof(fddetdataformats::WIBEthFrame));
    
    // Print the timestamp of the first frame
    uint64_t timestamp = frame_ptr->get_timestamp();
    std::cout << "Timestamp of first frame: " << timestamp << std::endl;

    num_wibeth_frames_total++;

    std::cout << "Channel: " << frame_ptr->get_channel() << std::endl;
  }

  std::cout << "Total number of WIBEth frames: " << num_wibeth_frames_total << std::endl;
  std::cout << "s_num_channels: "           << dunedaq::fddetdataformats::WIBEthFrame::s_num_channels << " channels per frame.\n";
  std::cout << "s_num_adc_words_per_ts: "   << dunedaq::fddetdataformats::WIBEthFrame::s_num_adc_words_per_ts << " ADC words per timestamp.\n";
  std::cout << "s_time_samples_per_frame: " << dunedaq::fddetdataformats::WIBEthFrame::s_time_samples_per_frame << " time samples per frame.\n";

  std::cout << "Size of wibeth payload: " << sizeof(fddtdataformats::WIBEthFrame::adc_words) << " bytes.\n";
  std::cout << "Total size of wibeth header: " << sizeof(fddetdataformats::WIBEthFrame::header) << " bytes.\n";
  std::cout << "Total size of daq header: " << sizeof(fddetdataformats::WIBEthFrame::daq_header) << " bytes.\n";
  std::cout << "Total size of wibeth frame: " << sizeof(fddetdataformats::WIBEthFrame) << " bytes.\n";
  std::cout << "Total size validation: " << sizeof(daqdataformats::FragmentHeader) + sizeof(fddetdataformats::WIBEthFrame) << " bytes.\n";
  std::cout << "Fragment size : " << sizeof(daqdataformats::FragmentHeader) << " bytes.\n";

  return 0;
}
