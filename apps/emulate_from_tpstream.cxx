#include "trgtools/EmulateTAUnit.hpp"
#include "trgtools/EmulateTCUnit.hpp"
#include "trgtools/TAFileHandler.hpp"

#include "CLI/App.hpp"
#include "CLI/Config.hpp"
#include "CLI/Formatter.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/chrono.h>
#include <filesystem>

#include "hdf5libs/HDF5RawDataFile.hpp"
#include "trgdataformats/TriggerPrimitive.hpp"
#include "triggeralgs/TriggerActivityFactory.hpp"
#include "triggeralgs/TriggerCandidateFactory.hpp"
#include "triggeralgs/TriggerObjectOverlay.hpp"
#include "detchannelmaps/TPCChannelMap.hpp"

//-----------------------------------------------------------------------------

using namespace dunedaq;
using namespace trgtools;

int main(int argc, char const *argv[])
{

  // Do all the CLI processing first
  CLI::App app{"Trigger TA & TC emulatior"};

  std::vector<std::string> input_files;
  app.add_option("-i,--input-files", input_files, "List of input files (required)")
    ->required()
    ->check(CLI::ExistingFile); // Validate that each file exists

  std::string output_file;
  app.add_option("-o,--output-file", output_file, "Output file (required)")
    ->required(); // make the argument required

  // Default 0, meaning process everything
  uint64_t milliseconds_to_process = 0;
  app.add_option("-m,--milliseconds", milliseconds_to_process, "Number of milliseconds to process");

  std::string channel_map_name = "VDColdboxChannelMap";
  app.add_option("-c,--channel-map", channel_map_name, "Detector Channel Map")
    ->check(CLI::ExistingFile);

  std::string config_name;
  app.add_option("-j,--json-config", config_name, "Trigger Activity and Candidate config JSON to use.")
    ->required()
    ->check(CLI::ExistingFile);

  size_t worker_threads = 0;
  app.add_option("-w,--worker-threads", worker_threads, "Number of worker threads to spawn (default=0, nthreads = nplanes)");

  bool ram_efficient = 0;
  app.add_option("-r,--ram-efficient", ram_efficient, "RAM-efficient mode (but slower)");

  bool quiet = false;
  app.add_flag("--quiet", quiet, "Quiet outputs.");

  bool latencies = false;
  app.add_flag("--latencies", latencies, "Saves latencies per TP into csv");

  try {
    app.parse(argc, argv);
  }
  catch (const CLI::ParseError &e) {
    return app.exit(e);
  }
  std::ifstream config_stream(config_name);
  nlohmann::json config = nlohmann::json::parse(config_stream);

  std::cout << "Files to process:\n";
  for (const std::string& file : input_files) {
    std::cout << "- " << file << "\n";
  }

  // Create the file handlers
  std::vector<std::unique_ptr<TAFileHandler>> file_handlers;
  for (const std::string& file : input_files) {
    file_handlers.push_back(std::make_unique<TAFileHandler>(file, config, worker_threads));
  }

  // Start each file handler
  for (const auto& handler : file_handlers) {
    handler->start_processing();
  }

  for (const auto& handler : file_handlers) {
    handler->wait_to_complete_work();
  }
  std::cout << "All threads joined" << std::endl;


  // Extract & sort all the TAs when ready

  return 0;
}
