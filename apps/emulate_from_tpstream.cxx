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
#include "hdf5libs/HDF5SourceIDHandler.hpp"

#include "triggeralgs/TriggerCandidateFactory.hpp"

//-----------------------------------------------------------------------------

using namespace dunedaq;
using namespace trgtools;

/**
 * @brief Saves fragments in timeslice format into output HDF5 file
 *
 * @param _outputfilename Name of the hdf5 file to save the output into
 * @param _sourceid_geoid_map sourceid--geoid map required to create a HDF5 file
 * @param _frags Map of fragments, with a vector of fragment pointers for each slice id
 * @param _quiet Do we want to quiet down the cout?
 */
void SaveFragments(const std::string& _outputfilename,
                   const hdf5libs::HDF5SourceIDHandler::source_id_geo_id_map_t& _sourceid_geoid_map,
                   std::map<uint64_t, std::vector<std::unique_ptr<daqdataformats::Fragment>>> _frags,
                   bool _quiet)
{
  // Create layout parameter object required for HDF5 creation
  hdf5libs::HDF5FileLayoutParameters layout_params;

  // Create HDF5 parameter path required for the layout
  hdf5libs::HDF5PathParameters params_trigger;
  params_trigger.detector_group_type = "Detector_Readout";
  /// @todo Maybe in the future we will want to emulate PDS TPs. 
  params_trigger.detector_group_name = "TPC";
  params_trigger.element_name_prefix = "Link";
  params_trigger.digits_for_element_number = 5;

  // Fill the HDF5 layout
  std::vector<hdf5libs::HDF5PathParameters> params;
  params.push_back(params_trigger);
  layout_params.record_name_prefix = "TimeSlice";
  layout_params.digits_for_record_number = 6;
  layout_params.digits_for_sequence_number = 0;
  layout_params.record_header_dataset_name = "TimeSliceHeader";
  layout_params.raw_data_group_name = "RawData";
  layout_params.view_group_name = "Views";
  layout_params.path_params_list = {params};

  // Create pointer to a new output HDF5 file
  std::unique_ptr<hdf5libs::HDF5RawDataFile> output_file = std::make_unique<hdf5libs::HDF5RawDataFile>(
      _outputfilename + ".hdf5",
      _frags.begin()->second[0]->get_run_number(),
      0,
      "emulate_from_tpstream",
      layout_params,
      _sourceid_geoid_map);

  // Iterate over the time slices & save all the fragments
  for (auto& [slice_id, vec_frags]: _frags) {
    // Create a new timeslice header
    daqdataformats::TimeSliceHeader tsh;
    tsh.timeslice_number = slice_id;
    tsh.run_number = vec_frags[0]->get_run_number();
    tsh.element_id = dunedaq::daqdataformats::SourceID(dunedaq::daqdataformats::SourceID::Subsystem::kTRBuilder, 0);

    // Create a new timeslice
    dunedaq::daqdataformats::TimeSlice ts(tsh);
    if (!_quiet) {
      std::cout << "Time slice number: " << slice_id << std::endl;
    }
    // Add the fragments to the timeslices
    for (std::unique_ptr<daqdataformats::Fragment>& frag_ptr: vec_frags) {
      if (!_quiet) {
        std::cout << "  Writing elementid: " << frag_ptr->get_element_id()  << " trigger number: " << frag_ptr->get_trigger_number() << " trigger_timestamp: " << frag_ptr->get_trigger_timestamp() << " window_begin: " << frag_ptr->get_window_begin() << " sequence_no: " << frag_ptr->get_sequence_number() << std::endl;
      }
      ts.add_fragment(std::move(frag_ptr));
    }

    // Write the timeslice to output file
    output_file->write(ts);
  }
}

/**
 * @brief Returns sorted map of HDF5 files per datawriter application
 *
 * @param _files: vector of strings corresponding to the input file paths
 */
std::map<std::string, std::vector<std::shared_ptr<hdf5libs::HDF5RawDataFile>>>
SortFilesPerWriter(const std::vector<std::string>& _files)
{
  std::map<std::string, std::vector<std::shared_ptr<hdf5libs::HDF5RawDataFile>>> files_sorted;

  // Put files into per-writer app groups
  for (const std::string& file : _files) {
    std::shared_ptr<hdf5libs::HDF5RawDataFile> fl = std::make_shared<hdf5libs::HDF5RawDataFile>(file);
    if (!fl->is_timeslice_type()) {
      throw std::runtime_error(fmt::format("ERROR: input file '{}' not of type 'TimeSlice'", file));
    }

    std::string application_name = fl->get_attribute<std::string>("application_name");
    files_sorted[application_name].push_back(fl);
  }

  // Sort files for each writer application individually
  for (auto& [app_name, vec_files]: files_sorted) {
    // Don't sort if we have 0 or 1 files in the application...
    if (vec_files.size() <= 1) {
      continue;
    }

    // Sort w.r.t. file index attribute
    std::sort(vec_files.begin(), vec_files.end(),
        [](const std::shared_ptr<hdf5libs::HDF5RawDataFile>& a, const std::shared_ptr<hdf5libs::HDF5RawDataFile>& b) {
        return a->get_attribute<size_t>("file_index") <
               b->get_attribute<size_t>("file_index");
        });
  }

  return files_sorted;
};

/**
 * @brief Retrieves the available slice ID range
 *
 * Finds the overlap in the slice ID range between the provided files, and
 * returns that overlap as an available range -- or crashes if there is a file
 * with a range that does not overlap.
 *
 * @todo: Rather than returning the sliceID range, should try to return a time range -- and have processors go off that.
 *
 * @param _files: a map of writer app names & vectors of HDF5 files from that application.
 * @param _quiet Do we want to quiet down the cout?
 */
std::pair<uint64_t, uint64_t>
GetAvailableSliceIDRange(const std::map<std::string, std::vector<std::shared_ptr<hdf5libs::HDF5RawDataFile>>>& _files,
                         bool _quiet)
{
  if (_files.empty()) {
    throw std::runtime_error("No files provided");
  }

  uint64_t global_start = std::numeric_limits<uint64_t>::min();
  uint64_t global_end = std::numeric_limits<uint64_t>::max();

  // Get the min & max record id per application, and the global
  for (auto& [appid, vec_files]: _files) {
    uint64_t app_start = std::numeric_limits<uint64_t>::max();
    uint64_t app_end = std::numeric_limits<uint64_t>::min();

    // Find min / max record id for this application
    for (auto& file : vec_files) {
      auto record_ids = file->get_all_record_ids();
      if (record_ids.empty()) {
        throw std::runtime_error(fmt::format("File from application {} contains no records.", appid));
      }
      app_start = std::min(app_start, record_ids.begin()->first);
      app_end = std::max(app_end, record_ids.rbegin()->first);
    }

    // Update the global min / max record id
    global_start = std::max(global_start, app_start);
    global_end = std::min(global_end, app_end);

    if (!_quiet) {
      std::cout << "Application: " << appid << " " << " TimeSliceID start: " << app_start << " end: " << app_end << std::endl;
    }
  }

  if (!_quiet) {
    std::cout << "Global start: " << global_start << " Global end: " << global_end << std::endl;
  }
  if (global_start > global_end) {
    throw std::runtime_error("One of the provided files' id range did not overlap with the rest. Please select files with overlapping TimeSlice IDs");
  }

  // Extra validation / error handling
  for (auto& [appid, vec_files]: _files) {
    for (auto& file : vec_files) {
      auto record_ids = file->get_all_record_ids();

      uint64_t file_start = record_ids.begin()->first;
      uint64_t file_end = record_ids.rbegin()->first;
      if ((file_start > global_end || file_end < global_start)) {
        uint64_t file_index = file->get_attribute<size_t>("file_index");
        throw std::runtime_error(fmt::format(
          "File from TPStreamWrite application '{}' (index '{}') has record id range [{}, {}], which does not overlap with global range [{}, {}].",
           appid, file_index, file_start, file_end, global_start, global_end
          ));
      }
    }
  }

  return {global_start, global_end};
}

/**
 * @brief Struct with available cli application options
 */
struct Options
{
  /// @brief vector of input filenames
  std::vector<std::string> input_files;
  /// @brief output filename
  std::string output_filename;
  /// @brief the configuration filename
  std::string config_name;
  /// @brief do we want to quiet down the cout? Default: no
  bool quiet = false;
  /// @brief do we want to measure latencies? Default: no
  /// @todo: Latencies currently not supported!
  bool latencies = false;
  /// @brief runs each TAMaker on a separate thread
  bool run_parallel = false;
};

/**
 * @brief Adds options to our CLI application
 * 
 * @param _app CLI application
 * @param _opts Struct with the available options
 */
void ParseApp(CLI::App& _app, Options& _opts)
{
  _app.add_option("-i,--input-files", _opts.input_files, "List of input files (required)")
    ->required()
    ->check(CLI::ExistingFile); // Validate that each file exists

  _app.add_option("-o,--output-file", _opts.output_filename, "Output file (required)")
    ->required(); // make the argument required

  _app.add_option("-j,--json-config", _opts.config_name, "Trigger Activity and Candidate config JSON to use (required)")
    ->required()
    ->check(CLI::ExistingFile);
  
  _app.add_flag("--parallel", _opts.run_parallel, "Run the TAMakers in parallel");

  _app.add_flag("--quiet", _opts.quiet, "Quiet outputs.");

  _app.add_flag("--latencies", _opts.latencies, "Saves latencies per TP into csv");
}

int main(int argc, char const *argv[])
{
  // Do all the CLI processing first
  CLI::App app{"Offline trigger TriggerActivity & TriggerCandidate emulatior"};
  Options opts{};

  ParseApp(app, opts);

  try {
    app.parse(argc, argv);
  }
  catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  // Get the configuration file
  std::ifstream config_stream(opts.config_name);
  nlohmann::json config = nlohmann::json::parse(config_stream);

  if (!opts.quiet) {
    std::cout << "Files to process:\n";
    for (const std::string& file : opts.input_files) {
      std::cout << "- " << file << "\n";
    }
  }

  // Sort the files into a map writer_id::vector<HDF5>
  std::map<std::string, std::vector<std::shared_ptr<hdf5libs::HDF5RawDataFile>>> sorted_files =
    SortFilesPerWriter(opts.input_files);

  // Get the available record_id range
  std::pair<uint64_t, uint64_t> recordid_range = GetAvailableSliceIDRange(sorted_files, opts.quiet);

  // Create the file handlers
  std::vector<std::unique_ptr<TAFileHandler>> file_handlers;
  for (auto [name, files] : sorted_files) {
    file_handlers.push_back(std::make_unique<TAFileHandler>(files, config, recordid_range, opts.run_parallel, opts.quiet));
  }

  // Start each file handler
  for (const auto& handler : file_handlers) {
    handler->start_processing();
  }

  // Output map of TA vectors & function that appends TAs to that vector
  std::map<uint64_t, std::vector<triggeralgs::TriggerActivity>> tas;
  auto append_tas = [&tas](std::map<uint64_t, std::vector<triggeralgs::TriggerActivity>>&& _tas) {
    for (auto& [sliceid, src_vec] : _tas) {
      auto& dest_vec = tas[sliceid];
      dest_vec.reserve(dest_vec.size() + src_vec.size());

      dest_vec.insert(dest_vec.end(), std::make_move_iterator(src_vec.begin()), std::make_move_iterator(src_vec.end()));
      src_vec.clear();
    }
  };

  // Output map of Fragment vectors & function that appends TAs to that vector
  std::map<uint64_t, std::vector<std::unique_ptr<daqdataformats::Fragment>>> frags;
  auto append_frags = [&frags](std::map<uint64_t, std::vector<std::unique_ptr<daqdataformats::Fragment>>>&& _frags) {
    for (auto& [sliceid, src_vec] : _frags) {
      auto& dest_vec = frags[sliceid];

      dest_vec.reserve(dest_vec.size() + src_vec.size());

      dest_vec.insert(dest_vec.end(), std::make_move_iterator(src_vec.begin()), std::make_move_iterator(src_vec.end()));
      src_vec.clear();
    }
  };

  // Iterate over the handlers, wait for them to complete their job & append
  // their TAs to our vector when ready.
  hdf5libs::HDF5SourceIDHandler::source_id_geo_id_map_t sourceid_geoid_map;
  for (const auto& handler : file_handlers) {
    // Wait for all TAs to be made
    handler->wait_to_complete_work();

    // Append output TA/fragments
    append_tas(std::move(handler->get_tas()));
    append_frags(std::move(handler->get_frags()));

    // Get and merge the source ID map
    hdf5libs::HDF5SourceIDHandler::source_id_geo_id_map_t map = handler->get_sourceid_geoid_map();
    sourceid_geoid_map.insert(map.begin(), map.end());
  }

  // Sort the TAs in each slice before pushing them into the TCMaker
  size_t n_tas = 0;
  for (auto& [sliceid, vec_tas]: tas) {
    std::sort(vec_tas.begin(), vec_tas.end(),
        [](const triggeralgs::TriggerActivity& a, const triggeralgs::TriggerActivity& b) {
        return std::tie(a.time_start, a.channel_start, a.time_end) <
               std::tie(b.time_start, b.channel_start, b.time_end);
        });
    n_tas += vec_tas.size();
  }
  if (!opts.quiet) {
    std::cout << "Total number of TAs made: " << n_tas << std::endl;
    std::cout << "Creating a TCMaker..." << std::endl;
  }
  // Create the TC emulator
  std::string algo_name = config["trigger_candidate_plugin"][0];
  nlohmann::json algo_config = config["trigger_candidate_config"][0];

  std::unique_ptr<triggeralgs::TriggerCandidateMaker> tc_maker =
    triggeralgs::TriggerCandidateFactory::get_instance()->build_maker(algo_name);
  tc_maker->configure(algo_config);

  trgtools::EmulateTCUnit tc_emulator;
  tc_emulator.set_maker(tc_maker);

  // Emulate the TriggerCandidates
  std::vector<triggeralgs::TriggerCandidate> tcs;
  for (auto& [sliceid, vec_tas]: tas) {
    std::unique_ptr<daqdataformats::Fragment> tc_frag = tc_emulator.emulate_vector(vec_tas);
    if (!tc_frag) {
      continue;
    }

    // Manipulate the fragment header
    daqdataformats::FragmentHeader frag_hdr = tc_frag->get_header();
    frag_hdr.element_id = daqdataformats::SourceID{daqdataformats::SourceID::Subsystem::kTrigger, tc_frag->get_element_id().id+10000};

    tc_frag->set_header_fields(frag_hdr);
    tc_frag->set_type(daqdataformats::FragmentType::kTriggerCandidate);
    tc_frag->set_trigger_number(sliceid);
    tc_frag->set_window_begin(frags[sliceid][0]->get_window_begin());
    tc_frag->set_window_end(frags[sliceid][0]->get_window_end());

    // Push the fragment to our list
    frags[sliceid].push_back(std::move(tc_frag));

    std::vector<triggeralgs::TriggerCandidate> tmp_tcs = tc_emulator.get_last_output_buffer();
    tcs.reserve(tcs.size() + tmp_tcs.size());
    tcs.insert(tcs.end(), std::make_move_iterator(tmp_tcs.begin()), std::make_move_iterator(tmp_tcs.end()));
  }
  if (!opts.quiet) {
    std::cout << "Total number of TCs made: " << tcs.size() << std::endl;
  }

  SaveFragments(opts.output_filename, sourceid_geoid_map, std::move(frags), opts.quiet);

  return 0;
}