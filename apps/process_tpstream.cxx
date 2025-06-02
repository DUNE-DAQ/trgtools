#include "trgtools/EmulateTAUnit.hpp"
#include "trgtools/EmulateTCUnit.hpp"

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

using namespace dunedaq;

class TimeSliceProcessor
{
private:
  /* data */
  std::unique_ptr<hdf5libs::HDF5RawDataFile> m_input_file;
  std::unique_ptr<hdf5libs::HDF5RawDataFile> m_output_file;

  void open_files(std::string input_path, std::string output_path);
  void close_files();

  void process( daqdataformats::TimeSlice& tls );

  // Can modify?
  std::function<void(daqdataformats::TimeSlice&)> m_processor;

public:

  TimeSliceProcessor(std::string input_path, std::string output_path);
  ~TimeSliceProcessor();

  void set_processor(std::function<void(daqdataformats::TimeSlice&)> processor);
  void loop(uint64_t num_records = 0, uint64_t offset = 0, bool quiet = false);

};

//-----------------------------------------------------------------------------
TimeSliceProcessor::TimeSliceProcessor(std::string input_path, std::string output_path)
{
  this->open_files(input_path, output_path);
}

//-----------------------------------------------------------------------------
TimeSliceProcessor::~TimeSliceProcessor()
{
  this->close_files();
}

//-----------------------------------------------------------------------------
void
TimeSliceProcessor::open_files(std::string input_path, std::string output_path) {
  // Open input file
  m_input_file = std::make_unique<hdf5libs::HDF5RawDataFile>(input_path);

  if (!m_input_file->is_timeslice_type()) {
    fmt::print("ERROR: input file '{}' not of type 'TimeSlice'\n", input_path);
    throw std::runtime_error(fmt::format("ERROR: input file '{}' not of type 'TimeSlice'", input_path));
  }

  auto run_number = m_input_file->get_attribute<daqdataformats::run_number_t>("run_number");
  auto file_index = m_input_file->get_attribute<size_t>("file_index");
  auto application_name = m_input_file->get_attribute<std::string>("application_name");

  fmt::print("Run Number: {}\nFile Index: {}\nApp name: '{}'\n", run_number, file_index, application_name);

  if (!output_path.empty()) {
    // Open output file
    m_output_file = std::make_unique<hdf5libs::HDF5RawDataFile>(
      output_path,
      m_input_file->get_attribute<daqdataformats::run_number_t>("run_number"),
      m_input_file->get_attribute<size_t>("file_index"),
      m_input_file->get_attribute<std::string>("application_name"),
      m_input_file->get_file_layout().get_file_layout_params(),
      m_input_file->get_srcid_geoid_map()
    );
  }
}

//-----------------------------------------------------------------------------
void
TimeSliceProcessor::close_files() {
  // Do something?
}

//-----------------------------------------------------------------------------
void
TimeSliceProcessor::set_processor(std::function<void(daqdataformats::TimeSlice& )> processor) {
  m_processor = processor;
}

//-----------------------------------------------------------------------------
void
TimeSliceProcessor::process( daqdataformats::TimeSlice& tls ) {
  if (m_processor)
    m_processor(tls);
}

//-----------------------------------------------------------------------------
void
TimeSliceProcessor::loop(uint64_t num_records, uint64_t offset, bool quiet) {

  // Replace with a record selection?
  auto records = m_input_file->get_all_record_ids();

  if (!num_records) {
    num_records = (records.size()-offset);
  }

  uint64_t first_rec = offset, last_rec = offset+num_records;

  uint64_t i_rec(0);
  for( const auto& rid : records ) {

    if (i_rec < first_rec || i_rec >= last_rec ) {
      ++i_rec;
      continue;
    }

    if (!quiet)
      fmt::print("\n-- Processing TSL {}:{}\n\n", rid.first, rid.second);
    auto tsl = m_input_file->get_timeslice(rid);
    // Or filter on a selection here using a lambda?

    // if (!quiet)
      // fmt::print("TSL number {}\n", tsl.get_header().timeslice_number);

    // Add a process method
    this->process(tsl);

    if (m_output_file)
      m_output_file->write(tsl);

    ++i_rec;
    if(!quiet)
      fmt::print("\n-- Finished TSL {}:{}\n\n", rid.first, rid.second);

  }

}


//-----------------------------------------------------------------------------
int main(int argc, char const *argv[])
{

  CLI::App app{"tapipe"};
  // argv = app.ensure_utf8(argv);

  std::string input_file_path;
  app.add_option("-i", input_file_path, "Input TPStream file path")->required();
  std::string output_file_path;
  app.add_option("-o", output_file_path, "Output TPStream file path");
  std::string channel_map_name = "VDColdboxTPCChannelMap";
  app.add_option("-m", channel_map_name, "Detector Channel Map");
  std::string config_name;
  app.add_option("-j", config_name, "Trigger Activity and Candidate config JSON to use.")->required();
  uint64_t skip_rec(0);
  app.add_option("-s", skip_rec, "Skip records");
  uint64_t num_rec(0);
  app.add_option("-n", num_rec, "Process records");

  bool quiet = false;
  app.add_flag("--quiet", quiet, "Quiet outputs.");

  bool latencies = false;
  app.add_flag("--latencies", latencies, "Saves latencies per TP into csv");
  CLI11_PARSE(app, argc, argv);


  if (!quiet)
    fmt::print("TPStream file: {}\n", input_file_path);

  TimeSliceProcessor rp(input_file_path, output_file_path);

  // TP source id (subsystem)
  auto tp_subsystem_requirement = daqdataformats::SourceID::Subsystem::kTrigger;

  auto channel_map = dunedaq::detchannelmaps::make_tpc_map(channel_map_name);

  // Read configuration
  std::ifstream config_stream(config_name);
  nlohmann::json config = nlohmann::json::parse(config_stream);

  // Only use the first plugin for now.
  nlohmann::json ta_algo = config["trigger_activity_plugin"][0];
  nlohmann::json ta_config = config["trigger_activity_config"][0];

  nlohmann::json tc_algo = config["trigger_candidate_plugin"][0];
  nlohmann::json tc_config = config["trigger_candidate_config"][0];


  // Finally create a TA maker
  std::unique_ptr<triggeralgs::TriggerActivityMaker> ta_maker =
    triggeralgs::TriggerActivityFactory::get_instance()->build_maker(ta_algo);
  ta_maker->configure(ta_config);
  std::unique_ptr<trgtools::EmulateTAUnit> ta_emulator = std::make_unique<trgtools::EmulateTAUnit>();
  ta_emulator->set_maker(ta_maker);
  // TODO: Use a better file naming scheme for CSV.
  if (latencies) {
    std::filesystem::path output_path(output_file_path);
    ta_emulator->set_timing_file((output_path.parent_path() / ("ta_timings_" + output_path.stem().string() + ".csv")).string());
    ta_emulator->write_csv_header("TP Time Start,TP ADC Integral,Time Diffs,Is Last TP In TA");
  }

  // Finally create a TA maker
  std::unique_ptr<triggeralgs::TriggerCandidateMaker> tc_maker =
    triggeralgs::TriggerCandidateFactory::get_instance()->build_maker(tc_algo);
  tc_maker->configure(tc_config);
  std::unique_ptr<trgtools::EmulateTCUnit> tc_emulator = std::make_unique<trgtools::EmulateTCUnit>();
  tc_emulator->set_maker(tc_maker);
  // TODO: Use a better file naming scheme for CSV.
  if (latencies) {
    std::filesystem::path output_path(output_file_path);
    tc_emulator->set_timing_file((output_path.parent_path() / ("tc_timings_" + output_path.stem().string() + ".csv")).string());
    tc_emulator->write_csv_header("Time Diffs");
  }

  // Generic filter hook
  std::function<bool(const trgdataformats::TriggerPrimitive&)> tp_filter;

  auto z_plane_filter = [&]( const trgdataformats::TriggerPrimitive& tp ) -> bool {
    return (channel_map->get_plane_from_offline_channel(tp.channel) != 2);
  };

  tp_filter = z_plane_filter;

  rp.set_processor([&]( daqdataformats::TimeSlice& tsl ) -> void {
    const std::vector<std::unique_ptr<daqdataformats::Fragment>>& frags = tsl.get_fragments_ref();
    const size_t num_frags = frags.size();
    if(!quiet)
      fmt::print("The number of fragments: {}\n", num_frags);

    uint64_t average_ta_time = 0;
    uint64_t average_tc_time = 0;

    size_t num_tas = 0;
    size_t num_tcs = 0;

    // Need a static for-loop: adding fragments to tsl will mutate frags even though it's const.
    for (size_t i = 0; i < num_frags; i++) {
      const auto& frag = frags[i];

      // The fragment has to be for the trigger (not e.g. for retreival from readout)
      if (frag->get_element_id().subsystem != tp_subsystem_requirement) {
        if(!quiet)
          fmt::print("  Warning, got non kTrigger SourceID {}\n", frag->get_element_id().to_string());
        continue;
      }

      // The fragment has to be TriggerPrimitive
      if(frag->get_fragment_type() != daqdataformats::FragmentType::kTriggerPrimitive){
        if(!quiet)
          fmt::print("  Error: FragmentType is: {}!\n", fragment_type_to_string(frag->get_fragment_type()));
        continue;
      }

      // This bit should be outside the loop
      if (!quiet)
        fmt::print("  Fragment id: {} [{}]\n", frag->get_element_id().to_string(), daqdataformats::fragment_type_to_string(frag->get_fragment_type()));

      // Pull tps out
      size_t n_tps = frag->get_data_size()/sizeof(trgdataformats::TriggerPrimitive);
      if (!quiet) {
        fmt::print("  TP fragment size: {}\n", frag->get_data_size());
        fmt::print("  Num TPs: {}\n", n_tps);
      }

      // Create a TP buffer
      std::vector<trgdataformats::TriggerPrimitive> tp_buffer;
      // Prepare the TP buffer, checking for time ordering
      tp_buffer.reserve(tp_buffer.size()+n_tps);

      // Populate the TP buffer
      trgdataformats::TriggerPrimitive* tp_array = static_cast<trgdataformats::TriggerPrimitive*>(frag->get_data());
      uint64_t last_ts = 0;
      for(size_t i(0); i<n_tps; ++i) {
        auto& tp = tp_array[i];
        if (tp.time_start <= last_ts && !quiet) {
          fmt::print("  ERROR: {} {} ", +tp.time_start, last_ts );
        }
        tp_buffer.push_back(tp);
      }

      // Print some useful info
      uint64_t d_ts = tp_array[n_tps-1].time_start - tp_array[0].time_start;
      if (!quiet)
        fmt::print("  TS gap: {} {} ms\n", d_ts, d_ts*16.0/1'000'000);

      //
      // TA Processing
      //

      const auto ta_start = std::chrono::steady_clock::now();
      std::unique_ptr<daqdataformats::Fragment> ta_frag = ta_emulator->emulate_vector(tp_buffer);
      const auto ta_end = std::chrono::steady_clock::now();

      if (ta_frag == nullptr) // Buffer was empty.
        continue;
      num_tas += ta_emulator->get_last_output_buffer().size();

      // TA time calculation.
      const uint64_t ta_diff = std::chrono::nanoseconds(ta_end - ta_start).count();
      average_ta_time += ta_diff;
      if (!quiet) {
        fmt::print("\tTA Time Process: {} ns.\n", ta_diff);
      }

      daqdataformats::FragmentHeader frag_hdr = frag->get_header();

      // Customise the source id (add 1000 to id)
      frag_hdr.element_id = daqdataformats::SourceID{daqdataformats::SourceID::Subsystem::kTrigger, frag->get_element_id().id+1000};

      ta_frag->set_header_fields(frag_hdr);
      ta_frag->set_type(daqdataformats::FragmentType::kTriggerActivity);


      tsl.add_fragment(std::move(ta_frag));
      //
      // TA Processing Ends
      //

      //
      // TC Processing
      //

      std::vector<triggeralgs::TriggerActivity> ta_buffer = ta_emulator->get_last_output_buffer();
      const auto tc_start = std::chrono::steady_clock::now();
      std::unique_ptr<daqdataformats::Fragment> tc_frag = tc_emulator->emulate_vector(ta_buffer);
      const auto tc_end = std::chrono::steady_clock::now();

      if (tc_frag == nullptr) // Buffer was empty.
        continue;
      num_tcs += tc_emulator->get_last_output_buffer().size();

      // TC time calculation.
      const uint64_t tc_diff = std::chrono::nanoseconds(tc_end - tc_start).count();
      average_tc_time += tc_diff;
      if (!quiet) {
        fmt::print("\tTC Time Process: {} ns.\n", tc_diff);
      }

      // Shares the same frag_hdr.
      tc_frag->set_header_fields(frag_hdr);
      tc_frag->set_type(daqdataformats::FragmentType::kTriggerCandidate);

      tsl.add_fragment(std::move(tc_frag));

    } // Fragment for loop

    if (num_tas == 0) average_ta_time = 0;
    else average_ta_time /= num_tas;
    if (num_tcs == 0) average_tc_time = 0;
    else average_tc_time /= num_tcs;
    if (!quiet) {
      fmt::print("\t\tAverage TA Time Process ({} TAs): {} ns.\n", num_tas, average_ta_time);
      fmt::print("\t\tAverage TC Time Process ({} TCs): {} ns.\n", num_tcs, average_tc_time);
    }
  });

  rp.loop(num_rec, skip_rec);

  /* code */
  return 0;
}
