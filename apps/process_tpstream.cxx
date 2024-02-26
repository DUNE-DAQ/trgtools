#include "CLI/App.hpp"
#include "CLI/Config.hpp"
#include "CLI/Formatter.hpp"

#include <fmt/core.h>
#include <fmt/format.h>

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
  std::string channel_map_name = "VDColdboxChannelMap";
  app.add_option("-m", channel_map_name, "Detector Channel Map");
  std::string config_name;
  app.add_option("-j", config_name, "Trigger Activity and Candidate config JSON to use.")->required();
  uint64_t skip_rec(0);
  app.add_option("-s", skip_rec, "Skip records");
  uint64_t num_rec(0);
  app.add_option("-n", num_rec, "Process records");

  bool quiet = false;
  app.add_flag("--quiet", quiet, "Quiet outputs.");
  CLI11_PARSE(app, argc, argv);


  if (!quiet)
    fmt::print("TPStream file: {}\n", input_file_path);

  TimeSliceProcessor rp(input_file_path, output_file_path);

  // TP source id (subsystem)
  auto tp_subsystem_requirement = daqdataformats::SourceID::Subsystem::kTrigger;

  auto channel_map = dunedaq::detchannelmaps::make_map(channel_map_name);

  // Read configuration
  std::ifstream config_stream(config_name);
  nlohmann::json config = nlohmann::json::parse(config_stream);

  // Only use the first plugin for now.
  nlohmann::json ta_algo = config["trigger_activity_plugin"][0];
  nlohmann::json ta_config = config["trigger_activity_config"][0];

  nlohmann::json tc_algo = config["trigger_candidate_plugin"][0];
  nlohmann::json tc_config = config["trigger_candidate_config"][0];


  // Finally create a TA maker
  auto ta_maker = triggeralgs::TriggerActivityFactory::get_instance()->build_maker(ta_algo);
  ta_maker->configure(ta_config);


  // Finally create a TA maker
  auto tc_maker = triggeralgs::TriggerCandidateFactory::get_instance()->build_maker(tc_algo);
  tc_maker->configure(tc_config);


  // Generic filter hook
  std::function<bool(const trgdataformats::TriggerPrimitive&)> tp_filter;

  auto z_plane_filter = [&]( const trgdataformats::TriggerPrimitive& tp ) -> bool {
    return (channel_map->get_plane_from_offline_channel(tp.channel) != 2);
  };

  tp_filter = z_plane_filter;

  rp.set_processor([&]( daqdataformats::TimeSlice& tsl ) -> void {
    const std::vector<std::unique_ptr<daqdataformats::Fragment>>& frags = tsl.get_fragments_ref();
    fmt::print("The numbert of fragments: {}\n", frags.size());
    for( const auto& frag : frags ) {

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
          fmt::print("  ERROR: {} {} ", tp.time_start, last_ts );
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
      // Create the output buffer
      std::vector<triggeralgs::TriggerActivity> ta_buffer;

      // Loop over TPs
      size_t n_tas = 0;

      for( const auto& tp : tp_buffer ) {

        if ( tp_filter(tp) ){
          // if(!quiet)
              // fmt::print("  TP filtered out!");
          continue;
        }

        (*ta_maker)(tp, ta_buffer);

        if (n_tas != ta_buffer.size()) {
          if (!quiet) {
            for( size_t i=n_tas; i<ta_buffer.size(); ++i){
              const auto& ta = ta_buffer[i];
              fmt::print("  + {} : ca_s={} ca_e={} ta_s={}, ta_e={} d_ta = {} | tp_s={} tp_s-ta_s={}\n", i, ta.channel_start, ta.channel_end, ta.time_start, ta.time_end, ta.time_end-ta.time_start, tp.time_start, (tp.time_start-ta.time_start) );
            }
          }
          n_tas = ta_buffer.size();
        }
      }

      // Count how many TAs were generated
      if (!quiet)
        fmt::print("  ta_buffer.size() = {}\n", ta_buffer.size());

      // Their size
      size_t payload_size(0);
      for ( const auto& ta : ta_buffer ) {
        payload_size += triggeralgs::get_overlay_nbytes(ta);
      }

      // Print the total size
      if (!quiet)
        fmt::print("  ta_buffer in bytes = {}\n", payload_size);

      // Could be that a TA is empty; skip as it will show up in the next fragment.
      // TC also gets skipped, but what's the use of trying to propagate a TC with no TAs?
      if (payload_size == 0) {
        if (!quiet)
          fmt::print("Skipped saving an empty TA frag.");
        continue;
      }
      
      // Create the fragment buffer
      void* payload = malloc(payload_size);

      size_t payload_offset(0);
      for ( const auto& ta : ta_buffer ) {
        triggeralgs::write_overlay(ta, payload + payload_offset);
        payload_offset += triggeralgs::get_overlay_nbytes(ta);
      }

      // Hand it to the fragment
      std::unique_ptr<daqdataformats::Fragment> ta_frag = std::make_unique<daqdataformats::Fragment>(payload, payload_size);

      // And release it
      free(payload);

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
      // Create the output buffer
      std::vector<triggeralgs::TriggerCandidate> tc_buffer;

      // Loop over TPs
      size_t n_tcs = 0;

      for( const auto& ta : ta_buffer ) {

        // if ( tp_filter(tp) ){
        //   if(!quiet)
        //       fmt::print("  TP filtered out!");
        //   continue;
        // }

        (*tc_maker)(ta, tc_buffer);

        if (n_tcs != tc_buffer.size()) {
          if (!quiet){
            for( size_t i=n_tcs; i<tc_buffer.size(); ++i){
              const auto& tc = tc_buffer[i];
                fmt::print("  + {} tc_s={}, tc_e={} | ta_s={} ta_s-tc_s={}\n", i, tc.time_start, tc.time_end, ta.time_start, (ta.time_start-tc.time_start) );
            }
          }
          n_tcs = tc_buffer.size();
        }
      }

      // Count how many TCs were generated
      if (!quiet)
        fmt::print("  tc_buffer.size() = {}\n", tc_buffer.size());

      // Their size
      size_t tc_payload_size(0);
      for ( const auto& tc : tc_buffer ) {
        tc_payload_size += triggeralgs::get_overlay_nbytes(tc);
      }

      // Print the total size
      if (!quiet)
        fmt::print("tc_buffer in bytes = {}\n", tc_payload_size);

      // Could be that a TC is empty. Skip, as it will show up in the next fragment.
      if (tc_payload_size == 0) {
        if (!quiet)
          fmt::print("Skipped saving an empty TC frag.");
        continue;
      }

      // Create the fragment buffer
      // Reuse the old payload since it's still defined.
      payload = malloc(tc_payload_size);

      if (!quiet)
        fmt::print("Post payload memory allocation.");

      // Reuse the old payload_offset.
      // TODO: TA/TC scopes should be separated, so should probably use a new function for each.
      payload_offset = 0;
      for ( const auto& tc : tc_buffer ) {
        triggeralgs::write_overlay(tc, payload + payload_offset);
        payload_offset += triggeralgs::get_overlay_nbytes(tc);
      }
      if (!quiet)
        fmt::print("Post write_overlay and offset calculation.");

      // Hand it to the fragment
      std::unique_ptr<daqdataformats::Fragment> tc_frag = std::make_unique<daqdataformats::Fragment>(payload, payload_size);
      if (!quiet)
        fmt::print("Post fragment creation.");

      // And release it
      free(payload);
      if (!quiet)
        fmt::print("Post payload freeing.");

      tc_frag->set_header_fields(frag_hdr);
      tc_frag->set_type(daqdataformats::FragmentType::kTriggerCandidate);
      if (!quiet)
        fmt::print("Post tc_frag header set.");


      tsl.add_fragment(std::move(tc_frag));
      if (!quiet)
        fmt::print("Post fragment movement.");
    }

  });

  rp.loop(num_rec, skip_rec);

  /* code */
  return 0;
}
