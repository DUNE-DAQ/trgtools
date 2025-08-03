#ifndef TRGTOOLS_TPFILEHANDLER_CPP_
#define TRGTOOLS_TPFILEHANDLER_CPP_

#include "trgtools/TPFileHandler.hpp"

namespace dunedaq::trgtools 
{

uint16_t TPFileHandler::m_id_next = 0;

TPFileHandler::TPFileHandler(std::vector<std::shared_ptr<hdf5libs::HDF5RawDataFile>> input_files,
                             nlohmann::json config,
                             std::pair<uint64_t, uint64_t> sliceid_range,
                             bool run_parallel,
                             bool quiet)
  : m_input_files(input_files),
    m_sliceid_range(sliceid_range),
    m_run_parallel(run_parallel),
    m_quiet(quiet),
    m_id(m_id_next++)
{
  //std::string algo_name = config["trigger_activity_plugin"][0];
  //nlohmann::json algo_config = config["trigger_activity_config"][0];

  // Get the input file
  // Extract the run number etc
  std::vector<daqdataformats::run_number_t> run_numbers;
  std::vector<size_t> file_indices;
  for (const auto& input_file : input_files) {
    if (std::find(run_numbers.begin(), run_numbers.end(),
        input_file->get_attribute<daqdataformats::run_number_t>("run_number")) ==
        run_numbers.end()) {
          run_numbers.push_back(input_file->get_attribute<daqdataformats::run_number_t>("run_number"));
    }

    if (std::find(file_indices.begin(), file_indices.end(),
        input_file->get_attribute<daqdataformats::run_number_t>("run_number")) ==
        file_indices.end()) {
          file_indices.push_back(input_file->get_attribute<size_t>("file_index"));
    }
  }

  std::string application_name = m_input_files.front()->get_attribute<std::string>("application_name");

  if (!m_quiet) {
    fmt::print("Run Numbers: {}\nFile Indices: {}\nApp name: '{}'\n", fmt::join(run_numbers, ","), fmt::join(file_indices, ","), application_name);
  }
 
  // Extract offline channel map name from json configuration
  m_offline_channel_map_name = config["offline_channel_map"][0];
  fmt::print("Using offline channel map: {}\n", m_offline_channel_map_name);

  // std::set of record IDs (pair of record number & sequence number)
  auto records = m_input_files.front()->get_all_record_ids();

  // Extract the number of TP generators to create
  daqdataformats::TriggerRecord first_record = m_input_files.front()->get_trigger_record(*records.begin());
  std::vector<std::pair<daqdataformats::SourceID, std::unique_ptr<tpglibs::TPGenerator>>> valid_sources = get_valid_sourceids(first_record, config);
  fmt::print("Number of makers to make: {}\n", valid_sources.size());
  
  // Create TP generators, add them to the emulators
  for (auto& [sid, tp_maker] : valid_sources) {
    m_tp_emulators[sid] = std::make_unique<trgtools::EmulateTPUnit>();
    m_tp_emulators[sid]->set_maker(tp_maker);

    // Create a worker thread per emulator
    if (m_run_parallel) {
      m_thread_pool.emplace_back(&TPFileHandler::worker_thread, this);
    }
  }
}

std::vector<std::pair<daqdataformats::SourceID, std::unique_ptr<tpglibs::TPGenerator>>> 
TPFileHandler::get_valid_sourceids(daqdataformats::TriggerRecord& _trigger_record, nlohmann::json config)
{
  const auto& fragments = _trigger_record.get_fragments_ref();

  std::vector<std::pair<daqdataformats::SourceID, std::unique_ptr<tpglibs::TPGenerator>>> ret;
  for (const auto& fragment : fragments) {
    if (fragment->get_fragment_type() != daqdataformats::FragmentType::kWIBEth) {
      continue;
    }

    daqdataformats::SourceID sourceid = fragment->get_element_id();

    fddetdataformats::WIBEthFrame* fr = 
	    reinterpret_cast<fddetdataformats::WIBEthFrame*>(
            static_cast<char*>(fragment->get_data()));

    auto det_id = fr->daq_header.det_id;
    auto crate_id = fr->daq_header.crate_id;
    auto slot_id = fr->daq_header.slot_id;
    auto stream_id = fr->daq_header.stream_id;

    std::shared_ptr<detchannelmaps::TPCChannelMap> offline_channel_map = 
	    dunedaq::detchannelmaps::make_tpc_map(m_offline_channel_map_name);
    std::vector<std::pair<trgdataformats::channel_t, int16_t>> channel_plane_numbers;
    channel_plane_numbers.reserve(64);

    // Initialise pedestals with ADC values from first WIB frame 
    std::vector<uint16_t> channel_pedestals;
    channel_pedestals.reserve(64);

    for (int chan = 0; chan < 64; chan++) {
      trgdataformats::channel_t off_channel = offline_channel_map->get_offline_channel_from_det_crate_slot_stream_chan(det_id, crate_id, slot_id, stream_id, chan);
    int16_t plane = offline_channel_map->get_plane_from_offline_channel(off_channel);
      channel_plane_numbers.push_back(std::make_pair(off_channel, plane));
      channel_pedestals.push_back(fr->get_adc(chan, 0));
    }

    std::list<uint16_t> pedestals(channel_pedestals.begin(), channel_pedestals.end());

    nlohmann::json tpg_configs_json = config["tpg_config"][0];

    std::vector<std::pair<std::string, nlohmann::json>> tpg_configs;
    for (const auto& it : tpg_configs_json.items()) {
      if (it.key() == "AVXFrugalPedestalSubtractProcessor") {
        if (it.value().contains("pedestals")) {
          it.value()["pedestals"] = pedestals; 
        }
      }
      tpg_configs.push_back(std::make_pair(it.key(), it.value()));
    }

    std::unique_ptr<tpglibs::TPGenerator> tp_generator = std::make_unique<tpglibs::TPGenerator>();
    tp_generator->configure(tpg_configs, channel_plane_numbers, fdreadoutlibs::types::DUNEWIBEthTypeAdapter::samples_tick_difference);

    ret.push_back(std::make_pair(sourceid, std::move(tp_generator)));
  }

  return ret;
}

hdf5libs::HDF5SourceIDHandler::source_id_geo_id_map_t
TPFileHandler::get_sourceid_geoid_map()
{
  if (!m_input_files.size()) {
    throw "Files not set yet!";
  }

  return m_input_files.front()->get_srcid_geoid_map();
}

void TPFileHandler::worker_thread()
{
  while (true) {
    /// Get a task from the queue (with locking)
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(m_queue_mutex);
      m_condition.wait(lock, [this]() {return m_stop || !m_task_queue.empty(); });

      if (m_stop && m_task_queue.empty()) {
        return;
      }

      task = std::move(m_task_queue.front());
      m_task_queue.pop();
    }

    // Run & complete a task
    task();

    // Notify that task was completed
    {
      std::lock_guard<std::mutex> lock(m_queue_mutex);
      --m_active_tasks;
      if (m_active_tasks == 0) {
        m_task_complete_condition.notify_all();
      }
    }
  }
}

void TPFileHandler::process_task(daqdataformats::SourceID _source_id,
                                 uint64_t _rec,
                                 daqdataformats::FragmentHeader _header,
                                 std::vector<fddetdataformats::WIBEthFrame*>&& _wibs)
{

  // Get te last fragment
  std::unique_ptr<daqdataformats::Fragment> frag = m_tp_emulators[_source_id]->emulate_vector_raw(_wibs);

  // Don't do anything if no fragments found
  if (!frag) {
    return;
  }
  
  // Get all the TriggerActivities from the TP Emulator buffer
  std::vector<triggeralgs::TriggerPrimitive> tp_buffer = m_tp_emulators[_source_id]->get_last_output_buffer();

  // Don't continue if no TPs found
  size_t n_tps = tp_buffer.size();
  if (!n_tps) {
    return;
  }

  if (!m_quiet && n_tps) {
    fmt::print(" Found {} TPs!\n", n_tps);
  }

  // Set the fragment header & push into our output (with locking!)
  {
    if (m_run_parallel) {
      std::lock_guard<std::mutex> lock(m_savetps_mutex);
    }
    m_tps[_rec].reserve(m_tps[_rec].size() + tp_buffer.size());
    m_tps[_rec].insert(m_tps[_rec].end(), std::make_move_iterator(tp_buffer.begin()), std::make_move_iterator(tp_buffer.end()));

    frag->set_header_fields(_header);
    frag->set_type(daqdataformats::FragmentType::kTriggerPrimitive);

    m_tp_fragments[_rec].push_back(std::move(frag));
  }

  return;
}



void TPFileHandler::process_tasks()
{
  // Iterate over the input files
  for (auto& input_file: m_input_files) {
    // std::set of record IDs (pair of record number & sequence number)
    auto records = input_file->get_all_record_ids();

    for (const auto& record : records) {
      if (record.first < m_sliceid_range.first || record.first > m_sliceid_range.second) {
        if (!m_quiet)
          fmt::print("  Will not process RecordID {} because it's outside of our range!", record.first);
        continue;
      }

      if (record.first != 1401) {
        continue;
      }
      fmt::print("  Will only process RecordID {}!\n", record.first);

      // Get all the fragments
      daqdataformats::TriggerRecord trigger_record = input_file->get_trigger_record(record);
      const auto& fragments = trigger_record.get_fragments_ref();

      // Iterate over the fragments & process each fragment
      for (const auto& fragment : fragments) {
        daqdataformats::SourceID sid = fragment->get_element_id();

        if (!m_tp_emulators.contains(sid)) {
          continue;
        }

        // Pull tps out
        size_t n_frames = fragment->get_data_size()/SIZE_WIB_FRAME;
        if (!m_quiet) {
          fmt::print("  WIBEth fragment size: {}\n", fragment->get_data_size());
          fmt::print("  Num WIB frames: {}\n", n_frames);
        }

        // Create a WIB buffer
        std::vector<fddetdataformats::WIBEthFrame*> wib_buffer;
        // Prepare the TP buffer, checking for time ordering
        wib_buffer.reserve(n_frames);

        // Populate the WIB buffer
        fddetdataformats::WIBEthFrame* wib_array = static_cast<fddetdataformats::WIBEthFrame*>(fragment->get_data());
        uint64_t last_ts = wib_array[n_frames-1].get_timestamp();
	uint64_t expected_tick_diff = fdreadoutlibs::types::DUNEWIBEthTypeAdapter::expected_tick_difference;
        for(size_t frid(0); frid<n_frames; ++frid) {
          auto& wib = wib_array[frid];
	  uint64_t current_ts = wib.get_timestamp();
          uint64_t expected_ts = last_ts - (n_frames-1-frid) * expected_tick_diff;
	  uint64_t current_tick_diff = current_ts - expected_ts;
	  if (current_tick_diff != 0 && !m_quiet) {
            fmt::print("  ERROR: {} {} ", +current_tick_diff, expected_tick_diff);
          }

          wib_buffer.push_back(&wib);
        }

        daqdataformats::FragmentHeader frag_hdr = fragment->get_header();

        // Customise the source id (add 1000 to id)
        frag_hdr.element_id = daqdataformats::SourceID{daqdataformats::SourceID::Subsystem::kTrigger, fragment->get_element_id().id+1000};

        // Either enqueue the task if using parallel processing, or execute the task now
        if (m_run_parallel) {
          enqueue_task([this, sid, record, frag_hdr, wib_buffer = std::move(wib_buffer)]() mutable {
            this->process_task(sid, record.first, frag_hdr, std::move(wib_buffer));
          });
        }
        else {
          this->process_task(sid, record.first, frag_hdr, std::move(wib_buffer));
        }
      }
      // If running in parallel, wait to process entire slice before we move to
      // the next one
      if (m_run_parallel) {
        wait_to_complete_tasks();
      }
    }

    size_t total = 0;
    for (auto& [key, vec_tps]: m_tps) {
      total += vec_tps.size();
    }
    std::cout << "We have a total of " << total << " TPs!" << std::endl;
  }
}


void TPFileHandler::start_processing()
{
  m_main_thread = std::thread(&TPFileHandler::process_tasks, this);
}

void TPFileHandler::enqueue_task(std::function<void()> task)
{
  {
    std::lock_guard<std::mutex> lock(m_queue_mutex);
    m_task_queue.push(std::move(task));
    ++m_active_tasks;
  }
  m_condition.notify_one();
}

void TPFileHandler::wait_to_complete_tasks()
{
  std::unique_lock<std::mutex> lock(m_queue_mutex);
  m_task_complete_condition.wait(lock, [this]() { return m_active_tasks == 0; });
}

void TPFileHandler::wait_to_complete_work()
{
  // Wait for the main threads to join
  m_main_thread.join();
  fmt::print("TPFileHandler_{} work completed\n", m_id);

  // Wait for the tasks to complete
  if (m_run_parallel) {
    wait_to_complete_tasks();

    {
      std::lock_guard<std::mutex> lock(m_queue_mutex);
      m_stop = true;
    }
    m_condition.notify_all();

    fmt::print("m_stop issued\n");
    for (std::thread& thread : m_thread_pool) {
      thread.join();
    }
  }
}

std::map<uint64_t, std::vector<triggeralgs::TriggerPrimitive>> TPFileHandler::get_tps()
{
  return std::move(m_tps);
}

std::map<uint64_t, std::vector<std::unique_ptr<daqdataformats::Fragment>>> TPFileHandler::get_frags()
{
  return std::move(m_tp_fragments);
}

}; // namespace dunedaq::trgtools

#endif //TRGTOOLS_TPFILEHANDLER_CXX_
