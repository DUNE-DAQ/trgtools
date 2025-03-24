#ifndef TRGTOOLS_TAFILEHANDLER_CPP_
#define TRGTOOLS_TAFILEHANDLER_CPP_

#include "trgtools/TAFileHandler.hpp"

namespace dunedaq::trgtools 
{

uint16_t TAFileHandler::m_id_next = 0;

TAFileHandler::TAFileHandler(std::vector<std::shared_ptr<hdf5libs::HDF5RawDataFile>> input_files,
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
  std::string algo_name = config["trigger_activity_plugin"][0];
  nlohmann::json algo_config = config["trigger_activity_config"][0];

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

  // std::set of record IDs (pair of record number & sequence number)
  auto records = m_input_files.front()->get_all_record_ids();

  // Extract the number of TAMakers to create
  daqdataformats::TimeSlice first_timeslice = m_input_files.front()->get_timeslice(*records.begin());
  std::vector<daqdataformats::SourceID> valid_sources = get_valid_sourceids(first_timeslice);
  fmt::print("Number of makers to make: {}\n", valid_sources.size());

  for (const daqdataformats::SourceID& sid : valid_sources) {
    // Create TAMaker
    std::unique_ptr<triggeralgs::TriggerActivityMaker> ta_maker =
      triggeralgs::TriggerActivityFactory::get_instance()->build_maker(algo_name);
    ta_maker->configure(algo_config);

    // Add it to the enulators
    m_ta_emulators[sid] = std::make_unique<trgtools::EmulateTAUnit>();
    m_ta_emulators[sid]->set_maker(ta_maker);

    // Create a worker thread per emulator
    if (m_run_parallel) {
      m_thread_pool.emplace_back(&TAFileHandler::worker_thread, this);
    }
  }
}

std::vector<daqdataformats::SourceID> 
TAFileHandler::get_valid_sourceids(daqdataformats::TimeSlice& _timeslice)
{
  const auto& fragments = _timeslice.get_fragments_ref();

  std::vector<daqdataformats::SourceID> ret;
  for (const auto& fragment : fragments) {
    if (fragment->get_fragment_type() != daqdataformats::FragmentType::kTriggerPrimitive) {
      continue;
    }

    daqdataformats::SourceID sourceid = fragment->get_element_id();

    ret.push_back(sourceid);
  }

  return ret;
}

hdf5libs::HDF5SourceIDHandler::source_id_geo_id_map_t
TAFileHandler::get_sourceid_geoid_map()
{
  if (!m_input_files.size()) {
    throw "Files not set yet!";
  }

  return m_input_files.front()->get_srcid_geoid_map();
}

void TAFileHandler::worker_thread()
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

void TAFileHandler::process_tasks()
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

      // Get all the fragments
      daqdataformats::TimeSlice timeslice = input_file->get_timeslice(record);
      const auto& fragments = timeslice.get_fragments_ref();

      // Iterate over the fragments & process each fragment
      for (const auto& fragment : fragments) {
        daqdataformats::SourceID sid = fragment->get_element_id();

        if (!m_ta_emulators.contains(sid)) {
          continue;
        }

        // Pull tps out
        size_t n_tps = fragment->get_data_size()/SIZE_TP;
        if (!m_quiet) {
          fmt::print("  TP fragment size: {}\n", fragment->get_data_size());
          fmt::print("  Num TPs: {}\n", n_tps);
        }

        // Create a TP buffer
        std::vector<trgdataformats::TriggerPrimitive> tp_buffer;
        // Prepare the TP buffer, checking for time ordering
        tp_buffer.reserve(n_tps);

        // Populate the TP buffer
        trgdataformats::TriggerPrimitive* tp_array = static_cast<trgdataformats::TriggerPrimitive*>(fragment->get_data());
        uint64_t last_ts = 0;
        for(size_t tpid(0); tpid<n_tps; ++tpid) {
          auto& tp = tp_array[tpid];
          if (tp.time_start <= last_ts && !m_quiet) {
            fmt::print("  ERROR: {} {} ", +tp.time_start, last_ts );
          }
          tp_buffer.push_back(tp);
        }

        daqdataformats::FragmentHeader frag_hdr = fragment->get_header();

        // Customise the source id (add 1000 to id)
        frag_hdr.element_id = daqdataformats::SourceID{daqdataformats::SourceID::Subsystem::kTrigger, fragment->get_element_id().id+1000};

        // Either enqueue the task if using parallel processing, or execute the task now
        if (m_run_parallel) {
          enqueue_task([this, sid, record, frag_hdr, tp_buffer = std::move(tp_buffer)]() mutable {
            this->process_task(sid, record.first, frag_hdr, std::move(tp_buffer));
          });
        }
        else {
          this->process_task(sid, record.first, frag_hdr, std::move(tp_buffer));
        }
      }
      // If running in parallel, wait to process entire slice before we move to
      // the next one
      if (m_run_parallel) {
        wait_to_complete_tasks();
      }
    }

    size_t total = 0;
    for (auto& [key, vec_tas]: m_tas) {
      total += vec_tas.size();
    }
    std::cout << "We have a total of " << total << " TAs!" << std::endl;
  }
}

void TAFileHandler::start_processing()
{
  m_main_thread = std::thread(&TAFileHandler::process_tasks, this);
}


void TAFileHandler::enqueue_task(std::function<void()> task)
{
  {
    std::lock_guard<std::mutex> lock(m_queue_mutex);
    m_task_queue.push(std::move(task));
    ++m_active_tasks;
  }
  m_condition.notify_one();
}

void TAFileHandler::wait_to_complete_tasks()
{
  std::unique_lock<std::mutex> lock(m_queue_mutex);
  m_task_complete_condition.wait(lock, [this]() { return m_active_tasks == 0; });
}

void TAFileHandler::wait_to_complete_work()
{
  // Wait for the main threads to join
  m_main_thread.join();
  fmt::print("TAFileHandler_{} work completed\n", m_id);

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

void TAFileHandler::process_task(daqdataformats::SourceID _source_id,
                                 uint64_t _rec,
                                 daqdataformats::FragmentHeader _header,
                                 std::vector<trgdataformats::TriggerPrimitive>&& _tps)
{
  // Get te last fragment
  std::unique_ptr<daqdataformats::Fragment> frag = m_ta_emulators[_source_id]->emulate_vector(_tps);

  // Don't do anything if no fragments found
  if (!frag) {
    return;
  }

  // Get all the TriggerActivities from the TA Emulator buffer
  std::vector<triggeralgs::TriggerActivity> ta_buffer = m_ta_emulators[_source_id]->get_last_output_buffer();

  // Don't continue if no TAs found
  size_t n_tas = ta_buffer.size();
  if (!n_tas) {
    return;
  }

  if (!m_quiet && n_tas) {
    fmt::print(" Found {} TAs!\n", n_tas);
  }

  // Set the fragment header & push into our output (with locking!)
  {
    if (m_run_parallel) {
      std::lock_guard<std::mutex> lock(m_savetps_mutex);
    }
    m_tas[_rec].reserve(m_tas[_rec].size() + ta_buffer.size());
    m_tas[_rec].insert(m_tas[_rec].end(), std::make_move_iterator(ta_buffer.begin()), std::make_move_iterator(ta_buffer.end()));

    frag->set_header_fields(_header);
    frag->set_type(daqdataformats::FragmentType::kTriggerActivity);

    m_ta_fragments[_rec].push_back(std::move(frag));
  }
}

std::map<uint64_t, std::vector<triggeralgs::TriggerActivity>> TAFileHandler::get_tas()
{
  return std::move(m_tas);
}

std::map<uint64_t, std::vector<std::unique_ptr<daqdataformats::Fragment>>> TAFileHandler::get_frags()
{
  return std::move(m_ta_fragments);
}


}; // namespace dunedaq::trgtools

#endif //TRGTOOLS_TAFILEHANDLER_CXX_
