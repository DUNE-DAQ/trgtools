#ifndef TRGTOOLS_TAFILEHANDLER_CPP_
#define TRGTOOLS_TAFILEHANDLER_CPP_

#include "trgtools/TAFileHandler.hpp"

namespace dunedaq::trgtools 
{

TAFileHandler::TAFileHandler(std::string input_path, nlohmann::json config, size_t n_threads)
  : m_input_path(input_path), m_num_threads(n_threads)
{
  std::string algo_name = config["trigger_activity_plugin"][0];
  nlohmann::json algo_config = config["trigger_activity_config"][0];

  // Get the input file
  m_input_file = std::make_unique<hdf5libs::HDF5RawDataFile>(input_path);
  if (!m_input_file->is_timeslice_type()) {
    fmt::print("ERROR: input file '{}' not of type 'TimeSlice'\n", input_path);
    throw std::runtime_error(fmt::format("ERROR: input file '{}' not of type 'TimeSlice'", input_path));
  }

  // Extract the run number etc
  daqdataformats::run_number_t run_number = m_input_file->get_attribute<daqdataformats::run_number_t>("run_number");
  size_t file_index = m_input_file->get_attribute<size_t>("file_index");
  std::string application_name = m_input_file->get_attribute<std::string>("application_name");

  fmt::print("Run Number: {}\nFile Index: {}\nApp name: '{}'\n", run_number, file_index, application_name);

  // std::set of record IDs (pair of record number & sequence number)
  auto records = m_input_file->get_all_record_ids();

  // Extract the number of TAMakers to create
  daqdataformats::TimeSlice first_timeslice = m_input_file->get_timeslice(*records.begin());
  size_t makers_to_make = first_timeslice.get_fragments_ref().size();

  fmt::print("Number of makers to make: {}\n", makers_to_make);

  for (size_t i = 0; i < makers_to_make; ++i) {
    std::unique_ptr<triggeralgs::TriggerActivityMaker> ta_maker =
      triggeralgs::TriggerActivityFactory::get_instance()->build_maker(algo_name);
    ta_maker->configure(algo_config);
    m_ta_emulators.push_back(std::make_unique<trgtools::EmulateTAUnit>());
    m_ta_emulators.back()->set_maker(ta_maker);
  }

  if (m_num_threads == 0) {
    m_num_threads = makers_to_make;
  }
  for (size_t i = 0; i < m_num_threads; ++i) {
    m_thread_pool.emplace_back(&TAFileHandler::worker_thread, this);
  }
}

void TAFileHandler::worker_thread()
{
  while (true) {
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

    task();

    {
      std::lock_guard<std::mutex> lock(m_queue_mutex);
      --m_active_tasks;
      if (m_active_tasks == 0) {
        m_task_complete_condition.notify_all();
      }
    }
  }
}

void TAFileHandler::process_tasks(uint64_t time, bool quiet)
{
  // std::set of record IDs (pair of record number & sequence number)
  auto records = m_input_file->get_all_record_ids();

  for (const auto& record : records) {
    daqdataformats::TimeSlice timeslice = m_input_file->get_timeslice(record);

    const auto& fragments = timeslice.get_fragments_ref();

    size_t frags_size = fragments.size();
    for (size_t i = 0; i < frags_size; ++i) {
      const auto& fragment = fragments[i];
      if (fragment->get_element_id().subsystem != daqdataformats::SourceID::Subsystem::kTrigger) {
        if (!quiet)
          fmt::print("  Warning, got non kTrigger SourceID {}\n", fragment->get_element_id().to_string());
        continue;
      }

      if (fragment->get_fragment_type() != daqdataformats::FragmentType::kTriggerPrimitive) {
        if (!quiet)
          fmt::print("  Error: FragmentType is: {}!\n", dunedaq::daqdataformats::fragment_type_to_string(fragment->get_fragment_type()));
        continue;
      }

      // Pull tps out
      size_t n_tps = fragment->get_data_size()/sizeof(trgdataformats::TriggerPrimitive);
      if (!quiet) {
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
        if (tp.time_start <= last_ts && !quiet) {
          fmt::print("  ERROR: {} {} ", tp.time_start, last_ts );
        }
        tp_buffer.push_back(tp);
      }

      enqueue_task([this, i, record, tp_buffer = std::move(tp_buffer), time, quiet]() {
          this->process_task(i, record.first, tp_buffer, time, quiet);
          });

    }
    wait_to_complete_tasks();
  }
};

void TAFileHandler::start_processing(uint64_t time, bool quiet)
{
  m_main_thread = std::thread(&TAFileHandler::process_tasks, this, time, quiet);
};


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
  std::cout << "Trying to complete work!\n";
  m_main_thread.join();
  std::cout << "Main thread joined...\n";

  wait_to_complete_tasks();

  {
    std::lock_guard<std::mutex> lock(m_queue_mutex);
    m_stop = true;
  }
  m_condition.notify_all();

  std::cout << "m_stop issued\n";
  for (std::thread& thread : m_thread_pool) {
    thread.join();
  }
}

void TAFileHandler::process_task(int thread_id, uint64_t rec, std::vector<trgdataformats::TriggerPrimitive> tps, uint64_t time, bool quiet)
{
  std::unique_ptr<daqdataformats::Fragment> tas = m_ta_emulators[thread_id]->emulate_vector(tps);
  std::vector<triggeralgs::TriggerActivity> ta_buffer = m_ta_emulators[thread_id]->get_last_output_buffer();

  if (size_t n_tas = ta_buffer.size()) {
    std::cout << " Found " << n_tas << " TAs!\n";
  }

  std::cout << "FILE: " << m_input_path << " plane: " << thread_id << " rec: " << rec << " completed!\n";
};

}; // namespace dunedaq::trgtools

#endif //TRGTOOLS_TAFILEHANDLER_HXX_
