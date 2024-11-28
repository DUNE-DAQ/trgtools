#ifndef TRGTOOLS_TAFILEHANDLER_HPP_
#define TRGTOOLS_TAFILEHANDLER_HPP_

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


namespace dunedaq::trgtools 
{

class TAFileHandler
{
  public:
    TAFileHandler(std::string input_path, nlohmann::json config, size_t n_threads = 0);

    ~TAFileHandler() = default;
  
    void start_processing(uint64_t time = 0, bool quiet=false);

    void wait_to_complete_work();

  private:
    void process_tasks(uint64_t time, bool quiet);

    void process_task(int thread_id, uint64_t rec, std::vector<trgdataformats::TriggerPrimitive> tps, uint64_t time, bool quiet);

    void worker_thread();

    void enqueue_task(std::function<void()> task);

    void wait_to_complete_tasks();

  private:
    /// @brief A pointer to the input file
    std::unique_ptr<hdf5libs::HDF5RawDataFile> m_input_file;

    /// @brief configuration for the TA-makers
    nlohmann::json m_configuration;

    /// Vector of TA-processors
    std::vector<std::function<void(daqdataformats::Fragment&)>> m_processors;
    std::vector<std::unique_ptr<trgtools::EmulateTAUnit>> m_ta_emulators;

    /// Output vector of TAs
    std::vector<triggeralgs::TriggerActivity> m_trigger_activities;

    std::string m_input_path;

    std::thread m_main_thread;
    std::vector<std::thread> m_thread_pool;
    std::queue<std::function<void()>> m_task_queue;
    std::mutex m_queue_mutex;
    std::condition_variable m_condition;
    std::condition_variable m_task_complete_condition;
    std::atomic<bool> m_stop{false};
    std::atomic<size_t> m_active_tasks;
    size_t m_num_threads;
};
};

#endif
