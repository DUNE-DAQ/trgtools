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
    /**
     * @brief Constructor, takes file input path & configuration
     * @param _quiet Do we want to print logs?
     */
    TAFileHandler(std::string input_path, 
                  nlohmann::json config,
                  std::pair<uint64_t, uint64_t> sliceid_range,
                  bool run_parallel,
                  bool quiet);

    ~TAFileHandler() = default;

    std::vector<daqdataformats::SourceID> get_valid_sourceids(daqdataformats::TimeSlice& _timeslice);
  
    /**
     * @brief User interaction for task processing
     */
    void start_processing();
                          

    /// @brief Waits for all the tasks to complete
    void wait_to_complete_work();

    /// @brief Retreives all the TAs with std::move operator
    std::map<uint64_t, std::vector<triggeralgs::TriggerActivity>> get_tas();

    /// @brief Retreives all the unique pointers to the TA fragments
    std::map<uint64_t, std::vector<std::unique_ptr<daqdataformats::Fragment>>> get_frags();

    hdf5libs::HDF5SourceIDHandler::source_id_geo_id_map_t get_sourceid_geoid_map();


  private:
    /**
     * @brief Function that processes the whole file
     */
    void process_tasks();

    /**
     * @brief Function that processes one slice for one plane
     *
     * @param _source_id
     * @param _tps
     */
    void process_task(daqdataformats::SourceID _source_id,
                      uint64_t _rec,
                      daqdataformats::FragmentHeader _header,
                      std::vector<trgdataformats::TriggerPrimitive>&& _tps);

    /// @brief
    void worker_thread();

    /**
     * @brief
     */
    void enqueue_task(std::function<void()> task);

    /// @brief
    void wait_to_complete_tasks();

  private:
    /// @brief A pointer to the input file
    std::unique_ptr<hdf5libs::HDF5RawDataFile> m_input_file;

    /// @brief configuration for the TA-makers
    nlohmann::json m_configuration;

    /// @brief input vector of tpstream input paths
    std::vector<std::string> m_input_paths;

    /// @brief Vector of TA emulators
    //std::vector<std::unique_ptr<trgtools::EmulateTAUnit>> m_ta_emulators;
    std::map<daqdataformats::SourceID, std::unique_ptr<trgtools::EmulateTAUnit>> m_ta_emulators;

    /// @brief Range of TimeSlice IDs to process
    std::pair<uint64_t, uint64_t> m_sliceid_range;

    /// @brief Run the TA makers in parllel
    const bool m_run_parallel;

    /// @brief Quiet down the cout output
    const bool m_quiet;

    /*
     * Threading objects for the main file handler
     */
    /// @brief The file handler thread
    std::thread m_main_thread;
    std::atomic<bool> m_stop{false};

    /*
     * Optional threading objects for the tasks
     * i.e. one thread per TAMaker
     */

    /// @brief Mutex for saving the TPs
    std::mutex m_savetps_mutex;
    std::vector<std::thread> m_thread_pool;
    std::condition_variable m_condition;
    std::condition_variable m_task_complete_condition;
    std::queue<std::function<void()>> m_task_queue;
    std::mutex m_queue_mutex;
    std::atomic<size_t> m_active_tasks;

    /// @brief Output vector of TAs
    std::map<uint64_t, std::vector<triggeralgs::TriggerActivity>> m_tas;

    /// @brief Output vector of TA fragments
    std::map<uint64_t, std::vector<std::unique_ptr<daqdataformats::Fragment>>> m_ta_fragments;

    uint16_t m_id;
    static uint16_t m_id_next;
    static const size_t SIZE_TP  = sizeof(trgdataformats::TriggerPrimitive);
};

};

#endif
