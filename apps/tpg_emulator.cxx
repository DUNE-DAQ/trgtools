/**
 * @file tapipe.cxx
 *
 * Developer(s) of this DAQ application have yet to replace this line with a brief description of the application.
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2020.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#include "CLI/App.hpp"
#include "CLI/Config.hpp"
#include "CLI/Formatter.hpp"

#include <fmt/core.h>
#include <fmt/format.h>

#include "hdf5libs/HDF5RawDataFile.hpp"
#include "detchannelmaps/TPCChannelMap.hpp"

#include "fdreadoutlibs/DUNEWIBEthTypeAdapter.hpp"
#include "fddetdataformats/WIBEthFrame.hpp"
#include "trgdataformats/TriggerPrimitive.hpp"
#include "tpglibs/TPGenerator.hpp"


using namespace dunedaq;

void tpgemu_config(auto& frag_ptr, auto& info, bool& verbose) 
{
  dunedaq::fddetdataformats::WIBEthFrame* fr =
    reinterpret_cast<dunedaq::fddetdataformats::WIBEthFrame*>(
    static_cast<char*>(frag_ptr->get_data()));

  info.m_det_id = fr->daq_header.det_id;
  info.m_crate_id  = fr->daq_header.crate_id;
  info.m_slot_id   = fr->daq_header.slot_id;
  info.m_stream_id =  fr->daq_header.stream_id;
}

void tpgemu_process(auto& frag_ptr, auto& info, 
		    auto& m_channel_map, auto& m_tpg_configs, 
		    bool& verbose)
{
  int16_t num_frames = frag_ptr->get_data_size() / sizeof(dunedaq::fddetdataformats::WIBEthFrame);

  auto fp_first_frame = reinterpret_cast<dunedaq::fdreadoutlibs::types::DUNEWIBEthTypeAdapter*>(
                        static_cast<char*>(frag_ptr->get_data()));


  std::vector<std::pair<trgdataformats::channel_t, int16_t>> m_channel_plane_numbers;
  m_channel_plane_numbers.reserve(64);
  for (int chan = 0; chan < 64; chan++) {
    trgdataformats::channel_t off_channel = m_channel_map->get_offline_channel_from_det_crate_slot_stream_chan(info.m_det_id, info.m_crate_id, info.m_slot_id, info.m_stream_id, chan);
    int16_t plane = m_channel_map->get_plane_from_offline_channel(off_channel);
    m_channel_plane_numbers.push_back(std::make_pair(off_channel, plane));
  }
  std::unique_ptr<tpglibs::TPGenerator> m_tp_generator = std::make_unique<tpglibs::TPGenerator>();
  m_tp_generator->configure(m_tpg_configs, m_channel_plane_numbers, dunedaq::fdreadoutlibs::types::DUNEWIBEthTypeAdapter::samples_tick_difference);

  for (int16_t ifr = 0; ifr < num_frames; ifr++) {
    auto fp = reinterpret_cast<dunedaq::fdreadoutlibs::types::DUNEWIBEthTypeAdapter*>(
              static_cast<char*>(frag_ptr->get_data()) + ifr * sizeof(dunedaq::fddetdataformats::WIBEthFrame));

    auto wfptr = reinterpret_cast<dunedaq::fddetdataformats::WIBEthFrame*>((uint8_t*)fp); // NOLINT

    std::vector<trgdataformats::TriggerPrimitive> tps = (*m_tp_generator)(wfptr);

    for (const auto& tp : tps) {
      if (verbose) std::cout << tp.channel << "," << tp.time_start << "," << tp.samples_over_threshold << "," << tp.samples_to_peak << "," << tp.adc_peak << "," << tp.adc_integral << "," << tp.detid << "\n";
    }
  }
}

int
main(int argc, char* argv[])
{
  bool verbose = false;
  CLI::App app{"tpgemu"};
  // argv = app.ensure_utf8(argv);

  std::string input_file_path;
  app.add_option("-i", input_file_path, "Input Trigger Record file path")->required();
  std::string output_file_path;
  app.add_option("-o", output_file_path, "Output TPStream/TR file path")->required();
  std::string channel_map_name = "VDColdboxTPCChannelMap";
  app.add_option("-m", channel_map_name, "Detector Channel Map");
  int trigger_number = -1;
  app.add_option("-n", trigger_number, "Trigger number to analyse. Default: -1 (all trigger records).");

  app.add_flag("-v", verbose);
  CLI11_PARSE(app, argc, argv);

  fmt::print("TPStream file: {}\n", input_file_path);

  // Pointer to DD hdf5 file
  std::unique_ptr<hdf5libs::HDF5RawDataFile> input_file, output_file;
  int tr_first = 0, tr_last = 0, num_trs = 0;

  try {
    input_file = std::make_unique<hdf5libs::HDF5RawDataFile>(input_file_path);
  } catch(const hdf5libs::FileOpenFailed& e) {
    fmt::print("ERROR: failed to open input file '{}'\n", input_file_path);
    std::cerr << e.what() << '\n';
    exit(-1);
  }

  if (!input_file->is_trigger_record_type()) {
    fmt::print("ERROR: input file '{}' not of type 'TimeSlice'\n", input_file_path);
    exit(-1);
  } else {
    auto records = input_file->get_all_record_ids();
    auto first_rec = *(records.begin());
    auto all_rh_paths = input_file->get_record_header_dataset_paths();
    auto trh_ptr = input_file->get_trh_ptr(first_rec);
    tr_first = trh_ptr->get_header().trigger_number;
    tr_last = input_file->get_trh_ptr(all_rh_paths.back())->get_header().trigger_number;
    num_trs = tr_last - tr_first + 1;
  }

  auto run_number = input_file->get_attribute<daqdataformats::run_number_t>("run_number");
  auto file_index = input_file->get_attribute<size_t>("file_index");
  // auto creation_timestamp = input_file->get_attribute("creation_timestamp");
  auto application_name = input_file->get_attribute<std::string>("application_name");

  fmt::print("Run Number: {}\nFile Index: {}\nApp name: '{}'\n", run_number, file_index, application_name);
  fmt::print("First Trigger Record: {}\nLast Trigger Record: {}\nNumber of Trigger Records: {}\n", tr_first, tr_last, num_trs);

  struct {
    uint32_t m_det_id = 0;    // NOLINT(build/unsigned)
    uint32_t m_crate_id = 0;  // NOLINT(build/unsigned)
    uint32_t m_slot_id = 0;   // NOLINT(build/unsigned)
    uint32_t m_stream_id = 0; // NOLINT(build/unsigned)
  } info;

  std::shared_ptr<dunedaq::detchannelmaps::TPCChannelMap> m_channel_map;
  m_channel_map = dunedaq::detchannelmaps::make_tpc_map(channel_map_name);
  // move to file
  std::vector<std::pair<std::string, nlohmann::json>> m_tpg_configs;
  std::string name = "";
  name = "AVXFrugalPedestalSubtractProcessor";
  nlohmann::json json_pedsub;
  json_pedsub["accum_limit"] = 10;
  m_tpg_configs.push_back(std::make_pair(name, json_pedsub));
  name = "AVXThresholdProcessor";
  nlohmann::json json;
  json["plane0"] = 90;
  json["plane1"] = 90;
  json["plane2"] = 90;
  m_tpg_configs.push_back(std::make_pair(name, json));


  for (auto const& rid : input_file->get_all_record_ids()) {

    auto trh_ptr = input_file->get_trh_ptr(rid);
    int tr_num = trh_ptr->get_header().trigger_number;

    if (trigger_number > 0 && tr_num != trigger_number) {
      continue;
    }

    int elmid = 0;
    int num_frags = 0;
    for(auto const& frag_dataset : input_file->get_fragment_dataset_paths(rid)) {
      auto frag_ptr = input_file->get_frag_ptr(frag_dataset);

      if (frag_ptr->get_data_size() == 0) {
         continue;
      }

      if (frag_ptr->get_fragment_type() != dunedaq::daqdataformats::FragmentType::kWIBEth) {
        continue;
      }

      elmid = frag_ptr->get_header().element_id.id;
      if (verbose) std::cout << "INFO elmid " << elmid << "\n";
      tpgemu_config(frag_ptr, info, verbose);
      tpgemu_process(frag_ptr, info, m_channel_map, m_tpg_configs, verbose);

      num_frags++;

    } // fragments
    fmt::print("Number of fragments: {}\n", num_frags);

  } // records




  return 0;
}
