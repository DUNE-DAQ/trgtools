#include "trgtools/EmulateTAUnit.hpp"
#include "trgtools/EmulateTCUnit.hpp"

#include "CLI/App.hpp"
#include "CLI/Config.hpp"
#include "CLI/Formatter.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/chrono.h>
#include <filesystem>
#include <fstream>

#include "hdf5libs/HDF5RawDataFile.hpp"
#include "trgdataformats/TriggerPrimitive.hpp"
#include "triggeralgs/TriggerActivityFactory.hpp"
#include "triggeralgs/TriggerCandidateFactory.hpp"
#include "triggeralgs/TriggerObjectOverlay.hpp"
#include "detchannelmaps/TPCChannelMap.hpp"
#include "fddetdataformats/DAPHNEFrame.hpp"

#include "TH1D.h"
#include "TFile.h"
#include "TNtupleD.h"
#include "TH2D.h"
#include "TCanvas.h"
#include "TPaveText.h"
#include "TLine.h"
#include "TGraph.h"
#include "TMultiGraph.h"
//using DAPHNEFrame = fddetdataformats::DAPHNEFrame;

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

  std::cout<< "Hello, this is me breaking this code 1" << std::endl;
  std::cout << "m_input_file->is_timeslice_type() "<< m_input_file->is_timeslice_type() << std::endl;
  std::cout << "m_input_file->get_record_type() "<< m_input_file->get_record_type() << std::endl;
  std::cout << "m_input_file->is_trigger_record_type() "<< m_input_file->is_trigger_record_type() << std::endl;
  if (!m_input_file->is_trigger_record_type()) {
    fmt::print("ERROR: input file '{}' not of type 'TriggerRecord'\n", input_path);
    throw std::runtime_error(fmt::format("ERROR: input file '{}' not of type 'TriggerRecord'", input_path));
  }

  auto run_number = m_input_file->get_attribute<daqdataformats::run_number_t>("run_number");
  auto file_index = m_input_file->get_attribute<size_t>("file_index");
  auto application_name = m_input_file->get_attribute<std::string>("application_name");

  std::cout<< "Hello, this is me breaking this code 2" << std::endl;
  std::cout << "run_number "<< run_number << std::endl;
  std::cout << "file_index) "<< file_index << std::endl;
  std::cout << "application_name) "<< application_name << std::endl;

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
  std::cout<< "Hello, this is me breaking this code 3" << std::endl;

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
void Print(fddetdataformats::DAPHNEFrame &tp,int run, int trgnum, int first)
{
  TH1D *h = new TH1D("h",Form("h ch%i SlotID: %i- Run: %i, Trigger %i",
                    tp.header.channel,tp.daq_header.slot_id,run,
                    trgnum),1024,0,1024);
    for(int j=0;j<1024;j++) h->SetBinContent(j+1,tp.get_adc(j));
  TCanvas *c = new TCanvas("c");
  c->Divide(2,1);
  c->cd(1);
  h->Draw("HIST");
  std::vector<TLine*> tl1, tl2, tl3, tl4;
  TLine* baseline= new TLine(0,tp.header.get_baseline(),
                          1000,tp.header.get_baseline());
  baseline->SetLineStyle(3);
  baseline->SetLineColor(37);
  baseline->SetLineWidth(2);
  baseline->Draw();
  for(int i=0; tp.get_da(i);i++)
  {
    tl1.push_back(new TLine(64+tp.get_time_peak(i),tp.header.get_baseline(),
                         64+tp.get_time_peak(i),tp.header.get_baseline()-tp.get_max_peak(i)));
    tl2.push_back(new TLine(tp.get_time_peak(i),h->GetMinimum(),tp.get_time_peak(i),h->GetMaximum()));

    tl3.push_back(new TLine(64+tp.get_time_peak(0),tp.header.get_baseline(),
                         64+tp.get_time_peak(0)+tp.get_time_pulse(0),tp.header.get_baseline()));
    tl4.push_back(new TLine(64+tp.get_time_peak(0)+tp.get_time_pulse(0),tp.header.get_baseline(),
                         64+tp.get_time_peak(0)+tp.get_time_pulse(0)+tp.get_time_pulse_ob(0),tp.header.get_baseline()));
    tl1[i]->SetLineColor(2);
    tl1[i]->SetLineWidth(2);
    tl2[i]->SetLineColor(1);
    tl2[i]->SetLineWidth(2);
    tl3[i]->SetLineColor(3);	
    tl3[i]->SetLineWidth(2);        
    tl4[i]->SetLineColor(4);
    tl4[i]->SetLineWidth(2);
    tl1[i]->Draw();
    //tl2[i]->Draw();
    tl3[i]->Draw();tl4[i]->Draw();
  }
  long long int q=0;
  for(int i=tp.get_time_peak(0);i<tp.get_time_peak(0)+tp.get_time_pulse(0);i++) q+=h->GetBinContent(i+1)-tp.header.get_baseline();
//  std::cout << q << std::endl;

  c->cd(2);
  TPaveText *pt[5]; float titwidth=0.2;
  TPaveText *pttit= new TPaveText(0.0,0.0,titwidth,1.0);
  pttit->AddText("TP");
  pttit->AddText("DA");
  pttit->AddText("charge");
  pttit->AddText("max_peak");((TText*)pttit->GetListOfLines()->Last())->SetTextColor(2);
  pttit->AddText("time_peak");((TText*)pttit->GetListOfLines()->Last())->SetTextColor(2);
  pttit->AddText("time_pulse");((TText*)pttit->GetListOfLines()->Last())->SetTextColor(3);
  pttit->AddText("time_pulse_ob");((TText*)pttit->GetListOfLines()->Last())->SetTextColor(4);
  pttit->AddText("peak_ub");
  pttit->AddText("peak_ob");
  pttit->Draw();
  for (size_t i=0;i<5;i++)
  {
    pt[i]= new TPaveText(titwidth+(1.0-titwidth)/5*(i),0.0,titwidth+(1.0-titwidth)/5*(i+1),1.0);
    pt[i]->AddText(Form("%li",i));
    pt[i]->AddText(Form("%i",tp.get_da(i)));
    pt[i]->AddText(Form("%i",tp.get_charge(i)));
    pt[i]->AddText(Form("%i",tp.get_max_peak(i)));
    pt[i]->AddText(Form("%i",tp.get_time_peak(i)));
    pt[i]->AddText(Form("%i",tp.get_time_pulse(i)));
    pt[i]->AddText(Form("%i",tp.get_time_pulse_ob(i)));
    pt[i]->AddText(Form("%i",tp.get_num_peak_ub(i)));
    pt[i]->AddText(Form("%i",tp.get_num_peak_ob(i)));
    pt[i]->Draw();
  }
  if(first==1){ c->Print("wvf.pdf(","pdf");}
  if(first==0) { c->Print("wvf.pdf","pdf");}
  if(first==-1) { c->Print("wvf.pdf)","pdf");}
}
//-----------------------------------------------------------------------------
int main(int argc, char const *argv[])
{
  std::cout<< "Hello, this is me breaking this code 4 " << std::endl;

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
  int num_wav(0);
  app.add_option("-p", num_wav, "Print waveforms to pdf");

  bool quiet = false;
  app.add_flag("--quiet", quiet, "Quiet outputs.");

  bool latencies = false;
  app.add_flag("--latencies", latencies, "Saves latencies per TP into csv");
  CLI11_PARSE(app, argc, argv);


  if (!quiet)
    fmt::print("TPStream file: {}\n", input_file_path);

  TimeSliceProcessor rp(input_file_path, output_file_path);

  // TP source id (subsystem)
  auto tp_subsystem_requirement = daqdataformats::SourceID::Subsystem::kDetectorReadout;
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
  std::unique_ptr<triggeralgs::TriggerActivityMaker> ta_maker =
  triggeralgs::TriggerActivityFactory::get_instance()->build_maker(ta_algo);
  ta_maker->configure(ta_config);
  std::unique_ptr<trgtools::EmulateTAUnit> ta_emulator = std::make_unique<trgtools::EmulateTAUnit>();
  ta_emulator->set_maker(ta_maker);
  
  // TODO: Use a better file naming scheme for CSV.
  if (latencies) {
    std::filesystem::path output_path(output_file_path);
    ta_emulator->set_timing_file((output_path.parent_path() / ("ta_timings_" + output_path.stem().string() + ".csv")).string());
//    ta_emulator->write_csv_header("TP Time Start,TP ADC Integral,Time Diffs,Is Last TP In TA");
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
//    tc_emulator->write_csv_header("Time Diffs");
  }

  // Generic filter hook
  std::cout << " Entro aquí" <<std::endl;
  std::function<bool(const trgdataformats::TriggerPrimitive&)> tp_filter;
  
  auto z_plane_filter = [&]( const trgdataformats::TriggerPrimitive& tp ) -> bool {
    std::cout << "he2" << std::endl;
    return (channel_map->get_plane_from_offline_channel(tp.channel) != 2);
  };

  tp_filter = z_plane_filter;

  int pdfcounter=0; 
  TNtupleD ntp_tp("ntp_tp","ntp_tp","ev:ch:i:amp:charg:time_start:time_peak:time_ovth");
  TNtupleD ntp_ta("ntp_ta","ntp_ta","ev:nch:ntp:amp:charg:time_start:time_peak:time_end:duration");
  TNtupleD ntp_ev("ntp_ev","ntp_ev","ev:ntp:nta");
  double myAbsolutt0=0;

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
//        if(!quiet)
//          fmt::print("  Warning, got non kTrigger SourceID {}\n", frag->get_element_id().to_string());
        continue;
      }

      // The fragment has to be Daphne!
      if(frag->get_fragment_type() != dunedaq::daqdataformats::FragmentType::kDAPHNE){
//        if(!quiet)
//          fmt::print("  Error: FragmentType is: {}!\n", fragment_type_to_string(frag->get_fragment_type()));
        continue;
      }

      // This bit should be outside the loop
      if (!quiet)
      {
        std::cout << "i "<< i<<std::endl;
        std::cout <<  "Printing fragment: " << frag->get_header() << std::endl;
        fmt::print("  Fragment id: {} [{}]\n", frag->get_element_id().to_string(), daqdataformats::fragment_type_to_string(frag->get_fragment_type()));
      }

      // Pull tps out
//       static const size_t FrameSize = sizeof(DAPHNEFrame);
//      auto frame = reinterpret_cast<DAPHNEFrame*>(
//              static_cast<uint8_t*>(frag->get_data()) + i*FrameSize);
      size_t n_tps = frag->get_data_size()/sizeof(fddetdataformats::DAPHNEFrame);
      if (!quiet) {
        fmt::print("  DaphneFragment Size: {}\n", frag->get_data_size());
        fmt::print("  Num DaphneFrames: {}\n", n_tps);
      }

      // Create a TP buffer
      std::vector<dunedaq::trgdataformats::TriggerPrimitive> tp_buffer;
      // Prepare the TP buffer, checking for time ordering
      tp_buffer.reserve(tp_buffer.size()+n_tps);

      // Populate the TP buffer
//      [](WIB2Frame& self) -> const DaphneFrame::Trailer& {return self.trailer;})
      std::cout << "get_data_size " << frag->get_data_size() << std::endl;
      fddetdataformats::DAPHNEFrame* tp_array = static_cast<fddetdataformats::DAPHNEFrame*>(frag->get_data());
      std::map<int,int> ch;
      for(size_t i(0); i<n_tps; ++i) {
        auto& tp = tp_array[i]; ch[tp.get_channel()]=1;
          if (!quiet){
            std::cout << tp.daq_header << std::endl;
            fmt::print("  ch: {} ", tp.get_channel() );
            fmt::print("  Baseline: {}, DAs ({},{},{},{},{})\n", tp.header.get_baseline(),tp.get_da(0),tp.get_da(1)
            ,tp.get_da(2),tp.get_da(3),tp.get_da(4));
//            for(size_t j=0; j<5;j++) tp.Print(j);
          }
          /*===============FIXED TO READ ONLY FIRST TP SINCE THE OTHER ARE WRONG!!!=================*/
          for(size_t j=0; j<5;j++) if(tp.get_da(j)==1){
            dunedaq::trgdataformats::TriggerPrimitive thistp=tp.get_TP(j);
            tp_buffer.push_back(thistp);

          }
          if(num_wav>0)
          {
            if(pdfcounter==0) {Print(tp,frag->get_header().run_number,frag->get_header().trigger_number,1);pdfcounter++;}
            if(num_wav>pdfcounter){ Print(tp,frag->get_header().run_number,frag->get_header().trigger_number,0);pdfcounter++;}
            if(num_wav==pdfcounter){ Print(tp,frag->get_header().run_number,frag->get_header().trigger_number,-1);pdfcounter++;}
          }
      }
      //TPs do not come in order in the hdf5, but they will come ordered in the DAQ! Let's order them!
      std::vector<int> sorted;
      std::size_t n(0);
      sorted.resize(n_tps);
      std::generate(std::begin(sorted), std::end(sorted), [&]{ return n++; });
      std::sort(  std::begin(tp_buffer), std::end(tp_buffer), [&](dunedaq::trgdataformats::TriggerPrimitive i1, dunedaq::trgdataformats::TriggerPrimitive i2) { return i1.get_time_peak() < i2.get_time_peak(); }); 

      std::cout << "Num channels: " << ch.size() << std::endl;
      std::cout << "# of TPs: " << tp_buffer.size() <<std::endl;
      // Print some useful info
      uint64_t d_ts = tp_array[n_tps-1].get_timestamp() - tp_array[0].get_timestamp();
      if (!quiet)
        fmt::print("  TS gap: {} {} ms\n", d_ts, d_ts*16.0/1'000'000);

      //
      // TA Processing
      //

      const auto ta_start = std::chrono::steady_clock::now();
      std::unique_ptr<daqdataformats::Fragment> ta_frag = ta_emulator->emulate_vector(tp_buffer);
      const auto ta_end = std::chrono::steady_clock::now();
      
      const uint64_t ta_diff = std::chrono::nanoseconds(ta_end - ta_start).count();
      average_ta_time += ta_diff;
      if (!quiet) {
        fmt::print("\tTA Time Process: {} ns.\n", ta_diff);
      }

      double t0=tp_buffer[0].get_time_peak();
      if(myAbsolutt0==0)myAbsolutt0=tp_buffer[0].get_time_start();

      for(auto &thistp : tp_buffer) ntp_tp.Fill(frag->get_header().trigger_number,thistp.get_channel(),0,static_cast<float>(thistp.get_adc_peak()), thistp.get_adc_integral(),thistp.get_time_peak()-myAbsolutt0,thistp.get_time_start()-myAbsolutt0,thistp.get_time_over_threshold());
                  
      if (ta_frag == nullptr) // Buffer was empty.
        continue;
      auto ta_list =  ta_emulator->get_last_output_buffer();
      
      num_tas += ta_list.size();
      std::cout<< "#of TAs: " << ta_list.size() << std::endl;

      if(ta_list[0].time_peak<t0)
      {
        //It seems that the TP buffer was not emptied. Remove first TA. First TA correspond to the previous event!
          std::cout << "ERROR!- " << ta_list[0].time_start <<  " "  << ta_list[0].time_peak << std::endl;
          ta_list.erase(ta_list.begin(), ta_list.begin()+1);
      }
            
      int ii=0;
      auto myt0 = ta_list[0].time_start;
      for(auto &ta_bit : ta_list)
      {
        std::map<int,int> channelcounter;
        if(!quiet) std::cout << " TA " << ii << " "<< ta_bit.inputs.size()<< " "<< ta_bit.adc_integral << "  " << ta_bit.time_start-myt0 <<  " " << ta_bit.time_end-myt0 <<  " TimePeak: " << ta_bit.time_peak   << std::endl;

        for(auto &mytp : ta_bit.inputs)
        {
           channelcounter[mytp.channel]++;
        }
        ntp_ta.Fill(frag->get_header().trigger_number,channelcounter.size(),ta_bit.inputs.size(),ta_bit.adc_peak,ta_bit.adc_integral,ta_bit.time_start-myAbsolutt0,ta_bit.time_peak-myAbsolutt0,ta_bit.time_end-myAbsolutt0,ta_bit.time_end-ta_bit.time_start);
        ii++;
      }
      ntp_ev.Fill(frag->get_header().trigger_number,tp_buffer.size(),ta_list.size());
    // TA time calculation.

      daqdataformats::FragmentHeader frag_hdr = frag->get_header();

      // Customise the source id (add 1000 to id)
      frag_hdr.element_id = daqdataformats::SourceID{daqdataformats::SourceID::Subsystem::kTrigger, frag->get_element_id().id+1000};

      ta_frag->set_header_fields(frag_hdr);
      ta_frag->set_type(daqdataformats::FragmentType::kTriggerActivityPDS); // keep or change to kTriggerActivityPDS?


      tsl.add_fragment(std::move(ta_frag));
      //
      // TA Processing Ends
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

//      std::cout << "Enter 0 to exit, any other to continue. " << std::endl;
//      int nn;std::cin >> nn; if (nn==0) std::exit(0);

    } // Fragment for loop

    if (num_tas == 0) average_ta_time = 0;
    else average_ta_time /= num_tas;
    
    fmt::print("\t\tAverage TA Time Process ({} TAs): {} ns.\n", num_tas, average_ta_time);
    
  });

  rp.loop(num_rec, skip_rec);
  TFile tf("out.root","RECREATE");
  tf.cd(); ntp_ta.Write("ntp_ta");ntp_tp.Write("ntp_tp");ntp_ev.Write("ntp_ev");
  tf.Close();

  /* code */
  return 0;
}