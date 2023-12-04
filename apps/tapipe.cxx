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
#include "trgdataformats/TriggerPrimitive.hpp"
#include "triggeralgs/HorizontalMuon/TriggerActivityMakerHorizontalMuon.hpp"

using namespace dunedaq;


class HDF5FileReader
{
private:
  std::unique_ptr<hdf5libs::HDF5RawDataFile> m_file;

public:
  HDF5FileReader(const std::string& path);
  ~HDF5FileReader();
};

HDF5FileReader::HDF5FileReader(const std::string& path)
{
    m_file = std::make_unique<hdf5libs::HDF5RawDataFile>(path);
}

HDF5FileReader::~HDF5FileReader()
{
}
//-----

template <> struct fmt::formatter<dunedaq::daqdataformats::SourceID> {
  // Presentation format: 'f' - fixed, 'e' - exponential.
  // char presentation = 'f';

  // // Parses format specifications of the form ['f' | 'e'].
  // constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
  //   // [ctx.begin(), ctx.end()) is a character range that contains a part of
  //   // the format string starting from the format specifications to be parsed,
  //   // e.g. in
  //   //
  //   //   fmt::format("{:f} - point of interest", point{1, 2});
  //   //
  //   // the range will contain "f} - point of interest". The formatter should
  //   // parse specifiers until '}' or the end of the range. In this example
  //   // the formatter should parse the 'f' specifier and return an iterator
  //   // pointing to '}'.

  //   // Please also note that this character range may be empty, in case of
  //   // the "{}" format string, so therefore you should check ctx.begin()
  //   // for equality with ctx.end().

  //   // Parse the presentation format and store it in the formatter:
  //   auto it = ctx.begin(), end = ctx.end();
  //   if (it != end && (*it == 'f' || *it == 'e')) presentation = *it++;

  //   // Check if reached the end of the range:
  //   if (it != end && *it != '}') throw_format_error("invalid format");

  //   // Return an iterator past the end of the parsed range:
  //   return it;
  // }

  // Parses format specifications of the form ['f' | 'e'].
  constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {

    // Parse the presentation format and store it in the formatter:
    // auto it = ctx.begin(), end = ctx.end();
    // if (it != end && (*it == 'f' || *it == 'e')) presentation = *it++;

    // Check if reached the end of the range:
    // if (it != end && *it != '}')
      // fmt::detail::throw_format_error("invalid format");
  
    // return it;  
    return ctx.begin();
  }
  

  // Formats the point p using the parsed format specification (presentation)
  // stored in this formatter.
  auto format(const dunedaq::daqdataformats::SourceID& sid, format_context& ctx) const -> format_context::iterator {
    return fmt::format_to(ctx.out(), "({}, {})", dunedaq::daqdataformats::SourceID::subsystem_to_string(sid.subsystem), sid.id);
  }
};


template <> struct fmt::formatter<hdf5libs::HDF5RawDataFile::record_id_t> {

  // Parses format specifications of the form ['f' | 'e'].
  constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
    return ctx.begin();
  }
  
  // Formats the point p using the parsed format specification (presentation)
  // stored in this formatter.
  auto format(const hdf5libs::HDF5RawDataFile::record_id_t& rid, format_context& ctx) const -> format_context::iterator {
    return fmt::format_to(ctx.out(), "({}, {})", rid.first, rid.second);
  }
};
//-----

int
main(int argc, char* argv[])
{
  std::string input_file;
  bool verbose = false;
  CLI::App app{"tapipe"};
  // argv = app.ensure_utf8(argv);

  app.add_option("-i", input_file, "Input TPStream file path")->required();
  app.add_flag("-v", verbose);
  CLI11_PARSE(app, argc, argv);

  fmt::print("TPStream file: {}\n", input_file);

  // Pointer to DD hdf5 file
  std::unique_ptr<hdf5libs::HDF5RawDataFile> tpstream_file;

  try {
    tpstream_file = std::make_unique<hdf5libs::HDF5RawDataFile>(input_file);
  } catch(const hdf5libs::FileOpenFailed& e)   {
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    std::cerr << e.what() << '\n';
    exit(-1);
  }

  fmt::print("{} opened\n", input_file);

  // Check the record type
  fmt::print("  File type: {}\n", tpstream_file->get_record_type());

  // Extract the list of timeslices in the file
  auto records = tpstream_file->get_all_record_ids();  
  std::set<daqdataformats::SourceID> source_ids;
  // Find what Source IDs are available in the file (all records)
  for( const auto& rid : records ) {
    const auto& [id, slice] = rid;
    auto sids = tpstream_file->get_source_ids(rid);
    source_ids.merge(sids);
    if (verbose)
      fmt::print("TR {}:{} [{}]\n", id, slice, fmt::join(sids, ", "));
  }
  fmt::print("Source IDs [{}]\n", fmt::join(source_ids, ", "));

  // 
  for( const auto& sid: source_ids ) {
    hdf5libs::HDF5RawDataFile::record_id_set recs;
    for( const auto& rid : records ) {
      auto rec_sids = tpstream_file->get_source_ids(rid);
      if (rec_sids.find(sid) != rec_sids.end()) {
        recs.insert(rid);
      }
    }
    fmt::print("Record IDs for {} : [{}]\n", sid, fmt::join(recs, ", "));
  }

  // Print the number of timeslices in the files
  fmt::print("  Number of slices: {}\n", records.size());

  // Prepare the TP buffer
  std::vector<trgdataformats::TriggerPrimitive> tp_buffer;

  auto first_slice_id = *records.begin();
  fmt::print("Processing slice {}:{}\n", first_slice_id.first, first_slice_id.second);
  auto tsl = tpstream_file->get_timeslice(first_slice_id.first);

  auto tsl_hdr = tsl.get_header();

  fmt::print("  Run number: {}\n", tsl_hdr.run_number);
  fmt::print("  TSL number: {}\n", tsl_hdr.timeslice_number);

  const auto& fragments = tsl.get_fragments_ref();
  fmt::print("  Number of fragments: {}\n", fragments.size());

  for( const auto& frag : fragments ) {
    // auto h = frag->get_header();
    fmt::print("  Fragment id: {} [{}]\n", frag->get_element_id().to_string(), daqdataformats::fragment_type_to_string(frag->get_fragment_type()));

    if ( frag->get_fragment_type() != daqdataformats::FragmentType::kTriggerPrimitive )
      continue;

    size_t n_tps = frag->get_data_size()/sizeof(trgdataformats::TriggerPrimitive);
    fmt::print("TP fragment size: {}\n", frag->get_data_size());
    fmt::print("Num TPs: {}\n", n_tps);


    trgdataformats::TriggerPrimitive* tp_array = static_cast<trgdataformats::TriggerPrimitive*>(frag->get_data());

    tp_buffer.resize(tp_buffer.size()+n_tps);

    uint64_t last_ts = 0;
    for(size_t i(0); i<n_tps; ++i) {
      auto& tp = tp_array[i];
      if (tp.time_start <= last_ts) {
        fmt::print("ERROR: {} {} ", tp.time_start, last_ts );
      }
      // fmt::print("> {}: {} {:016x} {:16x}\n", i, tp->channel, tp->time_start, tp->time_peak);
      tp_buffer.push_back(tp);
    }

    uint64_t d_ts = tp_array[n_tps-1].time_start - tp_array[0].time_start;
    fmt::print("TS gap: {} {} ms\n", d_ts, d_ts*16.0/1'000'000);
  }

  
  triggeralgs::TriggerActivityMakerHorizontalMuon hmta;
  std::vector<triggeralgs::TriggerActivity> output_tas;

  for( const auto& tp : tp_buffer ) {
    hmta(tp, output_tas);
  }

  fmt::print("output_tas.size() = {}\n", output_tas.size());
  return 0;
}
