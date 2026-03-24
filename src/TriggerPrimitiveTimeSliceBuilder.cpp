/* @file: TriggerPrimitiveTimeSliceBuilder.cpp
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2023.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#include "trgtools/TriggerPrimitiveTimeSliceBuilder.hpp"

#include "daqdataformats/Fragment.hpp"
#include "daqdataformats/FragmentHeader.hpp"
#include "detdataformats/DetID.hpp"

#include <algorithm>

namespace dunedaq {
namespace trgtools {

TriggerPrimitiveTimeSliceBuilder::TriggerPrimitiveTimeSliceBuilder(daqdataformats::timeslice_number_t ts_num,
                                                                   daqdataformats::run_number_t run_number,
                                                                   daqdataformats::SourceID element_id)
  : m_ts_num(ts_num)
  , m_run_number(run_number)
  , m_element_id(element_id)
{}

daqdataformats::TimeSlice&
TriggerPrimitiveTimeSliceBuilder::get_timeslice()
{
  if (!m_timeslice) {
    daqdataformats::TimeSliceHeader hdr;
    hdr.timeslice_number = m_ts_num;
    hdr.run_number = m_run_number;
    hdr.element_id = m_element_id;
    m_timeslice = std::make_unique<daqdataformats::TimeSlice>(hdr);
  }
  return *m_timeslice;
}

void
TriggerPrimitiveTimeSliceBuilder::reset_timeslice()
{
  m_timeslice.reset();
}

void
TriggerPrimitiveTimeSliceBuilder::add_fragment(daqdataformats::SourceID::ID_t element_number,
                                               const std::vector<trgdataformats::TriggerPrimitive>& tps)
{
  auto [min_it, max_it] = std::minmax_element(tps.begin(), tps.end(),
    [](const trgdataformats::TriggerPrimitive& a, const trgdataformats::TriggerPrimitive& b) {
      return a.time_start < b.time_start;
    });

  daqdataformats::FragmentHeader fh;
  fh.trigger_number = m_ts_num;
  fh.trigger_timestamp = min_it->time_start;
  fh.window_begin = min_it->time_start;
  fh.window_end = max_it->time_start;
  fh.run_number = m_run_number;
  fh.fragment_type =
    static_cast<daqdataformats::fragment_type_t>(daqdataformats::FragmentType::kTriggerPrimitive);
  fh.sequence_number = 0;
  fh.detector_id = static_cast<uint16_t>(detdataformats::DetID::Subdetector::kDAQ);
  fh.element_id = daqdataformats::SourceID(daqdataformats::SourceID::Subsystem::kTrigger, element_number);

  auto frag_ptr = std::make_unique<daqdataformats::Fragment>(
    const_cast<trgdataformats::TriggerPrimitive*>(tps.data()),
    tps.size() * sizeof(trgdataformats::TriggerPrimitive));
  frag_ptr->set_header_fields(fh);

  get_timeslice().add_fragment(std::move(frag_ptr));
}

} // namespace trgtools
} // namespace dunedaq
