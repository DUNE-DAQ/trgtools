/* @file: TriggerPrimitiveTimeSliceBuilder.hpp
 *
 * Standalone tool to build TimeSlice objects from test data.
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2023.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#ifndef TRGTOOLS_TRIGGERPRIMITIVETIMESLICEBUILDER_HPP_
#define TRGTOOLS_TRIGGERPRIMITIVETIMESLICEBUILDER_HPP_

#include "daqdataformats/SourceID.hpp"
#include "daqdataformats/TimeSlice.hpp"
#include "daqdataformats/TimeSliceHeader.hpp"
#include "daqdataformats/Types.hpp"
#include "trgdataformats/TriggerPrimitive.hpp"

#include <memory>
#include <vector>

namespace dunedaq {
namespace trgtools {

class TriggerPrimitiveTimeSliceBuilder
{
public:
  TriggerPrimitiveTimeSliceBuilder(daqdataformats::timeslice_number_t ts_num,
                                   daqdataformats::run_number_t run_number,
                                   daqdataformats::SourceID element_id);

  daqdataformats::TimeSlice& get_timeslice();

  void reset_timeslice();

  void add_fragment(daqdataformats::SourceID::ID_t element_number,
                    const std::vector<trgdataformats::TriggerPrimitive>& tps);

private:
  daqdataformats::timeslice_number_t m_ts_num;
  daqdataformats::run_number_t m_run_number;
  daqdataformats::SourceID m_element_id;
  std::unique_ptr<daqdataformats::TimeSlice> m_timeslice; ///< Staging TimeSlice for incoming Fragments
};

} // namespace trgtools
} // namespace dunedaq

#endif // TRGTOOLS_TRIGGERPRIMITIVETIMESLICEBUILDER_HPP_
