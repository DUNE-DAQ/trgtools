/* @file EmulationUnit.hxx
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2023.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#ifndef TRGTOOLS_EMULATIONUNIT_HXX_
#define TRGTOOLS_EMULATIONUNIT_HXX_

#include "trgdataformats/TriggerPrimitive.hpp"

namespace dunedaq {
namespace trgtools {

template <typename T, typename U, typename V>
std::unique_ptr<daqdataformats::Fragment>
EmulationUnit<T, U, V>::emulate_vector(const std::vector<T>& inputs) {
  // Create the output.
  std::vector<U> output_buffer;
  std::vector<U> temp_buffer;

  // Create the output tp variables
  std::vector<uint64_t> time_diffs;
  // TODO: Figure out a way of saving channelid if TPs. TCs don't have it
  std::vector<uint64_t> tp_adc_integral;
  std::vector<uint64_t> tp_time_start;
  std::vector<int> is_last_tp_in_ta;
  // Only pre-allocate memory if we're actually saving the latencies
  if (!m_timing_file_name.empty()) {
    time_diffs.reserve(inputs.size());
    tp_adc_integral.reserve(inputs.size());
    tp_time_start.reserve(inputs.size());
    is_last_tp_in_ta.reserve(inputs.size());
  }

  for (const T& input : inputs) {
    size_t output_buffer_size = output_buffer.size();
    uint64_t time_diff = emulate(input, temp_buffer);
    if (temp_buffer.size() != 0) {
      output_buffer.insert(output_buffer.end(), temp_buffer.begin(), temp_buffer.end());
      temp_buffer.clear();
    }
    // 1 if it's the TP that creates a TA, 0 otherwise
    int last_tp_in_ta = (output_buffer_size == output_buffer.size()) ? 0 : 1;

    // Don't save any times if we're not saving latencies
    if (m_timing_file_name.empty())
      continue;
    time_diffs.push_back(time_diff);

    // Don't save time_start, number of TPs etc. unless it's TA latencies per TP
    if (!std::is_same<T, dunedaq::trgdataformats::TriggerPrimitive>::value)
      continue;

    tp_time_start.push_back(input.time_start);
    tp_adc_integral.push_back(input.adc_integral);
    is_last_tp_in_ta.push_back(last_tp_in_ta);
  }

  // Write the timings for each new TP in this fragment.
  if (!m_timing_file_name.empty()) {
    std::fstream timings;
    timings.open(m_timing_file_name, std::ios::out | std::ios::app);
    for (size_t i = 0; i < time_diffs.size(); i++) {
      if (std::is_same<T, dunedaq::trgdataformats::TriggerPrimitive>::value) {
        timings << tp_time_start[i] << "," << tp_adc_integral[i] << "," 
                << time_diffs[i] << "," << is_last_tp_in_ta[i] << "\n";
      }
      else {
        timings << time_diffs[i] << "\n";
      }
    }
    timings.close();
  }

  // Get the size to save on.
  size_t payload_size(0);
  for (const U& output : output_buffer) {
    payload_size += triggeralgs::get_overlay_nbytes(output);
  }

  // Don't save empty fragments.
  // The incomplete TX contents will get pushed onto the next fragment.
  if (payload_size == 0)
    return nullptr;

  // Awkward type conversion to avoid compiler complaints on void* arithmetic.
  char* payload = static_cast<char*>(malloc(payload_size));
  size_t payload_offset(0);
  for (const U& output : output_buffer) {
    triggeralgs::write_overlay(output, static_cast<void*>(payload + payload_offset));
    payload_offset += triggeralgs::get_overlay_nbytes(output);
  }

  // Hand it to a fragment,
  std::unique_ptr<daqdataformats::Fragment> frag
    = std::make_unique<daqdataformats::Fragment>(static_cast<void*>(payload), payload_size);
  // And release it.
  free(static_cast<void*>(payload));

  m_last_output_buffer = output_buffer;
  return frag;
}

template <typename T, typename U, typename V>
uint64_t
EmulationUnit<T, U, V>::emulate(const T& input, std::vector<U>& outputs) {
  auto time_start = std::chrono::steady_clock::now();
  (*m_maker)(input, outputs); // Feed TX into the TXMaker
  auto time_end = std::chrono::steady_clock::now();

  uint64_t time_diff = std::chrono::nanoseconds(time_end - time_start).count();
  return time_diff;
}

template <typename T, typename U, typename V>
std::vector<U>
EmulationUnit<T, U, V>::get_last_output_buffer() {
  return m_last_output_buffer;
}

} // namespace trgtools
} // namespace dunedaq

#endif // TRGTOOLS_EMULATIONUNIT_HXX_
