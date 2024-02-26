/* @file EmulationUnit.hxx
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2023.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#ifndef TRGTOOLS_EMULATIONUNIT_HXX_
#define TRGTOOLS_EMULATIONUNIT_HXX_

namespace dunedaq {
namespace trgtools {

template <typename T, typename U, typename V>
std::unique_ptr<daqdataformats::Fragment>
EmulationUnit<T, U, V>::emulate(const std::vector<T>& inputs) {
  // Create the output.
  std::vector<U> output_buffer;

  for (const T& input : inputs) {
    (*m_maker)(input, output_buffer); // Feed TX into the TXMaker
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

  void* payload = malloc(payload_size);
  size_t payload_offset(0);
  for (const U& output : output_buffer) {
    triggeralgs::write_overlay(output, payload + payload_offset);
    payload_offset += triggeralgs::get_overlay_nbytes(output);
  }

  // Hand it to a fragment,
  std::unique_ptr<daqdataformats::Fragment> frag
    = std::make_unique<daqdataformats::Fragment>(payload, payload_size);
  // And release it.
  free(payload);

  m_last_output_buffer = output_buffer;
  return frag;
}

template <typename T, typename U, typename V>
std::vector<U>
EmulationUnit<T, U, V>::get_last_output_buffer() {
  return m_last_output_buffer;
}

} // namespace trgtools
} // namespace dunedaq

#endif // TRGTOOLS_EMULATIONUNIT_HXX_
