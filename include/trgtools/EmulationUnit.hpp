/* @file: EmulationUnit.hpp
 *
 * Base abstract class for emulation units.
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2023.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#ifndef TRGTOOLS_EMULATIONUNIT_HPP_
#define TRGTOOLS_EMULATIONUNIT_HPP_

#include "daqdataformats/TimeSlice.hpp"
#include "daqdataformats/Fragment.hpp"
#include "triggeralgs/TriggerObjectOverlay.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace dunedaq {
namespace trgtools {

template <typename T, typename U, typename V>
class EmulationUnit
{
  using input_t = T;
  using output_t = U;
  using maker_t = V;

  private:
    std::unique_ptr<maker_t> m_maker; // TODO: unique may become shared later.
    std::vector<output_t> m_last_output_buffer;
    std::string m_timing_file_name;

  public:
    uint64_t emulate(const input_t& input, std::vector<output_t>& outputs);
    std::unique_ptr<daqdataformats::Fragment> emulate_vector(const std::vector<input_t>& inputs);
    std::vector<output_t> get_last_output_buffer();
    void set_maker(std::unique_ptr<maker_t>& maker) { m_maker = std::move(maker); }
    void set_timing_file(const std::string& file_name) { m_timing_file_name = file_name; }
    void write_csv_header(const std::string& header) {
      std::fstream file_header;
      file_header.open(m_timing_file_name, std::ios::out | std::ios::app);
      file_header << header << "\n";
      file_header.close();
    }
};

} // namespace trgtools
} // namespace dunedaq

#include "detail/EmulationUnit.hxx"

#endif // TRGTOOLS_EMULATIONUNIT_HPP_
