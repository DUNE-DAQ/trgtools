/* @file: TCEmulationUnit.hpp
 *
 * Emulation unit for TriggerCandidates.
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2023.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#ifndef TRGTOOLS_TCEMULATIONUNIT_HPP_
#define TRGTOOLS_TCEMULATIONUNIT_HPP_

#include "trgtools/EmulationUnit.hpp"
#include "triggeralgs/TriggerActivity.hpp"
#include "triggeralgs/TriggerCandidateMaker.hpp"

namespace dunedaq {
namespace trgtools {

class TCEmulationUnit
  : public EmulationUnit<triggeralgs::TriggerActivity,
                         triggeralgs::TriggerCandidate,
                         triggeralgs::TriggerCandidateMaker>
{};

} // namespace trgtools
} // namespace dunedaq

#endif // TRGTOOLS_TCEMULATIONUNIT_HPP_
