/* @file: EmulateTAUnit.hpp
 *
 * Emulation unit for TriggerActivities.
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2023.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#ifndef TRGTOOLS_EMULATETAUNIT_HPP_
#define TRGTOOLS_EMULATETAUNIT_HPP_

#include "trgtools/EmulationUnit.hpp"
#include "trgdataformats/TriggerPrimitive.hpp"
#include "triggeralgs/TriggerActivityMaker.hpp"

namespace dunedaq {
namespace trgtools {

class EmulateTAUnit
  : public EmulationUnit<trgdataformats::TriggerPrimitive, triggeralgs::TriggerActivity, triggeralgs::TriggerActivityMaker>
{};

} // namespace trgtools
} // namespace dunedaq

#endif // TRGTOOLS_EMULATETAUNIT_HPP_
