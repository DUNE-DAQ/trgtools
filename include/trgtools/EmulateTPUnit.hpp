/* @file: EmulateTPUnit.hpp
 *
 * Emulation unit for TriggerActivities.
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2023.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#ifndef TRGTOOLS_EMULATETPUNIT_HPP_
#define TRGTOOLS_EMULATETPUNIT_HPP_

#include "trgtools/EmulationUnit.hpp"
#include "trgdataformats/TriggerPrimitive.hpp"
#include "triggeralgs/TriggerActivityMaker.hpp"
#include "fdreadoutlibs/DUNEWIBEthTypeAdapter.hpp"
#include "tpglibs/TPGenerator.hpp"

namespace dunedaq {
namespace trgtools {

class EmulateTPUnit
  : public EmulationUnit<fddetdataformats::WIBEthFrame*, trgdataformats::TriggerPrimitive, tpglibs::TPGenerator>
{};

} // namespace trgtools
} // namespace dunedaq

#endif // TRGTOOLS_EMULATETPUNIT_HPP_
