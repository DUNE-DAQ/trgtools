/**
 * @file OutputFile.cpp
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2020.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#include "trgtools/OutputFile.hpp"

#include "hdf5libs/HDF5Backend.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace dunedaq::trgtools {

void
check_output_path(const std::string& path, bool overwrite)
{
  const std::string inprogress_path = path + rawdatafilelibs::RawDataFileWriter::s_default_inprogress_suffix;
  if (std::filesystem::exists(inprogress_path)) {
    throw std::runtime_error("an unfinished output file " + inprogress_path +
                             " exists (left by a crash, or still being written); it is never removed automatically: "
                             "recover it (rawdatafile_recover) or delete it, then try again");
  }
  if (std::filesystem::exists(path) && !overwrite) {
    throw std::runtime_error("the output file " + path + " already exists; use --overwrite to replace it");
  }
}

void
prepare_output_path(const std::string& path, bool overwrite)
{
  check_output_path(path, overwrite);
  if (std::filesystem::exists(path)) {
    std::filesystem::remove(path);
  }
}

std::unique_ptr<rawdatafilelibs::RawDataFileWriter>
create_hdf5_timeslice_file(const std::string& path,
                           rawdatafilelibs::FileMetadata metadata,
                           const daqdataformats::SourceIDGeoIDMap& geo_id_map,
                           const std::optional<hdf5libs::HDF5FileLayoutParameters>& layout)
{
  metadata.record_type = rawdatafilelibs::RecordType::kTimeSlice;
  hdf5libs::HDF5WriterOptions options;
  options.file_layout_parameters = layout;
  return std::make_unique<rawdatafilelibs::RawDataFileWriter>(
    std::make_unique<hdf5libs::HDF5WriterBackend>(options), path, metadata, geo_id_map);
}

} // namespace dunedaq::trgtools
