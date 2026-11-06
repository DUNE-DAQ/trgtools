/**
 * @file OutputFile.hpp Creating the HDF5 TimeSlice files that trgtools applications write
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2020.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#ifndef TRGTOOLS_INCLUDE_TRGTOOLS_OUTPUTFILE_HPP_
#define TRGTOOLS_INCLUDE_TRGTOOLS_OUTPUTFILE_HPP_

#include "daqdataformats/SourceIDGeoIDMap.hpp"
#include "hdf5libs/HDF5FileLayoutParameters.hpp"
#include "rawdatafilelibs/FileMetadata.hpp"
#include "rawdatafilelibs/RawDataFileReader.hpp"
#include "rawdatafilelibs/RawDataFileWriter.hpp"

#include <memory>
#include <optional>
#include <string>

namespace dunedaq::trgtools {

/**
 * @brief Raw data files are never overwritten, so a tool that may replace its output removes the
 * old file itself:
 *  - if "<path>.writing" exists, always throws std::runtime_error and changes nothing: that file
 *    may be recoverable (rawdatafile_recover), or still being written by another process;
 *  - else, if path exists, throws unless overwrite is set, in which case path is removed.
 */
void
prepare_output_path(const std::string& path, bool overwrite);

/**
 * @brief The checks of prepare_output_path() without removing anything, e.g. to fail early before
 * long processing; call prepare_output_path() right before writing.
 */
void
check_output_path(const std::string& path, bool overwrite);

/**
 * @brief Creates an HDF5 file of TimeSlices with the given metadata and GeoID map, and the given
 * HDF5 file layout (e.g. the input file's) or else the standard TimeSlice layout.
 */
std::unique_ptr<rawdatafilelibs::RawDataFileWriter>
create_hdf5_timeslice_file(const std::string& path,
                           rawdatafilelibs::FileMetadata metadata,
                           const daqdataformats::SourceIDGeoIDMap& geo_id_map,
                           const std::optional<hdf5libs::HDF5FileLayoutParameters>& layout = std::nullopt);

/**
 * @brief The HDF5 file layout of a file read through the HDF5 backend, e.g. to write an output
 * file with its input's layout; none for a file read through another backend.
 */
std::optional<hdf5libs::HDF5FileLayoutParameters>
hdf5_file_layout(const rawdatafilelibs::RawDataFileReader& reader);

} // namespace dunedaq::trgtools

#endif // TRGTOOLS_INCLUDE_TRGTOOLS_OUTPUTFILE_HPP_
