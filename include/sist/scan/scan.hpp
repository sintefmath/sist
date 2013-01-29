#pragma once
/* Copyright STIFTELSEN SINTEF 2010
 *
 * This file is part of the SIST Library.
 *
 * Author(s): Christopher Dyken, <christopher.dyken@sintef.no>
 *
 * SIST is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * HPMC is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * SIST.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <cuda_runtime.h>

namespace sist {

/** CUDA functions to calculate variations of scan.
 *
 * Types of scan
 * -------------
 *
 * Given an input array
 * \code
 * [a, b, c, ... y, z ] // N elements
 * \endcode
 * inclusive scan calculates
 * \code
 * [a, a+b, a+b+c, ... (a+b+c+..+y+z)] // N elements
 * \endcode
 * and exclusive scan calculates
 * \code
 * [0, a, a+b, a+b+c, ... (a+b+c+..+y)] // N elements
 * \endcode
 * In addition, an exclusive scan padded with sum calculates
 * \code
 * [0, a, a+b, a+b+c, ... (a+b+c+..+y), (a+b+c+..+y+z) ] // N+1 elements
 * \endcode
 * which is convenient for creating offset tables (input[i] is number of elements
 * allocated for position i, and output[i] is beginning of output range and
 * output[i+1] is the end of output range for pos i).
 *
 * Retrieving the total sum
 * ------------------------
 *
 * In the scan process, the total sum of all input elements is calculated, which
 * often is a useful number. The scan functions can optionally write the this
 * total to a device memory address. If used in conjuction with device mapped
 * host memory, the sum is zero-copied directly to the host:
 * \code
 * unsigned int* zerocopy, zerocopy_d;
 * cudaHostAlloc( &zerocopy, sizeof(unsigned int), cudaHostAllocMapped );
 * cudaHostGetDevicePointer( &zerocopy_d, zerocopy, 0 );
 * Scan::inclusiveScanWriteSum( output_d,
 *                              zerocopy_d,
 *                              scratch_d,
 *                              input_d,
 *                              N );
 * std::cerr << "sum=" << *zerocopy << std::endl;
 * cudaFreeHost( zerocopy );
 * \endcode
 *
 * Scan of large arrays
 * --------------------
 *
 * Scan of arrays larger than blocksize require several passes, and a scratch
 * buffer into where to store intermediate values between buffers. The size of
 * this buffer for a given number of elements is provided by scratchBufferSize.
 * If this function returns zero, the scan functions doesn't need any
 * intermediate storage, and it is safe to pass NULL.
 *
 * In-place scans
 * --------------
 *
 * All scan functions support reading and writing to the same memory location
 * (i.e., output_d == input_d ) without penalty.
 *
 * Streams
 * -------
 *
 * All API entries that launches kernels have a stream parameter, and all CUDA
 * operations are pushed onto that stream.
 *
 * Limits
 * ------
 *
 * The functions have been tested for input arrays of size up to 0x10000000
 * (cudpp has a limit of 0x03fffc00).
 *
 */
namespace scan {

/** Get an upper bound for the scratch buffer needed by the scan algorithms.
 *
 * \param[in] N   The type and maximum number of elements to process.
 * \returns       The minimum scratchbuffer bytesize, or zero if no scratch
 *                buffer is needed.
 * \author Christopher Dyken, <christopher.dyken@sintef.no>
 */
size_t
scratchBufferBytesize( const unsigned int N );

/** Perform an inclusive scan.
 *
 * \param[out] output_d   Device pointer where to the output (N elements).
 * \param[in]  scratch_d  Device pointer to scratch area (NULL for small N).
 * \param[in]  input_d    Device pointer where fetch input (N elements).
 * \param[in]  N          The number of elements to process.
 * \param[in]  stream     Which CUDA stream to use.
 * \author Christopher Dyken, <christopher.dyken@sintef.no>
 */
void
inclusiveScan( unsigned int*         output_d,
               void*         scratch_d,
               const unsigned int*   input_d,
               const unsigned int    N,
               cudaStream_t  stream = 0);

/** Perform an inclusive scan and write total sum to a location.
 *
 * \param[out] output_d   Device pointer where to the output (N elements).
 * \param[out] sum_d      Device pointer where to the total sum (1 element).
 * \param[in]  scratch_d  Device pointer to scratch area (NULL for small N).
 * \param[in]  input_d    Device pointer where fetch input (N elements).
 * \param[in]  N          The number of elements to process.
 * \param[in]  stream     Which CUDA stream to use.
 * \author Christopher Dyken, <christopher.dyken@sintef.no>
 */
void
inclusiveScanWriteSum( unsigned int*         output_d,
                       unsigned int*         sum_d,
                       void*         scratch_d,
                       const unsigned int*   input_d,
                       const unsigned int    N,
                       cudaStream_t  stream = 0 );


/** Perform an exclusive scan.
 *
 * \param[out] output_d   Device pointer where to the output (N elements).
 * \param[in]  scratch_d  Device pointer to scratch area (NULL for small N).
 * \param[in]  input_d    Device pointer where fetch input (N elements).
 * \param[in]  N          The number of elements to process.
 * \param[in]  stream     Which CUDA stream to use.
 * \author Christopher Dyken, <christopher.dyken@sintef.no>
 */
void
exclusiveScan( unsigned int*         output_d,
               void*         scratch_d,
               const unsigned int*   input_d,
               const unsigned int    N,
               cudaStream_t  stream = 0);

/** Perform an exclusive scan and write total sum to a location.
 *
 * \param[out] output_d   Device pointer where to the output (N elements).
 * \param[out] sum_d      Device pointer where to the total sum (1 element).
 * \param[in]  scratch_d  Device pointer to scratch area (NULL for small N).
 * \param[in]  input_d    Device pointer where fetch input (N elements).
 * \param[in]  N          The number of elements to process.
 * \param[in]  stream     Which CUDA stream to use.
 * \author Christopher Dyken, <christopher.dyken@sintef.no>
 */
void
exclusiveScanWriteSum( unsigned int*         output_d,
                       unsigned int*         sum_d,
                       void*         scratch_d,
                       const unsigned int*   input_d,
                       const unsigned int    N,
                       cudaStream_t  stream = 0);



/** Perform an exclusive scan, but append the total sum at end.
 *
 * \param[out] output_d   Device memory pointer to where to store N+1 ouput elements.
 * \param[in]  scratch_d  Device memory pointer to scratch area (or NULL if N
 *                        is so small that the scan can be done in a single
 *                        pass), see scratchBufferBytesize.
 * \param[in]  input_d    Device memory pointer to where fetch N input elements.
 * \param[in]  N          The number of elements to process.
 * \param[in]  stream     Which CUDA stream to use.
 *
 * Exclusive scan produces the following result:
 * \code
 * unsigned int sum = 0;
 * for(unsigned int i=0; i<N; i++) {
 *     output[i] = sum;
 *     sum       = sum + input[i];
 * }
 * output[N] = sum;
 * \endcode
 *
 * \note Inplace scan (input_d and output_d points points to the same memory) is
 * supported without any penalties.
 * \author Christopher Dyken, <christopher.dyken@sintef.no>
 */
void
exclusiveScanPadWithSum( unsigned int*         output_d,
                         void*         scratch_d,
                         const unsigned int*   input_d,
                         const unsigned int    N,
                         cudaStream_t  stream = 0);

void
exclusiveScanPadWithSumWriteSum( unsigned int*         output_d,
                                 unsigned int*         sum_d,
                                 void*         scratch_d,
                                 const unsigned int*   input_d,
                                 const unsigned int    N,
                                 cudaStream_t  stream = 0);




} // of namespace scan
} // of namespace sist
