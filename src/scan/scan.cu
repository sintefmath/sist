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
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <sist/scan/scan.hpp>

namespace sist {
namespace scan {

static const unsigned int warpsize_l2       = 5;
static const unsigned int warpsize          = 1<<warpsize_l2;
static const unsigned int blocksize         = 4*warpsize;
static const unsigned int chunksize_l2      = warpsize_l2 + 4;
static const unsigned int chunksize         = 1<<chunksize_l2;
static const unsigned int level_padding     = 512;

/** Calculate offsets and number of elements at different levels for large scan. */
static inline
void
offsets( std::vector<size_t>& N_in,
             std::vector<size_t>& off,
             const size_t		  N )
{
    off.clear();
    off.push_back( 0 );

    N_in.clear();
    N_in.push_back( N );

    while( chunksize < N_in.back() ) {
        N_in.push_back( ( N_in.back() + chunksize - 1 )/chunksize );
        size_t o = off.back()+N_in.back();
        off.push_back( level_padding*( ( o + level_padding - 1 )/level_padding) );
    }
}

__host__
size_t
scratchBufferBytesize( const unsigned int N )
{
    std::vector<size_t> N_in;
    std::vector<size_t> off;
    offsets( N_in, off, N );
    return sizeof(unsigned int)*(off.back());
}



/** Kernel that reduces 4*warpsize elements into a single sum. */
__global__ void
__launch_bounds__(4*warpsize)
reduce4x4( unsigned int*        output,
           const unsigned int*  input,
           const unsigned int   N,
           const unsigned int           B )
{
    const unsigned int t  = threadIdx.x;
    const unsigned int b  = blockIdx.y*gridDim.x + blockIdx.x;
    if( B <= b ) {
        return;
    }
    const unsigned int ix = (b<<chunksize_l2) + (t<<2);    //  index of input uint4

    // -- Let each thread compute the sum of 4 elements individually
    unsigned int thread_sum = 0;
    if( ix+3 < N ) {
        uint4 t = *(uint4*)(input + ix);
        thread_sum += t.x + t.y + t.z + t.w;
    }
    else if( ix < N ) {
        thread_sum += input[ix+0];
        if( ix+1 < N ) thread_sum += input[ix+1];
        if( ix+2 < N ) thread_sum += input[ix+2];
    }

    // -- Let the the three upper warps write their result
    __shared__ volatile unsigned int sh[ 3*warpsize ];
    if( warpsize <= t ) {
        sh[ t-warpsize ] = thread_sum;
    }
    __syncthreads();

    // -- Let the first warp reduce the 3*warpsize values
    if( t < warpsize ) {
        sh[t] = thread_sum + sh[t] + sh[t+warpsize] + sh[t+2*warpsize];
        for(unsigned int r=warpsize_l2-1; r>0; r--) {
            if( t < (1<<r) ) {
                sh[t] = sh[2*t+0] + sh[2*t+1];
            }
        }
        // -- Let thread 0 write the total
        if( t == 0 ) {
            unsigned int sum = sh[0] + sh[1];
            output[ b ] = sum;
        }
    }
}


template<bool inclusive, bool pull, bool push, bool write_sum>
__global__
__launch_bounds__(4*warpsize)
void
scan4x4( unsigned int*         output,
         unsigned int*         apex,
         unsigned int*         sum,
         const unsigned int*   input,
         const unsigned int    N,
         const unsigned int    B )
{
    const unsigned int blocksize    = 4*warpsize;       // number of threads in block
    const unsigned int t            = threadIdx.x;      // thread id
    const unsigned int warp         = t>>warpsize_l2;   // warp of thread
    const unsigned int wt           = t&(warpsize-1);   // thread id within warp
    const unsigned int b            = blockIdx.y*gridDim.x + blockIdx.x;
    if( B <= b ) {
        return;
    }
    const unsigned int ix           = (b<<chunksize_l2) + (t<<2);    //  index of input uint4

    __shared__ volatile unsigned int row[ blocksize ];

    // -- fetch four values and store sum in row[t]
    uint4 vals;
    if( ix+3 < N ) {
        vals = *(uint4*)(input + ix);
    }
    else if( ix + 0 < N ) {
        vals.x = input[ix+0];
        vals.y = ix+1 < N ? input[ix+1] : 0;
        vals.z = ix+2 < N ? input[ix+2] : 0;
        vals.w = 0;
    }
    else {
        vals = make_uint4( 0, 0, 0, 0 );
    }
    row[t] = vals.x+vals.y+vals.z+vals.w;

    // -- let each warp build an exclusive scan indivudually
    if( (warpsize_l2>0) && (wt>= 1) ) { row[t] += row[t-1]; }
    if( (warpsize_l2>1) && (wt>= 2) ) { row[t] += row[t-2]; }
    if( (warpsize_l2>2) && (wt>= 4) ) { row[t] += row[t-4]; }
    if( (warpsize_l2>3) && (wt>= 8) ) { row[t] += row[t-8]; }
    if( (warpsize_l2>4) && (wt>=16) ) { row[t] += row[t-16]; }

    // --

    __syncthreads();
    if( t==0 ) {
        unsigned int s = 0, next;
        if( pull ) {
            s = apex[ b ];
        }
        next = row[ 1*warpsize-1 ];
        row[ 1*warpsize-1 ] = s;
        s = s + next;
        next = row[ 2*warpsize-1 ];
        row[ 2*warpsize-1 ] = s;
        s = s + next;
        next = row[ 3*warpsize-1 ];
        row[ 3*warpsize-1 ] = s;
        s = s + next;
        if( push || write_sum ) {
            next = row[ 4*warpsize-1 ];
        }
        row[ 4*warpsize-1 ] = s;
        s = s + next;
        if( push ) {
            apex[ b ] = s;
        }
        if( write_sum ) {
            *sum = s;
        }

    }
    __syncthreads();

    // --- all warps fetch block offset, create 4-element scan and write
    uint4 res;
    if( inclusive ) {
        res.x = (wt==0?0:row[t-1]) + row[ (warp+1)*warpsize - 1 ] + vals.x;
        res.y = res.x + vals.y;
        res.z = res.y + vals.z;
        res.w = res.z + vals.w;
    }
    else {
        res.x = (wt==0?0:row[t-1]) + row[ (warp+1)*warpsize - 1 ];
        res.y = res.x + vals.x;
        res.z = res.y + vals.y;
        res.w = res.z + vals.z;
    }

    if( ix+3 < N ) {
        *(uint4*)(output + ix) = res;
    }
    else if( ix < N ) {
        output[ix+0] = res.x;
        if( ix+1 < N ) output[ix+1] = res.y;
        if( ix+2 < N ) output[ix+2] = res.z;
    }
}


template<bool inclusive, bool pad, bool write_sum>
static
void
run( unsigned int*         output_d,
     unsigned int*         scratch_d,
     unsigned int*         sum_d,
     const unsigned int*   input_d,
     const unsigned int    N,
     cudaStream_t  stream )
{
    if( N <= chunksize ) {
        scan4x4< inclusive, false, pad, write_sum >
                <<< 1, 4*warpsize, 0, stream >>>( output_d,
                                                  output_d + N,
                                                  sum_d,
                                                  input_d,
                                                  N,
                                                  1 );
    }
    else {
        std::vector<size_t> N_in;
        std::vector<size_t> off;
        offsets( N_in, off, N );
        unsigned int levels = N_in.size();

        std::vector<dim3> grids(levels-1);
        for(unsigned int i=0; i<levels-1; i++ ) {
            unsigned int n = N_in[i+1];
            if( n <= 512 ) {
                grids[i] = dim3( n );
            }
            else {
                unsigned int nn = unsigned int(std::ceil( std::sqrt( static_cast<double>( n ) ) ));
                grids[i] = dim3( nn, (n+nn-1)/nn );
            }
        }

        // --- base level reduction
        reduce4x4<<< grids[0], blocksize, 0, stream >>>
                 ( scratch_d + off[0],
                   input_d,
                   N_in[0],
                   N_in[1] );
        // --- intermediate level reductions
        for(unsigned int i=1; i<levels-1; i++ ) {
            reduce4x4<<< grids[i], blocksize, 0, stream >>>
                     ( scratch_d + off[i],
                       scratch_d + off[i-1],
                       N_in[i],
                       N_in[i+1] );
        }
        // --- top level scan, block offset is zero
        scan4x4< false, false, pad, write_sum >
               <<< 1, blocksize, 0, stream >>>
               ( scratch_d + off[levels-2],
                 output_d + N,
                 sum_d,
                 scratch_d + off[levels-2],
                 N_in[levels-1],
                 1 );
        // --- scan intermediate levels top-to-bottom, pulling block offset
        for(unsigned int i=levels-2; i>0; i--) {
            scan4x4<false, true, false, false >
                   <<< grids[i], blocksize, 0, stream >>>
                   ( scratch_d + off[i-1],
                     scratch_d + off[i],
                     NULL,
                     scratch_d + off[i-1],
                     N_in[i],
                     N_in[i+1] );
        }
        // --- scan bottom level, pulling block offset
        scan4x4< inclusive, true, false, false >
               <<< grids[0], 4*warpsize, 0, stream >>>
               ( output_d,
                 scratch_d + off[0],
                 NULL,
                 input_d,
                 N_in[0],
                 N_in[1]);
    }

}

void
inclusiveScan( unsigned int*         output_d,
               void*         scratch_d,
               const unsigned int*   input_d,
               const unsigned int    N,
               cudaStream_t  stream )
{
    run<true, false, false>( output_d,
                             static_cast<unsigned int*>( scratch_d ),
                             NULL,
                             input_d,
                             N,
                             stream );
}

void
inclusiveScanWriteSum( unsigned int*         output_d,
                       unsigned int*         sum_d,
                       void*         scratch_d,
                       const unsigned int*   input_d,
                       const unsigned int    N,
                       cudaStream_t  stream )
{
    run<true, false, true>( output_d,
                            static_cast<unsigned int*>( scratch_d ),
                            sum_d,
                            input_d,
                            N,
                            stream );
}

void
exclusiveScan( unsigned int*         output_d,
               void*         scratch_d,
               const unsigned int*   input_d,
               const unsigned int    N,
               cudaStream_t  stream )
{
    run<false, false, false>( output_d,
                              static_cast<unsigned int*>( scratch_d ),
                              NULL,
                              input_d,
                              N,
                              stream );
}

void
exclusiveScanWriteSum( unsigned int*         output_d,
                       unsigned int*         sum_d,
                       void*         scratch_d,
                       const unsigned int*   input_d,
                       const unsigned int    N,
                       cudaStream_t  stream )
{
    run<false, false, true>( output_d,
                             static_cast<unsigned int*>( scratch_d ),
                             sum_d,
                             input_d,
                             N,
                             stream );
}

void
exclusiveScanPadWithSum( unsigned int*         output_d,
                         void*         scratch_d,
                         const unsigned int*   input_d,
                         const unsigned int    N,
                         cudaStream_t  stream )
{
    run<false, true, false>( output_d,
                             static_cast<unsigned int*>( scratch_d ),
                             NULL,
                             input_d,
                             N,
                             stream );
}

void
exclusiveScanPadWithSumWriteSum( unsigned int*         output_d,
                                 unsigned int*         sum_d,
                                 void*         scratch_d,
                                 const unsigned int*   input_d,
                                 const unsigned int    N,
                                 cudaStream_t  stream )
{
    run<false, true, true>( output_d,
                            static_cast<unsigned int*>( scratch_d ),
                            sum_d,
                            input_d,
                            N,
                            stream );
}



} // of namespace scan
} // of namespace sist
