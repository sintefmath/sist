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

#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>

namespace sist {
/** Implementation of the 5:1 HistoPyramid stream compaction and expansion.
 *
 * Implementation of the 5:1 HistoPyramid stream compaction and expansion
 * algorithm presented in
 *
 * C. Dyken, G. Ziegler, "GPU-Accelerated Data Expansion for the Marching
 * Cubes Algorithm", GPU Technology Conference 2010,
 *
 */

namespace hp5 {

/** 5-to-1 HistoPyramid Data Store and API entry points.
  *
  * The data store of the HP contains the actual HP (partial sums) as well as
  * some lookup tables. The buffers that contain input and output data is not
  * managed by the HP, but must be managed by the host application and pointers
  * are passed through the Predicate and Emitter
  *
  *
  */
class HP5
{
public:

    /** Allocate a HistoPyramid that can process N input elements. */
    HP5( unsigned int N );


    /** Analyze input data and build the HistoPyramid.
      *
      * The predicate is a functor object that takes an input stream index and
      * returns the number of times this element should be copied into the
      * output stream.
      *
      * The predicate object can also contain arbitrary data that the predicate
      * may need, e.g. pointers to buffers etc.
      *
      * An example predicate:
      * \code
      * struct TestPredicate
      * {
      *     __device__ __inline__
      *    unsigned int
      *    operator()( const unsigned int index ) const
      *    {
      *        return m_input_d[ index ] < m_threshold ? 1 : 0;
      *    }
      *    float*  m_input_d;
      *    float   m_threshold;
      * };
      * \endcode
      * This predicate checks if an input stream element is smaller than a given
      * threshold, and if so, it should be copied into the output stream.
      *
      * \tparam Predicate Predicate functor class.
      * \param predicate  The predicate that is passed to the kernels.
      * \param profile    Enable detailed profiling, slows down execution.
      *
      */
    template<class Predicate>
    void
    build( const Predicate  predicate,
           bool             profile = false );

    /** Fetch the number of output elements.
      *
      * During the HistoPyramid buildup, the number of output elements is
      * calculated.
      *
      * \returns The number of output stream elements.
      * \note Requires that build() has been invoked, or there is no result to
      *       return.
      */
    unsigned int
    outputSize();

    /** Populate the output stream.
      *
      * The HistoPyramid is used to find the corresponding input stream element
      * for every output stream element.
      *
      * The emitter is a functor object that takes the output stream index, the
      * clone index (only relevant for expansion), and the input stream index
      * and populates the output stream.
      *
      * The emitter object can also contain arbitrary data that the emitter may
      * need, e.g. pointers to buffers etc.
      *
      * An example predicate:
      * \code
      * struct TestEmitter
      * {
      *     __device__ __inline__
      *     void
      *     operator()( const unsigned int output_index,
      *                 const unsigned int input_index,
      *                 const unsigned int clone_index )
      *     {
      *         m_output_d[ output_index ] = m_input_d[ input_index ];
      *     }
      *     float*  m_input_d;
      *     float*  m_output_d;
      * };
      * \endcode
      * This emitter copies the selected input buffer elements to the output
      * buffer.
      *
      * The arguments to the emitter are:
      * - output_index: The index in the output stream of the element this
      *   invocation should populate. Adjacent threads have adjacent output
      *   indices.
      * - output_clone: Determines which clone this invocation corresponds to
      *   when using stream expansion. E.g., if the predicate returned 3, three
      *   adjacent invocations of the emitter will be invoked with identical
      *   input indices, but with clone indices 0, 1, and 2.
      *
      * \note Requires that outputSize() has been invoked since the last
      *       invocation of build(), as the number of elements is required to
      *       determine the number of blocks and threads to use.
      */
    template<class Emitter>
    void
    compact( Emitter  emitter,
             bool     use_texfetch,
             bool     use_constmem,
             bool     profile = false );

protected:
    enum State {
        STATE_ERROR,
        STATE_INITIALIZED,
        STATE_PYRAMID_BUILT,
        STATE_OUTPUT_SIZE_READ
    };
    State                       m_state;

    /** The number of elements in the input stream. */
    const unsigned int                  m_N;

    /** The number of elements in the output stream (after hp buildup). */
    unsigned int                        m_top;
    /** The number of reduction levels required. */
    unsigned int                        m_levels;
    unsigned int                        m_first_single_level;
    unsigned int                        m_first_double_level;
    unsigned int                        m_first_triple_level;
    /** The number of elements in each level. */
    std::vector<unsigned int>           m_level_size;
    /** Number of elements in the HP buffer. */
    unsigned int                        m_size;
    /** Position of each level in the HP buffer. */
    std::vector<unsigned int>           m_offsets;
    /** Device memory buffer to hold histopyramid. */
    uint4*                      m_histopyramid_dptr;
    /** Device memory buffer to hold sidebands. */
    unsigned int*                       m_sideband_dptr;

    void
    fillWithGarbage();


};

/** Internal functions */
namespace HP5Internal {
// TODO: Figure out how we can squeeze this into the class

/** Standard single level base population.
  *
  * \tparam Predicate  The predicate functor class, see HP5::build().
  * \param d_hp_0      Histopyramid level i+0, padded to 160 uint4's.
  * \param d_sb_0      Sideband level i+0, padded to 160 unsigned int's.
  * \param predicate   The predicuate functor object.
  * \param N           Number of elements in input stream.
  */
template<class Predicate>
void
__global__
__launch_bounds__( 160 )
buildup_base_single( uint4* __restrict__  d_hp_0,
                     unsigned int*  __restrict__  d_sb_0,
                     const Predicate      predicate,
                     const unsigned int           N )
{
    __shared__ unsigned int sb[160];
    const unsigned int gid5 = 160*blockIdx.x + threadIdx.x;
    unsigned int v = 0;
    if( gid5 < N ) {
        v = predicate(gid5);
    }
    sb[ threadIdx.x ] = v;
    __syncthreads();
    if( threadIdx.x < 32 ) {
        uint4 sums = make_uint4( sb[ 5*threadIdx.x + 0 ],
                                 sb[ 5*threadIdx.x + 1 ],
                                 sb[ 5*threadIdx.x + 2 ],
                                 sb[ 5*threadIdx.x + 3 ] );
        unsigned int sum = sums.x + sums.y + sums.z + sums.w + sb[ 5*threadIdx.x + 4 ];
        d_hp_0[ 32*blockIdx.x + threadIdx.x ] = sums;
        d_sb_0[ 32*blockIdx.x + threadIdx.x ] = sum;
    }
}

/** Standard single level reduction.
  *
  * \param d_hp_0  Histopyramid level i+0, padded to 160 uint4's.
  * \param d_sb_0  Sideband level i+0, padded to 160 unsigned int's.
  * \param d_sb_1  Sideband level i+1.
  * \param N_1     Quintuples in level i+1.
  */
void
__global__
__launch_bounds__( 160 )
buildup_level_single( uint4* __restrict__  d_hp_0,
                      unsigned int*  __restrict__  d_sb_0,
                      unsigned int*  __restrict__  d_sb_1,
                      const unsigned int           N_l )
{
    __shared__ unsigned int sb[160];
    const unsigned int gid5 = 160*blockIdx.x + threadIdx.x;
    if( gid5 < N_l ) {
        sb[threadIdx.x] = d_sb_1[ gid5 ];
    }
    else {
        sb[threadIdx.x] = 0;
    }
    __syncthreads();
    if( threadIdx.x < 32 ) {
        uint4 sums = make_uint4( sb[ 5*threadIdx.x + 0 ],
                                 sb[ 5*threadIdx.x + 1 ],
                                 sb[ 5*threadIdx.x + 2 ],
                                 sb[ 5*threadIdx.x + 3 ] );
        unsigned int sum = sums.x + sums.y + sums.z + sums.w + sb[ 5*threadIdx.x + 4 ];
        d_hp_0[ 32*blockIdx.x + threadIdx.x ] = sums;
        d_sb_0[ 32*blockIdx.x + threadIdx.x ] = sum;
    }
}

/** Double level base population.
  *
  * \param d_hp_0     Histopyramid level i+0, padded to 160 uint4's.
  * \param d_sb_0     Sideband level i+0, padded to 160 unsigned int's.
  * \param d_hp_1     Histopyramid level i+1, padded to 160 uint4's.
  * \param N          Number of elements in input stream.
  */
template<class Predicate>
__global__
void
__launch_bounds__( 160 )
buildup_base_double( uint4* __restrict__  d_hp_0,
                     unsigned int*  __restrict__  d_sb_0,
                     uint4* __restrict__  d_hp_1,
                     const Predicate      predicate,
                     const unsigned int           N )
{
    __shared__ unsigned int sb[160];
    __shared__ unsigned int sh[800];
    const unsigned int w  = threadIdx.x / 32;
    const unsigned int w5 = 5*w;
    const unsigned int wt = threadIdx.x % 32;
    const unsigned int wt5 = 5*wt;
    const unsigned int b32 = 32*blockIdx.x;
    // We let each warp process its own 5:1 reduction of a 160-element chunk.
    // Thus, there is no need to synchronize. Also, the shared mem accesses
    // are interlaced such that adjacent threads access adjacent elements,
    // eliminating bank conflicts.
    for(unsigned int p=0; p<5; p++) {
        const unsigned int gid5 = 5*5*b32 + (w5+p)*32 + wt;
        unsigned int v = 0;
        if( gid5 < N ) {
            v = predicate( gid5 );
        }
        sh[ 32*(w5+p) + wt ] = v;
    }
    const unsigned int sho = w5*32 + wt5;
    uint4 bl = make_uint4( sh[ sho+0 ], sh[ sho+1 ], sh[ sho+2 ], sh[ sho+3 ]);
    unsigned int b_o = 5*b32 + threadIdx.x;
    if( b_o < (N+3)/4 ) { // avoid excessive padding
        d_hp_1[ b_o ] = bl;
    }
    sb[ threadIdx.x ] = bl.x + bl.y + bl.z + bl.w + sh[ w5*32 + wt5 + 4 ];
    __syncthreads();
    if( w == 0 ) {
        uint4 bu = make_uint4( sb[ wt5+0 ], sb[ wt5+1 ], sb[ wt5+2 ], sb[ wt5+3 ] );
        d_hp_0[ b32 + wt ] = bu;
        d_sb_0[ b32 + wt ] = bu.x + bu.y + bu.z + bu.w + sb[ wt5 + 4 ];
    }
}

__global__
void
__launch_bounds__( 160 )
buildup_level_double( uint4* __restrict__  d_hp_0,
                      unsigned int*  __restrict__  d_sb_0,
                      uint4* __restrict__  d_hp_1,
                      unsigned int*  __restrict__  d_sb_2,
                      const unsigned int           N )
{
    __shared__ unsigned int sb[160];
    __shared__ unsigned int sh[800];
    const unsigned int w  = threadIdx.x / 32;
    const unsigned int w5 = 5*w;
    const unsigned int wt = threadIdx.x % 32;
    const unsigned int wt5 = 5*wt;
    const unsigned int b32 = 32*blockIdx.x;
    // Step 1: Populate hp_b and sb. Each warp calculates a 5:1 160-element
    // chunk, so there is no need for synchronization.
    for(unsigned int p=0; p<5; p++) {
        const unsigned int gid5 = 5*5*b32 + (w5+p)*32 + wt;
        unsigned int v = gid5<N ? d_sb_2[ gid5 ] : 0;
        sh[ 32*(w5+p) + wt ] = v;
    }
    // write the first four elements of the sideband to hp
    uint4 bl = make_uint4( sh[ w5*32 + wt5 + 0 ],
                           sh[ w5*32 + wt5 + 1 ],
                           sh[ w5*32 + wt5 + 2 ],
                           sh[ w5*32 + wt5 + 3 ]);
    if( w5+31 < N ) {
        d_hp_1[ 32*5*blockIdx.x + w*32 + wt ] = bl;
    }
    sb[ 32*w+wt ] = bl.x + bl.y + bl.z + bl.w + sh[ w5*32 + wt5 + 4 ];
    // Step 2: Last reduction. Only a single warp is used, but we have to wait
    // until all the other threads have finished populating sb.
    __syncthreads();
    if( w == 0 ) {
        uint4 sums = make_uint4( sb[ wt5+0 ], sb[ wt5+1 ], sb[ wt5+2 ], sb[ wt5+3 ] );
        unsigned int sum = sums.x + sums.y + sums.z + sums.w + sb[ wt5 + 4 ];
        d_hp_0[ b32 + wt ] = sums;
        d_sb_0[ b32 + wt ] = sum;
    }
}

/** Super-wide base level builder
  *
  * Fetches 5*5*160=4000 elements from the input stream, evaluates predicate,
  * outputs:
  *   3200 elements into hp_a (stores 5*5*32=800 in sh),
  *    640 elements into hp_b (stores 5*32=160 in sh),
  *    128 elements into hp_c,
  *     32 elements into sb_c.
  */
template<class Predicate>
__global__
void
__launch_bounds__( 160 )
buildup_base_triple( uint4* __restrict__  d_hp_c,
                     unsigned int*  __restrict__  d_sb_c,
                     uint4* __restrict__  d_hp_b,
                     uint4* __restrict__  d_hp_a,
                     const Predicate      predicate,
                     const unsigned int           N )
{
    __shared__ unsigned int sb[800];
    __shared__ unsigned int sh[800];
    const unsigned int w  = threadIdx.x / 32;
    const unsigned int wt = threadIdx.x % 32;
    const unsigned int b32 = 32*blockIdx.x;
    const unsigned int sh_i = 160*w + 5*wt;
    const unsigned int hp_b_o = 5*32*blockIdx.x + 32*w + wt;
    // There are 5 warps, each with 32 threads.
    // Each warp processes 5 input stream element chunks of size 160
    // sequentially. Each iteration produces 128 base level elements (outputted)
    // and 32 sideband elements. The sideband elements are stored interleaved
    // with the other warps, such that the shared mem consists of 160*5=800
    // sideband elements.
    for(unsigned int q=0; q<5; q++) {
        // Populate hp_b and sb. Each warp calculates a 5:1 160-element chunk,
        // so there is no need for synchronization.
        for(unsigned int p=0; p<5; p++) {
            const unsigned int gid5 = 5*5*5*32*blockIdx.x + 5*5*32*w + 32*5*q + 32*p + wt;
            unsigned int v = 0;
            if( gid5 < N ) {
                v = predicate(gid5);
            }
            sh[ 5*32*w + 32*p + wt ] = v;
        }
        // write the first four elements of the sideband to hp
        const unsigned int sb_b_o = 160*w + 32*q + wt;
        const unsigned int hp_a_o = 5*5*b32 + sb_b_o;
        uint4 bl = make_uint4( sh[ sh_i+0 ], sh[ sh_i+1 ], sh[ sh_i+2 ], sh[ sh_i+3 ]);
        if( hp_a_o < (N+3)/4 ) {
            d_hp_a[ hp_a_o ] = bl;
        }
        sb[ sb_b_o ] = bl.x + bl.y + bl.z + bl.w + sh[ sh_i + 4 ];
    }
    // Reduction
    uint4 bu = make_uint4( sb[ sh_i+0 ], sb[ sh_i+1 ], sb[ sh_i+2 ], sb[ sh_i+3 ] );
    d_hp_b[ hp_b_o ] = bu;
    __syncthreads(); // do I need to sync before writing here..? Yes. Deduce new pattern.
    sh[ 32*w + wt ] = bu.x + bu.y + bu.z + bu.w + sb[ sh_i + 4 ];
    // Reduction and write
    __syncthreads();
    if( w == 0 ) {
        uint4 bu = make_uint4( sh[ 5*wt+0 ], sh[ 5*wt+1 ], sh[ 5*wt+2 ], sh[ 5*wt+3 ] );
        d_hp_c[ 32*blockIdx.x + wt ] = bu;
        d_sb_c[ 32*blockIdx.x + wt ] = bu.x + bu.y + bu.z + bu.w + sh[ 5*wt + 4 ];
    }
}


/** Write the top 3 levels in one go.
  *
  * This replaces three tiny kernel invocations with one slightly larger
  * invocation. Also, it makes it easy to tightly pack the top levels.
  *
  * At level 3, there are maximally 5*5*5=125 non-zero sideband coefficients,
  * so 128 threads suffices.
  *
  * Also, just a single thread block runs.
  *
  * Lvl | tot |  sb |  hp | hp/4 | off4
  * ----+-----+-----+-----+------+-----
  * L0  |   5 |   1 |   4 |    1 |    1
  * L1  |  25 |   5 |  20 |    5 |    2
  * L2  | 125 |  25 | 100 |   25 |    7
  * L3  | 625 | 125 | 500 |  125 |   32
  */
__global__
void
__launch_bounds__( 128 )
buildup_apex( uint4* __restrict__  d_hp_012,
              unsigned int*  __restrict__  d_sb_3,
              const unsigned int           N_3 )
{
    __shared__ uint4 hp_012[32];
    __shared__ unsigned int sb[128];
    sb[ threadIdx.x ] = threadIdx.x < N_3 ? d_sb_3[ threadIdx.x ] : 0 ;
    // The computations are done solely by a single warp.
    __syncthreads();
    if( threadIdx.x < 25 ) {
        uint4 sums = make_uint4( sb[ 5*threadIdx.x + 0 ],
                                 sb[ 5*threadIdx.x + 1 ],
                                 sb[ 5*threadIdx.x + 2 ],
                                 sb[ 5*threadIdx.x + 3 ] );
        unsigned int sum = sums.x+sums.y+sums.z+sums.w+sb[5*threadIdx.x+4];
        hp_012[7+threadIdx.x] = sums;
        sb[ threadIdx.x ] = sum;
    }
    if( threadIdx.x < 5 ) {
        uint4 sums = make_uint4( sb[ 5*threadIdx.x + 0 ],
                                 sb[ 5*threadIdx.x + 1 ],
                                 sb[ 5*threadIdx.x + 2 ],
                                 sb[ 5*threadIdx.x + 3 ] );
        unsigned int sum = sums.x+sums.y+sums.z+sums.w+sb[5*threadIdx.x+4];
        hp_012[2+threadIdx.x] = sums;
        sb[ threadIdx.x ] = sum;
    }
    if( threadIdx.x < 1 ) {
        uint4 sums = make_uint4( sb[ 0 ], sb[ 1 ], sb[ 2 ], sb[ 3 ] );
        unsigned int sum = sums.x+sums.y+sums.z+sums.w+sb[ 4 ];
        hp_012[1] = sums;
        hp_012[0] = make_uint4( sum, 0, 0, 0 );
    }
    if( threadIdx.x < 32 ) {
        d_hp_012[ threadIdx.x ] = hp_012[ threadIdx.x ];
    }
}

// constant mem size: 64kb, cache working set: 8kb.
// Count + pad :  1+3 elements :    16 bytes :    16 bytes
// Level 0     :    4 elements :    16 bytes :    32 bytes
// Level 1     :   20 elements :    80 bytes :   112 bytes
// Level 2     :  100 elements :   400 bytes :   512 bytes
// Level 3     :  500 elements :  2000 bytes :  2112 bytes
// Level 4     : 2500 elements : 10000 bytes : 12112 bytes
// Levels 0-2: 32*4*4=512 bytes :
// Level  3:


/** Texture sampler used to fetch HistoPyramid elements. */
texture<uint4, 1, cudaReadModeElementType>  HP5_hp_tex;

/** Constant memory chunk that contains the apex of the HistoPyramid. */
static __constant__ uint4                   HP5_hp_const[528];      //=2112/4

/** Constant memory chunk that contains the offsets of the levels. */
static __constant__ unsigned int                    HP5_const_offsets[32];

/** Traverse a histopyramid.
  * \tparam Emitter The emitter functor class, see HP5::compact().
  * \tparam use_texfetch  Enables the use of textures during traversal. Speeds
  *                       up traversal, but create problems with streams.
  * \tparam use_constmem  Enable the use of storing the HP apex in constant
  *                       memory, also create problems with streams.
  * \param  emitter       The emitter functor object.
  * \param  d_hp          The buffer that contains the HistoPyramid.
  * \param  max_level     The number of levels in the HistoPyramid.
  */
template<class Emitter, bool use_texfetch,bool use_constmem>
__global__
void
traverse( Emitter              emitter,
          uint4* __restrict__  d_hp,
          const unsigned int           M,
          const unsigned int           max_level )
{
    const unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if( ix < M ) {
        unsigned int key = ix;
        unsigned int pos = 0;
        int l=0;
        if( use_constmem ) {
            for(l=0; l<4; l++ ) {
                uint4 val = HP5_hp_const[ HP5_const_offsets[l] + pos ];
                pos *= 5;
                if( val.x <= key ) {
                    pos++;
                    key -=val.x;
                    if( val.y <= key ) {
                        pos++;
                        key-=val.y;
                        if( val.z <= key ) {
                            pos++;
                            key-=val.z;
                            if( val.w <= key ) {
                                pos++;
                                key-=val.w;
                            }
                        }
                    }
                }
            }
        }

        for(; l<max_level; l++) {
            uint4 val;
            if(use_texfetch) {
                val = tex1Dfetch( HP5_hp_tex, HP5_const_offsets[l] + pos );
            }
            else {
                val = d_hp[ HP5_const_offsets[l] + pos ];
            }
            pos *= 5;

            if( val.x <= key ) {
                pos++;
                key -=val.x;
                if( val.y <= key ) {
                    pos++;
                    key-=val.y;
                    if( val.z <= key ) {
                        pos++;
                        key-=val.z;
                        if( val.w <= key ) {
                            pos++;
                            key-=val.w;
                        }
                    }
                }
            }
        }
        emitter( ix, pos, key );
    }
}


} // of namespace HP5Internal



HP5::HP5(unsigned int N)
    : m_N( N ),
      m_state( STATE_INITIALIZED )
{
    bool verbose = false;

    // Determine number of levels, we enforce that there is at least 4 levels
    m_levels = (unsigned int)ceilf( log2((float)N)/log2(5.0f));
    if( m_levels < 4 ) {
        m_levels = 4;
    }

    // Determine size of each level
    unsigned int n = m_N;
    m_level_size.resize( m_levels );
    for(unsigned int l=0; l<m_levels; l++) {
        m_level_size[ m_levels - 1 - l ] = n;
        n = (n+4)/5;
    }

    // Determine the jump pattern (when doing multiple reductions in one go)
    m_first_single_level = 3;
    if( m_first_single_level + 3u <= m_levels ) {
        m_first_triple_level = m_levels - 3u;
    }
    else {
        m_first_triple_level = m_levels; //-3
    }
    m_first_double_level = m_first_triple_level
                         - 2u*( (m_first_triple_level-m_first_single_level)/2 );


    m_offsets.resize( m_levels );
    m_offsets[0] = 1;
    m_offsets[1] = 2;
    m_offsets[2] = 7;
    m_size = 32;
    for(unsigned int l=m_first_single_level; l<m_first_double_level; l++ ) {
        m_offsets[l] = m_size;
        m_size += 5u*32u*( ( m_level_size[l]+159u)/160u );
    }
    for(unsigned int l=m_first_double_level; l<m_first_triple_level; l++ ) {
        m_offsets[l] = m_size;
        m_size += 5u*32u*(( m_level_size[l]+799u)/800u );
    }
    for(unsigned int l=m_first_triple_level; l<m_levels; l++ ) {
        m_offsets[l] = m_size;
        m_size += 5u*32u*(( m_level_size[l]+799u)/800u );
    }


    if( verbose ) {
        // Dump info to stderr
        std::cerr << "HP5: levels=" << m_levels << std::endl;
        std::cerr << "HP5: first single level=" << m_first_single_level << std::endl;
        std::cerr << "HP5: first double level=" << m_first_double_level << std::endl;
        std::cerr << "HP5: first triple level=" << m_first_triple_level << std::endl;

        for(unsigned int l=0; l<m_levels; l++ ) {
            std::cerr << "HP5: L" << l
                      << ": size=" << m_level_size[l]
                      << ", hp offset=" << (4*m_offsets[l])
                      << ", sb offset=" << m_offsets[l]
                      << std::endl;
        }
        std::cerr << "HP5: hp size=" << (4*m_size)
                  << ", sb size=" << m_size
                  << std::endl;
    }

    // Allocate memory
    cudaMalloc( reinterpret_cast<void**>(&m_histopyramid_dptr), 4*sizeof(unsigned int)*m_size );
    if( m_histopyramid_dptr == NULL ) {
        std::cerr << "HP5: Failed to allocate histopyramid buffer." << std::endl;
        m_state = STATE_ERROR;
        return;
    }



    cudaMalloc( reinterpret_cast<void**>(&m_sideband_dptr), sizeof(unsigned int)*m_size );
    if( m_sideband_dptr == NULL ) {
        std::cerr << "HP5: Failed to allocate sideband buffer." << std::endl;
        m_state = STATE_ERROR;
        return;
    }

    fillWithGarbage();


    cudaMemcpyToSymbol( HP5Internal::HP5_const_offsets,
                        m_offsets.data(),
                        m_offsets.size()*sizeof(unsigned int),
                        0,
                        cudaMemcpyHostToDevice );

    cudaError_t error = cudaGetLastError();
    if( error != cudaSuccess ) {
        std::cerr << "HP5: CUDA error in constructor: " << cudaGetErrorString( error ) << std::endl;
        m_state = STATE_ERROR;
        return;
    }
}

void
HP5::fillWithGarbage()
{
    cudaMemset( m_histopyramid_dptr, 0xCE, 4*sizeof(unsigned int)*m_size );
    cudaMemset( m_sideband_dptr, 0xCE, sizeof(unsigned int)*m_size );
    m_top = ~0;
}


template<class Predicate>
void
HP5::build( const Predicate predicate, bool profile )
{
    if( m_state == STATE_ERROR ) {
        std::cerr << "HP in error state." << std::endl;
        return;
    }

    std::vector<cudaEvent_t> events;

    bool build_base = true;
    // --- triple levels ---
    for(int i=m_levels; i>m_first_triple_level; i-=3 ) {
        int blocksize = (m_level_size[i-1]+3999)/4000;
        if( profile ) {
            std::cerr << "HP5: [TRIPLE] " << blocksize << " building " << (i-3) << ", " << (i-2) << ", " << (i-1) << (build_base?" base":"") << std::endl;
            events.resize( events.size()+1 );
            cudaEventCreate( &events.back() );
            cudaEventRecord( events.back() );
        }
        for(int it=0; it<(profile?100:1); it++) {
            if( build_base ) {
                HP5Internal::buildup_base_triple<Predicate><<<blocksize, 160>>>( m_histopyramid_dptr + m_offsets[ i-3 ],
                                                                                 m_sideband_dptr     + m_offsets[ i-3 ],
                                                                                 m_histopyramid_dptr + m_offsets[ i-2 ],
                                                                                 m_histopyramid_dptr + m_offsets[ i-1 ],
                                                                                 predicate,
                                                                                 m_N );
            }
            else {
                std::cerr << __func__ << ": Codepath not implemented, this should never happen." << std::endl;
                abort();
            }
        } // of profile loop
        build_base = false;
    }
    // --- double levels ---
    for(int i=m_first_triple_level; i>m_first_double_level; i-=2 ) {
        int blocksize = (m_level_size[i-1]+799)/800;
        if( profile ) {
            std::cerr << "HP5: [DOUBLE] " << blocksize << " building " << (i-2) << ", " << (i-1) << (build_base?" base":"") << std::endl;
            events.resize( events.size()+1 );
            cudaEventCreate( &events.back() );
            cudaEventRecord( events.back() );
        }
        for(int it=0; it<(profile?100:1); it++) {
            if( build_base ) {
                HP5Internal::buildup_base_double<Predicate><<<blocksize,160>>>( m_histopyramid_dptr + m_offsets[i-2],
                                                                                m_sideband_dptr + m_offsets[i-2],
                                                                                m_histopyramid_dptr + m_offsets[i-1],
                                                                                predicate,
                                                                                m_N );
            }
            else {
                HP5Internal::buildup_level_double<<<blocksize,160>>>( m_histopyramid_dptr + m_offsets[ i-2 ],
                                                                      m_sideband_dptr     + m_offsets[ i-2 ],
                                                                      m_histopyramid_dptr + m_offsets[ i-1 ],
                                                                      m_sideband_dptr     + m_offsets[ i ],
                                                                      m_level_size[ i-1 ] );
            }
        } // of profile loop
        build_base = false;
    }
    // --- single levels ---
    for(int i=m_first_double_level; i>m_first_single_level; --i ) {
        int blocksize = (m_level_size[i-1]+159)/160;
        if( profile ) {
            std::cerr << "HP5: [SINGLE] " << blocksize << " building " << (i-1) << (build_base?" base":"") << std::endl;
            events.resize( events.size()+1 );
            cudaEventCreate( &events.back() );
            cudaEventRecord( events.back() );
        }
        for(int it=0; it<(profile?100:1); it++) {
            if( build_base ) {
                HP5Internal::buildup_base_single<Predicate><<<blocksize,160>>>( m_histopyramid_dptr + m_offsets[ i-1 ],
                                                                                m_sideband_dptr     + m_offsets[ i-1 ],
                                                                                predicate,
                                                                                m_N );
            }
            else {
                HP5Internal::buildup_level_single<<<blocksize,160>>>( m_histopyramid_dptr + m_offsets[ i-1 ],
                                                                      m_sideband_dptr     + m_offsets[ i-1 ],
                                                                      m_sideband_dptr     + m_offsets[ i ],
                                                                      m_level_size[ i-1 ] );

            }
        } // of profile loop
        build_base = false;
    }
    // --- apex ---
    if( profile ) {
        std::cerr << "HP5: [APEX] " << 1 << " building " << 0 << ", " << 1 << ", " << 2  << std::endl;
        events.resize( events.size()+1 );
        cudaEventCreate( &events.back() );
        cudaEventRecord( events.back() );
    }
    for(int it=0; it<(profile?100:1); it++) {
        HP5Internal::buildup_apex<<<1,128>>>( m_histopyramid_dptr + 0,
                                              m_sideband_dptr     + 32,
                                              m_level_size[2] );
    }
    if( profile ) {
        events.resize( events.size()+1 );
        cudaEventCreate( &events.back() );
        cudaEventRecord( events.back() );
    }
    // Get result
    cudaMemcpy( &m_top, m_histopyramid_dptr, sizeof(unsigned int), cudaMemcpyDeviceToHost );
    if( profile ) {
        cudaEventSynchronize( events.back() );
        std::cerr << "HP5 Profile results for build pass:" << std::endl;
        float sum = 0.f;
        for(int i=0; i<events.size()-1; i++ ) {
            float time;
            cudaEventElapsedTime( &time, events[i], events[i+1] );
            time = time/100.0;
            sum += time;
            std::cerr << "    " << i << ": " << time << "ms." << std::endl;
        }
        std::cerr << "HP5: sum buildup " << sum << "ms" << std::endl;
        std::cerr << "HP5 output size=" << m_top << std::endl;
    }
    cudaError_t error = cudaGetLastError();
    if( error != cudaSuccess ) {
        std::cerr << "HP5: CUDA error in build: " << cudaGetErrorString( error ) << std::endl;
        abort();
    }
    m_state = STATE_PYRAMID_BUILT;
}

unsigned int
HP5::outputSize()
{
    if( m_state == STATE_ERROR ) {
        std::cerr << "HP in error state." << std::endl;
        return 0;
    }
    else if( m_state == STATE_INITIALIZED ) {
        std::cerr << "HP hasn't been built yet." << std::endl;
        return 0;
    }
    m_state = STATE_OUTPUT_SIZE_READ;
    return m_top;
}



template<class Emitter>
void
HP5::compact( Emitter  emitter,
              bool     use_texfetch,
              bool     use_constmem,
              bool     profile )
{
    if( m_state == STATE_ERROR ) {
        std::cerr << "HP in error state." << std::endl;
        return;
    }
    else if( m_state != STATE_OUTPUT_SIZE_READ ) {
        std::cerr << "Output size hasn't been read yet." << std::endl;
        return;
    }

    std::vector<cudaEvent_t> events;

    unsigned int bs = 256;
    unsigned int gs = (m_top+bs-1)/bs;
    unsigned int path = (use_texfetch ? 2 : 0 ) + (use_constmem ? 1 : 0 );

    if( profile ) {
        std::cerr << "HP5: compact: bs=" << bs << ", gs=" << gs << ", path=" << path << std::endl;
        events.resize( events.size()+1 );
        cudaEventCreate( &events.back() );
        cudaEventRecord( events.back() );
    }
    for( int it=0; it<(profile?100:1); it++ ) {
        if( use_constmem ) {
            cudaMemcpyToSymbol( HP5Internal::HP5_hp_const,
                                m_histopyramid_dptr,
                                528*sizeof(uint4),
                                0,
                                cudaMemcpyDeviceToDevice );
            cudaError error = cudaGetLastError();
            if( error != cudaSuccess ) {
                std::cerr << "HP5: CUDA error when copying to const mem: " << cudaGetErrorString(error) << std::endl;
                abort();
            }
        }
        if( use_texfetch ) {
            cudaBindTexture( NULL,
                             HP5Internal::HP5_hp_tex,
                             m_histopyramid_dptr,
                             cudaCreateChannelDesc( 32, 32, 32, 32, cudaChannelFormatKindUnsigned ),
                             4*sizeof(unsigned int)*m_size );
        }

        switch( path ) {
        case 0:
            HP5Internal::traverse<Emitter,false,false><<<gs,bs>>>( emitter,
                                                                   m_histopyramid_dptr,
                                                                   m_top,
                                                                   m_levels );
            break;
        case 1:
            HP5Internal::traverse<Emitter,false,true><<<gs,bs>>>( emitter,
                                                                  m_histopyramid_dptr,
                                                                  m_top,
                                                                  m_levels );
            break;
        case 2:
            HP5Internal::traverse<Emitter,true,false><<<gs,bs>>>( emitter,
                                                                  m_histopyramid_dptr,
                                                                  m_top,
                                                                  m_levels );
            break;
        case 3:
            HP5Internal::traverse<Emitter,true,true><<<gs,bs>>>( emitter,
                                                                 m_histopyramid_dptr,
                                                                 m_top,
                                                                 m_levels );
            break;

        }
    } // of profile loop
    if( profile ) {
        events.resize( events.size()+1 );
        cudaEventCreate( &events.back() );
        cudaEventRecord( events.back() );

        cudaEventSynchronize( events.back() );
        float time;
        cudaEventElapsedTime( &time,
                              events.front(),
                              events.back() );
        time = time/100.0;
        std::cerr << "HP5: compact used " << time << "ms" << std::endl;
    }
}


    } // of namespace hp5
} // of namespace sist
