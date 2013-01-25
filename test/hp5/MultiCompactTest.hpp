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

#include <sist/hp5/hp5_inline.hpp>
#include <vector>

struct TestPredicate
{
    __device__ __inline__
    uint
    operator()( const uint index ) const
    {
        return m_input_dptr[index] == 1 ? 1 : 0;
    }

    short*  m_input_dptr;
};

struct TestEmitter
{
    __device__ __inline__
    void
    operator()( const uint output_index, const uint input_index, const uint output_clone )
    {
        m_out_image1_d[ output_index ] = m_in_image1_d[ input_index ];
        m_out_image2_d[ output_index ] = m_in_image2_d[ input_index ];
        m_out_image3_d[ output_index ] = m_in_image3_d[ input_index ];
    }
    short* m_in_image1_d;
    short* m_in_image2_d;
    short* m_in_image3_d;
    short* m_out_image1_d;
    short* m_out_image2_d;
    short* m_out_image3_d;
};


struct Test
{
    Test( int input_size,
          int seed,
          bool use_texfetch,
          bool use_constmem,
          float threshold )
        : m_hp( input_size ),
          m_input_size( input_size ),
          m_seed( seed ),
          m_use_texfetch( use_texfetch ),
          m_use_constmem( use_constmem ),
          m_threshold( threshold )
    {
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );



        // Create input
        srand( seed );
        m_input0.resize( input_size );
        std::vector<short> in_classifier( input_size );
        for( size_t i=0; i<input_size; i++ ) {
            m_input0[i] = (rand()/(float)RAND_MAX) < threshold ? 1 : 0;
        }
        cudaMalloc( &m_input0_d, sizeof(short)*input_size );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaMemcpy( m_input0_d, m_input0.data(), sizeof(short)*m_input_size, cudaMemcpyHostToDevice );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );

        m_input1.resize( input_size );
        for( size_t i=0; i<input_size; i++ ) {
            m_input1[i] = rand();
        }
        cudaMalloc( &m_input1_d, sizeof(short)*input_size );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaMemcpy( m_input1_d, m_input1.data(), sizeof(short)*m_input_size, cudaMemcpyHostToDevice );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );

        m_input2.resize( input_size );
        for( size_t i=0; i<input_size; i++ ) {
            m_input2[i] = rand();
        }
        cudaMalloc( &m_input2_d, sizeof(short)*input_size );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaMemcpy( m_input2_d, m_input2.data(), sizeof(short)*m_input_size, cudaMemcpyHostToDevice );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );

        m_input3.resize( input_size );
        for( size_t i=0; i<input_size; i++ ) {
            m_input3[i] = rand();
        }
        cudaMalloc( &m_input3_d, sizeof(short)*input_size );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaMemcpy( m_input3_d, m_input3.data(), sizeof(short)*m_input_size, cudaMemcpyHostToDevice );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );


        // Create gold standard:
        for( int i=0; i<input_size; i++ ) {
            if( m_input0[i] == 1 ) {
                m_gold1.push_back( m_input1[i] );
                m_gold2.push_back( m_input2[i] );
                m_gold3.push_back( m_input3[i] );
            }
        }


        cudaMalloc( &m_output1_d, sizeof(short)*m_gold1.size() );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaMemset( m_output1_d, 17, m_gold1.size() );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );

        cudaMalloc( &m_output2_d, sizeof(short)*m_gold1.size() );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaMemset( m_output2_d, 17, m_gold1.size() );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );

        cudaMalloc( &m_output3_d, sizeof(short)*m_gold1.size() );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaMemset( m_output3_d, 17, m_gold1.size() );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );

    }

    ~Test()
    {
        cudaFree( m_input0_d );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaFree( m_input1_d );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaFree( m_input2_d );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaFree( m_input3_d );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaFree( m_output1_d );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaFree( m_output2_d );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaFree( m_output3_d );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
    }

    void
    buildupPass()
    {
        TestPredicate predicate;

        predicate.m_input_dptr = m_input0_d;
        m_hp.build( predicate, false );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
    }

    int
    getOutputCount()
    {
        m_count = m_hp.outputSize();
        BOOST_CHECK_EQUAL( (int)m_gold1.size(), m_count );
        return m_count;
    }

    void
    extractPass()
    {
        TestEmitter emitter;
        emitter.m_in_image1_d = m_input1_d;
        emitter.m_in_image2_d = m_input2_d;
        emitter.m_in_image3_d = m_input3_d;
        emitter.m_out_image1_d = m_output1_d;
        emitter.m_out_image2_d = m_output2_d;
        emitter.m_out_image3_d = m_output3_d;
        m_hp.compact( emitter,
                      m_use_texfetch,
                      m_use_constmem,
                      false );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
    }

    void
    verifyResults()
    {
        std::vector<short> output1( m_gold1.size() );
        std::vector<short> output2( m_gold1.size() );
        std::vector<short> output3( m_gold1.size() );

        cudaMemcpy( output1.data(), m_output1_d, sizeof(short)*m_gold1.size(), cudaMemcpyDeviceToHost );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaMemcpy( output2.data(), m_output2_d, sizeof(short)*m_gold2.size(), cudaMemcpyDeviceToHost );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        cudaMemcpy( output3.data(), m_output3_d, sizeof(short)*m_gold3.size(), cudaMemcpyDeviceToHost );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );


        for( size_t i=0; i<m_gold1.size(); i++ ) {
            if( output1[i] != m_gold1[i] ) {
                BOOST_CHECK_EQUAL( output1[i], m_gold1[i] );
                break;
            }
            if( output2[i] != m_gold2[i] ) {
                BOOST_CHECK_EQUAL( output2[i], m_gold2[i] );
                break;
            }
            if( output3[i] != m_gold3[i] ) {
                BOOST_CHECK_EQUAL( output3[i], m_gold3[i] );
                break;
            }
        }
    }


    void
    run()
    {
        buildupPass();
        getOutputCount();
        extractPass();
        verifyResults();
    }


    sist::hp5::HP5      m_hp;
    int                 m_count;

    short*              m_input0_d;
    short*              m_input1_d;
    short*              m_input2_d;
    short*              m_input3_d;
    short*              m_output1_d;
    short*              m_output2_d;
    short*              m_output3_d;


    std::vector<short>  m_input0;
    std::vector<short>  m_input1;
    std::vector<short>  m_input2;
    std::vector<short>  m_input3;

    std::vector<short>  m_gold1;
    std::vector<short>  m_gold2;
    std::vector<short>  m_gold3;

    std::vector<short>  m_result1;
    std::vector<short>  m_result2;
    std::vector<short>  m_result3;

    int                 m_input_size;
    int                 m_seed;
    bool                m_use_texfetch;
    bool                m_use_constmem;
    float               m_threshold;
};
