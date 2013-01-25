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
        return m_input_dptr[index] < m_threshold ? 1 : 0;
    }

    float*  m_input_dptr;
    float   m_threshold;
};

struct TestEmitter
{
    __device__ __inline__
    void
    operator()( const uint output_index, const uint input_index, const uint output_clone )
    {
        m_output_dptr[ output_index ] = m_input_dptr[ input_index ];
    }
    float*  m_input_dptr;
    float*  m_output_dptr;
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



        // create input data
        srand( m_seed );
        m_input.resize( m_input_size );
        for( size_t i=0; i<m_input.size(); i++ ) {
            m_input[ i ] = (rand()/(float)RAND_MAX);
        }

        cudaMalloc( &m_input_d, sizeof(float)*input_size );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        BOOST_CHECK_NE( m_input_d, (float*)NULL );

        cudaMemcpy( m_input_d,
                    m_input.data(),
                    sizeof(float)*m_input.size(),
                    cudaMemcpyHostToDevice );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );

        // create gold output
        m_gold.clear();
        for( size_t i=0; i<m_input.size(); i++ ) {
            if( m_input[i] < m_threshold ) {
                m_gold.push_back( m_input[i] );
            }
        }

        cudaMalloc( &m_output_d, sizeof(float)*m_gold.size() );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
        BOOST_CHECK_NE( m_output_d, (float*)NULL );

        cudaMemset( m_output_d, 17, m_gold.size() );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );

    }

    ~Test()
    {
        cudaFree( m_input_d );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );

        cudaFree( m_output_d );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
    }

    void
    buildupPass()
    {
        TestPredicate predicate;
        predicate.m_input_dptr = m_input_d;
        predicate.m_threshold = m_threshold;

        m_hp.build( predicate, false );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
    }

    int
    getOutputCount()
    {
        m_count = m_hp.outputSize();
        BOOST_CHECK_EQUAL( (int)m_gold.size(), m_count );
        return m_count;
    }

    void
    extractPass()
    {
        TestEmitter emitter;

        emitter.m_input_dptr = m_input_d;
        emitter.m_output_dptr = m_output_d;
        m_hp.compact( emitter,
                      m_use_texfetch,
                      m_use_constmem,
                      false );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );
    }

    void
    verifyResults()
    {
        std::vector<float> output( m_gold.size() );

        cudaMemcpy( output.data(),
                    m_output_d,
                    sizeof(float)*output.size(),
                    cudaMemcpyDeviceToHost );
        BOOST_CHECK_EQUAL( cudaGetLastError(), cudaSuccess );

        for( size_t i=0; i<m_gold.size(); i++ ) {
            if( output[i] != m_gold[i] ) {
                BOOST_CHECK_EQUAL( output[i], m_gold[i] );
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

    float*              m_input_d;
    float*              m_output_d;
    std::vector<float>  m_input;
    std::vector<float>  m_gold;
    std::vector<float>  m_result;
    int                 m_input_size;
    int                 m_seed;
    bool                m_use_texfetch;
    bool                m_use_constmem;
    float               m_threshold;
};

