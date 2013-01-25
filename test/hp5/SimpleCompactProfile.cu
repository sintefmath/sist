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
#define BOOST_CHECK_EQUAL(a,b)
#define BOOST_CHECK_NE(a,b)

#include <sstream>
#include "SimpleCompactTest.hpp"


struct Parameters {
    Parameters( int input_size,
                bool use_texfetch,
                bool use_constmem,
                float threshold )
        : m_input_size( input_size ),
          m_use_texfetch( use_texfetch ),
          m_use_constmem( use_constmem ),
          m_threshold( threshold )
    {}

    int     m_input_size;
    bool    m_use_texfetch;
    bool    m_use_constmem;
    float   m_threshold;
};



int main(int argc, char** argv)
{


    std::vector<Parameters> setup;


    setup.push_back( Parameters( 5000000, false, false, 0.1 ) );
    setup.push_back( Parameters( 5000000, false, true, 0.1 ) );
    setup.push_back( Parameters( 5000000, true, false, 0.1 ) );
    setup.push_back( Parameters( 5000000, true, true, 0.1 ) );

    setup.push_back( Parameters( 5000000, true, true, 0.5 ) );
    setup.push_back( Parameters( 5000000, false, false, 0.5 ) );
    setup.push_back( Parameters( 5000000, false, true, 0.5 ) );
    setup.push_back( Parameters( 5000000, true, false, 0.5 ) );

    setup.push_back( Parameters( 5000000, true, true, 0.9 ) );
    setup.push_back( Parameters( 5000000, false, false, 0.9 ) );
    setup.push_back( Parameters( 5000000, false, true, 0.9 ) );
    setup.push_back( Parameters( 5000000, true, false, 0.9 ) );

    cudaEvent_t event_a, event_b;
    cudaEventCreate( &event_a );
    cudaEventCreate( &event_b );


    std::cout << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
    std::cout << "<profile>" << std::endl;
    for(size_t i=0; i<setup.size(); i++ ) {
        const Parameters& param = setup[i];

        Test test( param.m_input_size,
                   42,
                   param.m_use_texfetch,
                   param.m_use_constmem,
                   param.m_threshold );

        // pass 1
        for( int i=0; i<10; i++ ) {
            test.buildupPass();
        }
        cudaEventRecord( event_a );
        for( int i=0; i<100; i++ ) {
            test.buildupPass();
        }
        cudaEventRecord( event_b );
        cudaEventSynchronize( event_b );

        float pass1 = 0.f;
        cudaEventElapsedTime( &pass1, event_a, event_b );
        pass1 = pass1/100.f;

        // pass 1 + 2
        for( int i=0; i<10; i++ ) {
            test.buildupPass();
            test.getOutputCount();
        }
        cudaEventRecord( event_a );
        for( int i=0; i<100; i++ ) {
            test.buildupPass();
            test.getOutputCount();
        }
        cudaEventRecord( event_b );
        cudaEventSynchronize( event_b );

        float pass12 = 0.f;
        cudaEventElapsedTime( &pass12, event_a, event_b );
        pass12 = pass12/100.f;



        for( int i=0; i<10; i++ ) {
            test.buildupPass();
            test.getOutputCount();
            test.extractPass();
        }

        cudaEventRecord( event_a );
        for( int i=0; i<100; i++ ) {
            test.buildupPass();
            test.getOutputCount();
            test.extractPass();
        }
        cudaEventRecord( event_b );
        cudaEventSynchronize( event_b );

        float pass123 = 0.f;
        cudaEventElapsedTime( &pass123, event_a, event_b );
        pass123 = pass123/100.f;


        std::cout << "  <run id=\"SimpleCompact-"
                  << param.m_input_size << "@"
                  << param.m_threshold << "-"
                  << (param.m_use_texfetch?"texfetch":"notexfetch") << "-"
                  << (param.m_use_constmem?"constmem":"noconstmem")
                  << "\">" << std::endl;
        std::cout << "    <input>" << test.m_input_size << "</input>" << std::endl;
        std::cout << "    <output>" << test.m_count << "</output>" << std::endl;
        std::cout << "    <pass1 unit=\"ms\">" << pass1 << "</pass1>"<< std::endl;
        std::cout << "    <pass12 unit=\"ms\">" << pass12 << "</pass12>"<< std::endl;
        std::cout << "    <pass123 unit=\"ms\">" << pass123 << "</pass123>"<< std::endl;
        std::cout << "    <throughput>" << ((param.m_input_size/pass123)/1000000) << "</throughput>"<< std::endl;
        std::cout << "  </run>" << std::endl;
    }
    std::cout << "</profile>" << std::endl;




}
