/* Copyright STIFTELSEN SINTEF 2010
 *
 * This file is part of the SIST Library.
 *
 * Author(s): Christopher Dyken, <christopher.dyken@sintef.no>
 *            Johan Seland, <johan.seland@sintef.no>
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
#include <cstdlib>
#include <cmath>
#include <string>
#include <algorithm>
#include <functional>
#include <cuda_runtime.h>
#include <sist/scan/scan.hpp>
#include <cudpp.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define BOOST_TEST

#ifdef BOOST_TEST
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ScanTest

#include <boost/test/unit_test.hpp>
#include <boost/test/parameterized_test.hpp>

#endif // BOOST_TEST

/** Helper macro that checks for CUDA errors, and exits if any. */
#define CHECK_CUDA do {                                                        \
    cudaError_t res = cudaGetLastError();                                      \
    if( res != cudaSuccess ) {                                                 \
        std::cerr << __FILE__ << '@'                                           \
                  << __LINE__ << ": CUDA error: "                              \
                  << cudaGetErrorString( res )                                 \
                  << std::endl;                                                \
        cudaThreadExit();                                                      \
        exit( EXIT_FAILURE );                                                  \
    }                                                                          \
} while(0)

struct ScanFixture {

    ScanFixture() {
        input.resize( 1024 );
        output.resize( input.size() + 1 );

        srand( 42 );
        for(size_t i=0; i<input.size(); i++ ) {
            input[i] = rand() & 0xffu;        
        }

        cudaMalloc( &input_d, sizeof(unsigned int)*(input.size()) );
        cudaMalloc( &scratch_d, sist::scan::scratchBufferBytesize( input.size() ) );
        cudaMalloc( &output_d, sizeof(unsigned int)*(output.size()) );
        cudaMemset( output_d, ~0u, sizeof(unsigned int)*output.size() );

        cudaMemcpy( input_d, input.data(), sizeof(unsigned int)*input.size(), cudaMemcpyHostToDevice );
    }

    ~ScanFixture() {
        cudaFree( input_d );
        cudaFree( output_d );
        cudaFree( scratch_d );

        cudaFree( input_d );
        cudaFree( output_d );
        cudaFree( scratch_d );
    }

    std::vector<unsigned int> input;
    std::vector<unsigned int> output;  

    unsigned int* input_d;
    unsigned int* output_d;
    unsigned int* scratch_d;
};
    

class AbstractScanBenchmark {
public:
    AbstractScanBenchmark( 
        unsigned int* input_d, unsigned int* output_d, 
        const std::vector<unsigned int>& input, std::vector<unsigned int>& output ) 
        : 
        input( input ),
        output( output ), 
        input_d( input_d ),
        output_d( output_d ),
        its( 100 ) 
    {
        cudaEventCreate( &start );
        cudaEventCreate( &stop );

    }

    ~AbstractScanBenchmark() {
        cudaEventDestroy( start );
        cudaEventDestroy( stop );
    }

    virtual void benchmarkScan( size_t N, float ref ) {
        ms = 0.0f;
        // WarmUp
        for(int i=0; i<(its+9)/10; i++) {
            doScan( N );
        }

        // Benchmark
        cudaEventRecord( start );
        for( int i = 0; i < its; ++i ) {
              doScan( N );
        }
        cudaEventRecord( stop );
        cudaMemcpy( output.data(), output_d, sizeof(unsigned int)*(output.size()), cudaMemcpyDeviceToHost  );
        cudaEventSynchronize( stop );

        cudaEventElapsedTime( &ms, start, stop );

        validate_output( N, ref );
    }

    virtual void validate_output( size_t N, float ref ) = 0;    
    virtual void doScan( size_t N ) = 0;

    int fails;
    float ms;
    const int its;

    cudaEvent_t start;
    cudaEvent_t stop;
    const std::vector<unsigned int>& input;
    std::vector<unsigned int>& output;
    unsigned int* input_d;
    unsigned int* output_d;
};


class BenchmarkThrustScan : public AbstractScanBenchmark {
public:
    BenchmarkThrustScan(
        unsigned int* input_d, unsigned int* output_d,
        const std::vector<unsigned int>& input, std::vector<unsigned int>& output  )
        :
          AbstractScanBenchmark( input_d, output_d, input, output ),
          input_d( thrust::device_pointer_cast( input_d ) ),
          output_d( thrust::device_pointer_cast( output_d ) )/*,
          input( input ), output( output )*/
    {
         //cudaEventCreate( &start );
        //cudaEventCreate( &stop );
    }
/*
    void benchmarkScan( size_t N, float ref ) {
            //its = 100;
         ms = 0;
                             
          for(int i=0; i<(its+9)/10; i++) {
            doScan( N );
          }
          
          cudaEventRecord( start );
          for( int i = 0; i < its; ++i ) {
              doScan( N );
          }
          cudaEventRecord( stop );
          cudaMemcpy( output.data(), output_d.get(), sizeof(unsigned int)*(output.size()), cudaMemcpyDeviceToHost  );
          cudaEventSynchronize( stop );

          cudaEventElapsedTime( &ms, start, stop );

          validate_output( N, ref );

    }
    */
    void doScan( size_t N ) {
        thrust::exclusive_scan( input_d, input_d + N, output_d );
    }

    void validate_output(size_t N, float ref ) {
        fails = 0;
        unsigned int sum = 0;
        for(size_t i=0; i<N; i++ ) {
            if( output[i] != sum ) {
                fails++;
            }
            sum += input[i];
        }
        std::cerr << "\tTHRUST\tE="<< fails
            << "\t"
            << "\t"
            << "\ttime=" << (ms/its) << "ms"
             << "\tspeedup=" << (ref/(ms/its)) <<"X.\n";
    }
 
    int fails;
private:    
    thrust::device_ptr<unsigned int> input_d;
    thrust::device_ptr<unsigned int> output_d;
    
    //size_t input_elements;
    //const std::vector<unsigned int>& input;
    //thrust::host_vector<unsigned int> output;
    
    //float ms;
    //int its;

    

    //cudaEvent_t start;
    //cudaEvent_t stop;

};

//class BenchmarkInclusiveScan : public AbstractScanBenchmark {
//};



#ifdef BOOST_TEST
BOOST_FIXTURE_TEST_CASE( BenchMarkThrustScan, ScanFixture ) {
//BOOST_AUTO_TEST_CASE( BenchMarkThrustScan/*, ScanFixture*/ ) {
    BenchmarkThrustScan bench( input_d, output_d, input, output );
    bench.benchmarkScan( input.size(), 0.0f );
    BOOST_CHECK_EQUAL( bench.fails, 0 );
}

int old_main( int argc, char** argv )
#else
int main( int argc, char** argv )
#endif // BOOST_TEST
{
    int cuda_device = 0;

    for( int i=1; i<argc; i++ ) {
        std::string arg( argv[i] );
        if( (arg == "-d") && (i+1 < argc) ) {
            cuda_device = atoi( argv[i+1] );
        }

    }

    int device_count;
    cudaGetDeviceCount( &device_count );
    if( device_count == 0 ) {
        std::cerr << "No CUDA devices present, exiting." << std::endl;
        return -1;
    }
    for(int dev=0; dev<device_count; dev++ ) {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties( &dev_prop, dev );
        std::cerr << "CUDA device " << dev << ": "
                  << dev_prop.name << ", "
                  << "compute cap " << dev_prop.major << "."  << dev_prop.minor
                  << std::endl;
    }
    if( (cuda_device < 0 ) || (device_count <= cuda_device) ) {
        std::cerr << "Illegal device " << cuda_device << ", exiting." << std::endl;
        return -1;
    }
    std::cerr << "Using cuda device " << cuda_device << std::endl;
    cudaSetDevice( cuda_device );


    std::vector<unsigned int> input( 0x03fffc00 );
    std::vector<unsigned int> output( input.size()+1 );

    unsigned int* input_d;
    cudaMalloc( &input_d, sizeof(unsigned int)*(input.size()) );


    unsigned int* scratch_d;
    cudaMalloc( &scratch_d, sist::scan::scratchBufferBytesize( input.size() ) );

    unsigned int* output_d;
    cudaMalloc( &output_d, sizeof(unsigned int)*(output.size()) );

    CHECK_CUDA;

    CUDPPHandle cudpp_handle;
    cudppCreate( &cudpp_handle );




#if 1
    for(int N=input.size(); N>0; N = N/2.15 ) {
#else
    if( 1 ) {
        int N=743;
#endif
        std::cerr << "N=" << N << "\n";
        //std::default_random_engine dre;
        //std::uniform_int_distribution<unsigned int> di( 0, 255 );
        //auto dice = std::bind( di, dre );

        //std::generate( input.begin(), input.end(), dice );

        srand( 42 );
        for(size_t i=0; i<input.size(); i++ ) {
            input[i] = rand() & 0xffu;
        }

        cudaMemcpy( input_d, input.data(), sizeof(unsigned int)*input.size(), cudaMemcpyHostToDevice );

        float ref;
        if(1) {
            float ms;
            int its = 100;
            cudaEvent_t start, stop;
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaMemset( output_d, ~0u, sizeof(unsigned int)*output.size() );

            CUDPPConfiguration cudpp_config;
            cudpp_config.op 		= CUDPP_ADD;
            cudpp_config.datatype 	= CUDPP_UINT;
            cudpp_config.algorithm 	= CUDPP_SCAN;
            cudpp_config.options	= CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

            CUDPPHandle scanplan = 0;
            CUDPPResult cudpp_res = cudppPlan( cudpp_handle, &scanplan, cudpp_config, N, 1, 0 );
            if( cudpp_res != CUDPP_SUCCESS ) {
                std::cerr << "CUDPP Failed to init, exiting.\n";
                exit( EXIT_FAILURE );
            }

            for(int i=0; i<(its+9)/10; i++) {
                cudpp_res = cudppScan( scanplan, output_d, input_d, N );
                if( cudpp_res != CUDPP_SUCCESS ) {
                    std::cerr << "CUDPP Failed in warmup.\n";
                }
            }

            cudaEventRecord( start );
            for(int i=0; i<its; i++) {
                cudpp_res = cudppScan( scanplan, output_d, input_d, N );
                if( cudpp_res != CUDPP_SUCCESS ) {
                    std::cerr << "CUDPP Failed in warmup.\n";
                    exit( EXIT_FAILURE );
                }
            }
            cudaEventRecord( stop );
            cudaEventSynchronize( stop );
            cudaMemcpy( output.data(), output_d, sizeof(unsigned int)*(output.size()), cudaMemcpyDeviceToHost  );
            cudaEventElapsedTime( &ms, start, stop );
            CHECK_CUDA;

            int fails = 0;
            unsigned int sum = 0;
            for(int i=0; i<N; i++ ) {
                if( output[i] != sum ) {
                    fails++;
                }
                sum += input[i];
            }
            std::cerr << "\tCUDPP\tE="<< fails
                      << "\t"
                      << "\t"
                      << "\ttime=" << (ms/its) << "ms.\n";
            ref = ms/its;
        }

        BenchmarkThrustScan thrustScan( input_d, output_d, input, output );
        thrustScan.benchmarkScan( N, ref );

        // inclusive scan
        if(1) {
            float ms;
            int its = 100;
            cudaEvent_t start, stop;
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaMemset( output_d, ~0u, sizeof(unsigned int)*output.size() );
            for(int i=0; i<(its+9)/10; i++) {
                sist::scan::inclusiveScan( output_d,
                                     scratch_d,
                                     input_d,
                                     N );
            }
            cudaEventRecord( start, 0 );
            for(int i=0; i<its; i++) {
                sist::scan::inclusiveScan( output_d,
                                     scratch_d,
                                     input_d,
                                     N );
            }
            cudaEventRecord( stop );
            cudaEventSynchronize( stop );
            cudaMemcpy( output.data(), output_d, sizeof(unsigned int)*(output.size()), cudaMemcpyDeviceToHost  );
            cudaEventElapsedTime( &ms, start, stop );
            CHECK_CUDA;

            int fails = 0;
            unsigned int sum = 0;
            for(int i=0; i<N; i++ ) {
                sum += input[i];
                if( output[i] != sum ) {
                    fails++;
                }
            }
            std::cerr << "\tin\tE="<< fails
                      << "\tS=" << (output[N]==~0u?"ok":"ERR" )
                      << "\t"
                      << "\ttime=" << (ms/its) << "ms"
                      << "\tspeedup=" << (ref/(ms/its)) <<"X.\n";
        }

        // inclusive scan with sum
        if(1) {
            unsigned int* zerocopy;
            cudaHostAlloc( &zerocopy, sizeof(unsigned int), cudaHostAllocMapped );

            unsigned int* zerocopy_d;
            cudaHostGetDevicePointer( &zerocopy_d, zerocopy, 0 );
            CHECK_CUDA;

            *zerocopy = 42;

            float ms;
            int its = 100;
            cudaEvent_t start, stop;
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaMemset( output_d, ~0u, sizeof(unsigned int)*output.size() );
            for(int i=0; i<(its+9)/10; i++) {
                sist::scan::inclusiveScanWriteSum( output_d,
                                             zerocopy_d,
                                             scratch_d,
                                             input_d,
                                             N );
            }
            cudaEventRecord( start, 0 );
            for(int i=0; i<its; i++) {
                sist::scan::inclusiveScanWriteSum( output_d,
                                             zerocopy_d,
                                             scratch_d,
                                             input_d,
                                             N );
            }
            cudaEventRecord( stop );
            cudaEventSynchronize( stop );
            cudaMemcpy( output.data(), output_d, sizeof(unsigned int)*(output.size()), cudaMemcpyDeviceToHost  );
            cudaEventElapsedTime( &ms, start, stop );
            CHECK_CUDA;

            int fails = 0;
            unsigned int sum = 0;
            for(int i=0; i<N; i++ ) {
                sum += input[i];
                if( output[i] != sum ) {
                    fails++;
                }
            }
            std::cerr << "\tin+S\tE="<< fails
                      << "\tS=" << (output[N]==~0u?"ok":"ERR" )
                      << "\tZ=" << ((*zerocopy == sum) ? "ok":"ERR" )
                      << "\ttime=" << (ms/its) << "ms"
                      << "\tspeedup=" << (ref/(ms/its)) <<"X.\n";
            cudaFreeHost( zerocopy );
        }

        // exclusive scan
        if(1) {
            float ms;
            int its = 100;
            cudaEvent_t start, stop;
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaMemset( output_d, ~0u, sizeof(unsigned int)*output.size() );
            for(int i=0; i<(its+9)/10; i++) {
                sist::scan::exclusiveScan( output_d,
                                     scratch_d,
                                     input_d,
                                     N );
            }
            cudaEventRecord( start, 0 );
            for(int i=0; i<its; i++) {
                sist::scan::exclusiveScan( output_d,
                                     scratch_d,
                                     input_d,
                                     N );
            }
            cudaEventRecord( stop );
            cudaEventSynchronize( stop );
            cudaMemcpy( output.data(), output_d, sizeof(unsigned int)*(output.size()), cudaMemcpyDeviceToHost  );
            cudaEventElapsedTime( &ms, start, stop );
            CHECK_CUDA;

            int fails = 0;
            unsigned int sum = 0;
            for(int i=0; i<N; i++ ) {
                if( output[i] != sum ) {
                    fails++;
                }
                sum += input[i];
            }
            std::cerr << "\tex\tE="<< fails
                      << "\tS=" << (output[N]==~0u?"ok":"ERR" )
                      << "\t"
                      << "\ttime=" << (ms/its) << "ms"
                      << "\tspeedup=" << (ref/(ms/its)) <<"X.\n";
        }


        // exclusive scan with sum
        if(1) {
            unsigned int* zerocopy;
            cudaHostAlloc( &zerocopy, sizeof(unsigned int), cudaHostAllocMapped );

            unsigned int* zerocopy_d;
            cudaHostGetDevicePointer( &zerocopy_d, zerocopy, 0 );
            CHECK_CUDA;

            *zerocopy = 42;

            float ms;
            int its = 100;
            cudaEvent_t start, stop;
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaMemset( output_d, ~0u, sizeof(unsigned int)*output.size() );
            for(int i=0; i<(its+9)/10; i++) {
                sist::scan::exclusiveScanWriteSum( output_d,
                                             zerocopy_d,
                                             scratch_d,
                                             input_d,
                                             N );
            }
            cudaEventRecord( start, 0 );
            for(int i=0; i<its; i++) {
                sist::scan::exclusiveScanWriteSum( output_d,
                                             zerocopy_d,
                                             scratch_d,
                                             input_d,
                                             N );
            }
            cudaEventRecord( stop );
            cudaEventSynchronize( stop );
            cudaMemcpy( output.data(), output_d, sizeof(unsigned int)*(output.size()), cudaMemcpyDeviceToHost  );
            cudaEventElapsedTime( &ms, start, stop );
            CHECK_CUDA;

            int fails = 0;
            unsigned int sum = 0;
            for(int i=0; i<N; i++ ) {
                if( output[i] != sum ) {
                    fails++;
                }
                sum += input[i];
            }
            std::cerr << "\tex+S\tE="<< fails
                      << "\tS=" << (output[N]==~0u?"ok":"ERR" )
                      << "\tZ=" << ((*zerocopy == sum) ? "ok":"ERR" )
                      << "\ttime=" << (ms/its) << "ms"
                      << "\tspeedup=" << (ref/(ms/its)) <<"X.\n";
            cudaFreeHost( zerocopy );
        }

        // exlusive with padded sum
        if(1) {
            float ms;
            int its = 100;
            cudaEvent_t start, stop;
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaMemset( output_d, ~0u, sizeof(unsigned int)*output.size() );
            for(int i=0; i<(its+9)/10; i++) {
                sist::scan::exclusiveScanPadWithSum( output_d,
                                           scratch_d,
                                           input_d,
                                           N );
            }
            cudaEventRecord( start, 0 );
            for(int i=0; i<its; i++) {
                sist::scan::exclusiveScanPadWithSum( output_d,
                                               scratch_d,
                                               input_d,
                                               N );
            }
            cudaEventRecord( stop );
            cudaEventSynchronize( stop );
            cudaMemcpy( output.data(), output_d, sizeof(unsigned int)*(output.size()), cudaMemcpyDeviceToHost  );
            cudaEventElapsedTime( &ms, start, stop );
            CHECK_CUDA;

            int fails = 0;
            unsigned int sum = 0;
            for(int i=0; i<N; i++ ) {
                if( output[i] != sum ) {
                    fails++;
                }
                sum += input[i];
            }
            std::cerr << "\tex+P\tE="<< fails
                      << "\tS=" << (output[N]==sum?"ok":"ERR" )
                      << "\t"
                      << "\ttime=" << (ms/its) << "ms"
                      << "\tspeedup=" << (ref/(ms/its)) <<"X.\n";
        }


        // exclusive scan with padded and written sum
        if(1) {
            unsigned int* zerocopy;
            cudaHostAlloc( &zerocopy, sizeof(unsigned int), cudaHostAllocMapped );

            unsigned int* zerocopy_d;
            cudaHostGetDevicePointer( &zerocopy_d, zerocopy, 0 );
            CHECK_CUDA;

            *zerocopy = 42;

            float ms;
            int its = 100;
            cudaEvent_t start, stop;
            cudaEventCreate( &start );
            cudaEventCreate( &stop );
            cudaMemset( output_d, ~0u, sizeof(unsigned int)*output.size() );
            for(int i=0; i<(its+9)/10; i++) {
                sist::scan::exclusiveScanPadWithSumWriteSum( output_d,
                                                       zerocopy_d,
                                                       scratch_d,
                                                       input_d,
                                                       N );
            }
            cudaEventRecord( start, 0 );
            for(int i=0; i<its; i++) {
                sist::scan::exclusiveScanPadWithSumWriteSum( output_d,
                                                       zerocopy_d,
                                                       scratch_d,
                                                       input_d,
                                                       N );
            }
            cudaEventRecord( stop );
            cudaEventSynchronize( stop );
            cudaMemcpy( output.data(), output_d, sizeof(unsigned int)*(output.size()), cudaMemcpyDeviceToHost  );
            cudaEventElapsedTime( &ms, start, stop );
            CHECK_CUDA;

            int fails = 0;
            unsigned int sum = 0;
            for(int i=0; i<N; i++ ) {
                if( output[i] != sum ) {
                    fails++;
                }
                sum += input[i];
            }
            std::cerr << "\tex+P+S\tE="<< fails
                      << "\tS=" << (output[N]==sum?"ok":"ERR" )
                      << "\tZ=" << ((*zerocopy == sum) ? "ok":"ERR" )
                      << "\ttime=" << (ms/its) << "ms"
                      << "\tspeedup=" << (ref/(ms/its)) <<"X.\n";
            cudaFreeHost( zerocopy );
        }



    }

    return EXIT_SUCCESS;

}

