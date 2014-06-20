CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(SIST)

OPTION( EXTEND_CMAKE_MODULE_PATH "Extend the CMAKE_MODULE_PATH variable with user directories?" ON)
OPTION( ENABLE_TESTS "Enable building of tests" ON )
OPTION( ENABLE_SSE  "Build SSE tidbits" ON )
OPTION( ENABLE_OPENCL "Build OpenCL tidbits" ON )
OPTION( ENABLE_CUDA "Build CUDA tidbits" ON )


IF( EXTEND_CMAKE_MODULE_PATH )
  SET( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                         ${CMAKE_INSTALL_PREFIX}/share/cmake/Modules
                         "./cmake/Modules"
                         "~/cmake/Modules"
                        "$ENV{HOMEPATH}/cmake/Modules"
                        "../siut/cmakemacros"
                        "../siut"
                        "."
                        "C:/cmake/Modules" )
ENDIF( EXTEND_CMAKE_MODULE_PATH )

INCLUDE_DIRECTORIES(
    "${PROJECT_SOURCE_DIR}/include"
)

IF( ENABLE_CUDA )
    FIND_PACKAGE(CUDA REQUIRED)
    OPTION( CUDA_GEN_SM20 "Generate compute capability 2.0 code" OFF )
    OPTION( CUDA_GEN_SM30 "Generate compute capability 3.0 code" OFF )
    OPTION( CUDA_GEN_FWD  "Generate beyond 3.0 compatible ptx code" ON )
    OPTION( CUDA_DEBUG_GDB "Add flags thad cuda-gdb wants" OFF )
    OPTION( CUDA_VERBOSE_PTXAS "Let ptxas be verbose about resource usage" ON )

	ADD_DEFINITIONS( -D_ITERATOR_DEBUG_LEVEL=0 )

    IF( CUDA_VERBOSE_PTXAS )
        SET( MY_NVCC_OPTIONS "${MY_NVCC_OPTIONS} --ptxas-options=-v"  )
    ENDIF( CUDA_VERBOSE_PTXAS )
    IF( CUDA_DEBUG_GDB )
        SET( MY_NVCC_OPTIONS "${MY_NVCC_OPTIONS} -g -G" )
    ENDIF( CUDA_DEBUG_GDB )
    IF( CUDA_GEN_SM20 )
        SET( MY_NVCC_OPTIONS "${MY_NVCC_OPTIONS} -gencode=arch=compute_20,code=sm_20" )
    ENDIF( CUDA_GEN_SM20 )
    IF( CUDA_GEN_SM30 )
        SET( MY_NVCC_OPTIONS "${MY_NVCC_OPTIONS} -gencode=arch=compute_30,code=sm_30" )
    ENDIF( CUDA_GEN_SM30 )
    IF( CUDA_GEN_FWD )
        SET( MY_NVCC_OPTIONS "${MY_NVCC_OPTIONS} -gencode=arch=compute_30,code=compute_30" )
    ENDIF( CUDA_GEN_FWD )

    FILE( GLOB CUSIST_HDRS
        "include/sist/scan/*.hpp"
        "include/sist/hp5/*.hpp" )
    SET_SOURCE_FILES_PROPERTIES( ${CUSIST_HDRS} PROPERTIES HEADER_FILE_ONLY TRUE )
    SOURCE_GROUP( "CUSIST headers" FILES ${CUSIST_HDRS} )

    FILE( GLOB CUSIST_SRCS
        "src/scan/*.cu"
        "src/scan/*.cpp")
    SOURCE_GROUP( "CUSIST sources" FILES ${CUSIST_SRCSS} )

    SET(CUDA_NVCC_FLAGS ${MY_NVCC_OPTIONS})
    CUDA_ADD_LIBRARY( cusist STATIC ${CUSIST_HDRS} ${CUSIST_SRCS} )
    
    TARGET_LINK_LIBRARIES( cusist )


    IF( ENABLE_TESTS )
        FIND_PACKAGE(Boost REQUIRED COMPONENTS unit_test_framework)
        
        # The scan test compares itself against cudpp
        FIND_LIBRARY( CUDPP_LIBRARY cudpp PATHS /work/cudpp_src_2.0/lib )
        FIND_PATH( CUDPP_INCLUDE_DIR cudpp.h PATHS /work/cudpp_src_2.0/include )
        INCLUDE_DIRECTORIES(
            ${CUDPP_INCLUDE_DIR}
            ${Boost_INCLUDE_DIRS}
        )

        ADD_DEFINITIONS( -DCUDPP_STATIC_LIB )
        # Needed for Boost-autolinking on Windows.
        # This avoids conflict if you have vc100 and vc110 compiled Boost-libs.
        LINK_DIRECTORIES( ${Boost_LIBRARY_DIRS} )

        FILE( GLOB CUSIST_TEST_HDRS "test/scan/*.hpp"
                                    "test/hp5/*.hpp" )
        SET_SOURCE_FILES_PROPERTIES( ${CUSIST_TEST_HDRS} ROPERTIES HEADER_FILE_ONLY TRUE )
        SOURCE_GROUP( "CUSIST test headers" FILES ${CUSIST_TEST_HDRS} )

        FILE( GLOB CUSIST_TEST_SRCS "test/scan/*.cu"
                                    "test/scan/*.cpp"
                                    "test/hp5/*.cpp"
                                    "test/hp5/*.cu" )
        SOURCE_GROUP( "CUSIST test sources" FILES ${CUSIST_TEST_SRCS} )

        CUDA_ADD_EXECUTABLE( test_scan ${CUSIST_HDRS}
                             "test/scan/ScanTest.cpp" )
        TARGET_LINK_LIBRARIES( test_scan cusist ${CUDPP_LIBRARY} )

        CUDA_ADD_EXECUTABLE( test_simple_compact ${CUSIST_HDRS}
                             "test/hp5/SimpleCompactTest.hpp"
                             "test/hp5/SimpleCompactUnit.cu" )
        TARGET_LINK_LIBRARIES( test_simple_compact ${Boost_LIBRARIES} )

        CUDA_ADD_EXECUTABLE( profile_simple_compact ${CUSIST_HDRS}
                             "test/hp5/SimpleCompactTest.hpp"
                             "test/hp5/SimpleCompactProfile.cu" )
        TARGET_LINK_LIBRARIES( profile_simple_compact )

        CUDA_ADD_EXECUTABLE( test_multi_compact ${CUSIST_HDRS}
                             "test/hp5/MultiCompactTest.hpp"
                             "test/hp5/MultiCompactUnit.cu" )
        TARGET_LINK_LIBRARIES( test_multi_compact ${Boost_LIBRARIES} )

        CUDA_ADD_EXECUTABLE( profile_multi_compact ${CUSIST_HDRS}
                             "test/hp5/MultiCompactTest.hpp"
                             "test/hp5/MultiCompactProfile.cu" )
        TARGET_LINK_LIBRARIES( profile_multi_compact )
    ENDIF( ENABLE_TESTS )

ENDIF( ENABLE_CUDA )

IF( ENABLE_OPENCL )
    # nothing here yet
ENDIF( ENABLE_OPENCL )

IF( ENABLE_SSE )
    # nothing here yet
ENDIF( ENABLE_SSE )


