CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(SIST)

OPTION( ENABLE_TESTS "Enable building of tests" ON )

#SET( HP5_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" CACHE PATH " ")
#ADD_SUBDIRECTORY( test )

INCLUDE_DIRECTORIES(
        "${PROJECT_SOURCE_DIR}/include"
)
FIND_PACKAGE(CUDA REQUIRED)

FILE( GLOB SIST_HDRS "include/sist/hp5/*.hpp" )
SET_SOURCE_FILES_PROPERTIES( ${SIST_HDRS} PROPERTIES HEADER_FILE_ONLY TRUE )
SOURCE_GROUP( "SIST headers" FILES ${SIST_HDRS} )

IF( ENABLE_TESTS )
	FIND_PACKAGE(Boost REQUIRED COMPONENTS unit_test_framework)

        FILE( GLOB SIST_TEST_HDRS "test/hp5/*.hpp" )
	SET_SOURCE_FILES_PROPERTIES( ${SIST_TEST_HDRS} ROPERTIES HEADER_FILE_ONLY TRUE )
	SOURCE_GROUP( "SIST test headers" FILES ${SIST_TEST_HDRS} )

        FILE( GLOB SIST_TEST_SRCS "test/hp5/*.cpp" "test/hp5/*.cu" )
	SOURCE_GROUP( "SIST test sources" FILES ${SIST_TEST_SRCS} )

        CUDA_ADD_EXECUTABLE( test_simple_compact ${SIST_HDRS} "test/hp5/SimpleCompactUnit.cu" )
	TARGET_LINK_LIBRARIES( test_simple_compact ${Boost_LIBRARIES} )

        CUDA_ADD_EXECUTABLE( profile_simple_compact ${SIST_HDRS} "test/hp5/SimpleCompactProfile.cu" )
	TARGET_LINK_LIBRARIES( profile_simple_compact )

        CUDA_ADD_EXECUTABLE( test_multi_compact ${SIST_HDRS} "test/hp5/MultiCompactUnit.cu" )
	TARGET_LINK_LIBRARIES( test_multi_compact ${Boost_LIBRARIES} )

        CUDA_ADD_EXECUTABLE( profile_multi_compact ${SIST_HDRS} "test/hp5/MultiCompactProfile.cu" )
	TARGET_LINK_LIBRARIES( profile_multi_compact )

ENDIF( ENABLE_TESTS )

