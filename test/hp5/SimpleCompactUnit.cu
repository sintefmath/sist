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
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SimpleCompactUnit
#include <boost/test/unit_test.hpp>

#include "SimpleCompactTest.hpp"



BOOST_AUTO_TEST_CASE( foo_test_100000_42_1_1_01 )
{
    Test test( 100000, 42, true, true, 0.1f );
    test.run();
}

BOOST_AUTO_TEST_CASE( foo_test_5000000_42_1_1_01 )
{
    Test test( 5000000, 42, true, true, 0.1f );
    test.run();
}
