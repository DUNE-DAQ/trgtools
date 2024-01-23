/**
 * @file TPGtools_unittest.cxx
 *
 * This file provides a skeleton off of which developers can write
 * unit tests for their package. The file is meant to be renamed as
 * well as edited (where editing includes replacing this comment with
 * an actual description of the unit test suite)
 *
 * This is part of the DUNE DAQ Application Framework, copyright 2020.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#define BOOST_TEST_MODULE TPGtools_unittest // NOLINT

#include "boost/test/unit_test.hpp"

BOOST_AUTO_TEST_SUITE(TPGtools_unittest)

BOOST_AUTO_TEST_CASE(ReplaceThisTest)
{
  BOOST_TEST_MESSAGE("This unit test is designed to fail. If you're reading this it means developers haven't replaced "
                     "this test with any actual unit tests test of their code.");
  BOOST_REQUIRE_EQUAL(1, 2);
}

BOOST_AUTO_TEST_SUITE_END()
