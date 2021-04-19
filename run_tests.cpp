//#define INCLUDE_TESTS
#define DEBUG_OUTPUT "output/"

#include <iostream>
#include "gtest/gtest.h"
#include <opt_cnn.hpp>
#include <sstream>
namespace Tests {

      	
	class OptimizationTests :  public ::testing::Test {
		
	};

	TEST_F(OptimizationTests, level_0_fc) {
		fc_test_activate<opt_fc_layer_t>   (1,1,1,1,1,1);
		fc_test_calc_grads<opt_fc_layer_t> (1,1,1,1,1,1);
		fc_test_fix_weights<opt_fc_layer_t>(1,1,1,1,1,1);
		fc_test<opt_fc_layer_t>            (1,1,1,1,1,1);
	}	
			  
	TEST_F(OptimizationTests, level_1_fc) {
#define FC_TEST_1(method)						\
		method<opt_fc_layer_t>(4,  4,  4,  4, 4, 1);\
		method<opt_fc_layer_t>(8,  8,  2,  2, 16,1);\
		method<opt_fc_layer_t>(32, 32, 16, 8, 8, 1);
		
		FC_TEST_1(fc_test_activate);
		FC_TEST_1(fc_test_calc_grads);
		FC_TEST_1(fc_test_fix_weights);
		FC_TEST_1(fc_test);
	}
	TEST_F(OptimizationTests, level_2_fc) {
#define FC_TEST_2(method)						\
		method<opt_fc_layer_t>(4,  6,  6,  6,  6,  1);\
		method<opt_fc_layer_t>(12, 12, 3,  2,  3,  1);\
		method<opt_fc_layer_t>(16, 96, 2,  2,  12, 1);
		
		FC_TEST_2(fc_test_activate);
		FC_TEST_2(fc_test_calc_grads);
		FC_TEST_2(fc_test_fix_weights);
		FC_TEST_2(fc_test);
	}

	TEST_F(OptimizationTests, level_3_fc) {
#define FC_TEST_3(method)						\
		method<opt_fc_layer_t>(3,  7,  13, 3, 7,  1);\
		method<opt_fc_layer_t>(31, 29, 5,  5, 13, 1);\
		method<opt_fc_layer_t>(3,  17, 31, 3, 23, 1);
		
		FC_TEST_3(fc_test_activate);
		FC_TEST_3(fc_test_calc_grads);
		FC_TEST_3(fc_test_fix_weights);
		FC_TEST_3(fc_test);
	}

	TEST_F(OptimizationTests, level_4_fc) {
		for (int i = 0; i < 10; i++) {
			srand(i);
			int x = RAND_LARGE(16);
			int y = RAND_LARGE(24);
			int z = RAND_LARGE(24);
			int b = RAND_LARGE(16);
			int out = RAND_LARGE(8);
			
			fc_test_activate<opt_fc_layer_t>(x,y,z,b,out,1);
			fc_test_calc_grads<opt_fc_layer_t>(x,y,z,b,out,1);
			fc_test_fix_weights<opt_fc_layer_t>(x,y,z,b,out,1);
			fc_test<opt_fc_layer_t>(x,y,z,b,out,1);

		}
		
	}

	class LabTests :  public ::testing::Test {
		
	};
	TEST_F(LabTests, test_lab_model) {
		for (int i = 0; i < 3; i++) {
			fc_test<opt_fc_layer_t>(32, 32, 3, 1, 100, i);
		}
	}

}

int main(int argc, char **argv) {
	if (argc >= 2) {
		if (!strcmp(argv[1], "--print-deltas")) {
			tensor_t<double>::diff_prints_deltas = true;
			argc--;
			argv++;
		}
	}
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
