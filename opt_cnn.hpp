#pragma once
#include"CNN/canela.hpp"
#include "pin_tags.h"


/* Here is an example usage of DUMP_START(), DUMP_STOP() and START_TRACE()
 * 	START_TRACE(); //This will tell moneta where to start tracing
 *
 *      // This will help moneta know what to trace
 *      // Here all the elements in weights are traced. data is an internal data structure holding all data accross x/y/z/b
 * 	DUMP_START("weights", (void *) &(weights.data[0]), (void *) &(weights.data[weights.element_count() - 1]), true);
 *      
 *	//ADD DUMP_START() CALLS HERE FOR OTHER DATA STRUCTURES IF NEEDED
 *
 * 	//the nested for loop in the activate function that you will copy into the opt_fc_layer_t class
 *      for ( int b = 0; b < in.size.y; b++ ) {
 *      	for ( int i = 0; i < in.size.x; i++ ) {
 *              	for ( int n = 0; n < out.size.x; n++ ) {
 *                      	double in_val = in(i, b, 0);
 *                              double weight_val = weights( i, n, 0 );
 *                              double mul_val = in_val * weight_val;
 *                              double acc_val = activator_input(n, 0, 0, b) + mul_val;
 *                              activator_input(n, 0, 0, b) = acc_val;
 *                       }
 *               }
 *      }
 *
 *      //this will help moneta know when to stop stop tracing the weights array
 *	DUMP_STOP("weights");
 *
 *	//REMEMBER TO DUMP_STOP() IF YOU ARE TRACING OTHER ARRAYS
 *
 */



// This class replaces its parent classes in the implementation of the learning
// model for this lab.  If you override functions in the baseclass by
// implementing them here, then the code here will run instead of the baseclass
// code.
//
// You should copy the functions you want to optimize into these classes, and
// confirm that the correctness tests pass.  Then, you can start modifying them
// to make them faster.
//
// The source code Canela is in /course/CSE141pp-SimpleCNN/CNN
class opt_fc_layer_t : public fc_layer_t
{
public:
	opt_fc_layer_t( tdsize in_size, int out_size ) : fc_layer_t(in_size, out_size) {

	}

			
};

