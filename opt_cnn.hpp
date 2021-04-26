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

        void activate( tensor_t<double>& in ) {
                copy_input(in);

                tdsize old_size = in.size;
                tdsize old_out_size = out.size;

				// std::cerr << out.element_count() << std::endl;

                // cast to correct shape
                in.size.x = old_size.x * old_size.y * old_size.z;
                in.size.y = old_size.b;
                in.size.z = 1;
                in.size.b = 1;

                out.size.x = old_out_size.x * old_out_size.y * old_out_size.z;
                out.size.y = old_out_size.b;
                out.size.z = 1;
                out.size.b = 1;

				// std::cerr << in.element_count() << std::endl;
				// std::cerr << out.element_count() << std::endl;
				// std::cerr << weights.element_count() << std::endl;
				// std::cerr << activator_input.element_count() << std::endl;

                for ( int b = 0; b < activator_input.size.b; b++) {
                        for ( int n = 0; n < activator_input.size.x; n++ ) {
                                activator_input(n, 0, 0, b) = 0;
                        }
                }

				// #define TILE_SIZE 4

				// for ( int nn = 0; nn < out.size.x; nn+=TILE_SIZE) {
				// 	for ( int b = 0; b < in.size.y; b++ ) {
				// 			for ( int n = nn; n < nn + TILE_SIZE && n < out.size.x; n++ ) {
				// 					for ( int i = 0; i < in.size.x; i++ ) {
				// 							double in_val = in(i, b, 0);
				// 							double weight_val = weights( i, n, 0 );
				// 							double mul_val = in_val * weight_val;
				// 							double acc_val = activator_input(n, 0, 0, b) + mul_val;
				// 							activator_input(n, 0, 0, b) = acc_val;
				// 					}
				// 			}
				// 	}
				// }

				// #define TILE_SIZE 16

				// for ( int nn = 0; nn < out.size.x; nn+=TILE_SIZE) {
				// 	for ( int b = 0; b < in.size.y; b++ ) {
				// 			for ( int n = nn; n < nn + TILE_SIZE && n < out.size.x; n++ ) {
				// 					for ( int i = 0; i < in.size.x; i++ ) {
				// 							double in_val = in(i, b, 0);
				// 							double weight_val = weights( i, n, 0 );
				// 							double mul_val = in_val * weight_val;
				// 							double acc_val = activator_input(n, 0, 0, b) + mul_val;
				// 							activator_input(n, 0, 0, b) = acc_val;
				// 					}
				// 			}
				// 	}
				// }
				

                // for ( int b = 0; b < in.size.y; b++ ) {
                //         for ( int n = 0; n < out.size.x; n++ ) {
                //                 for ( int i = 0; i < in.size.x; i++ ) {
                //                         double in_val = in(i, b, 0);
                //                         double weight_val = weights( i, n, 0 );
                //                         double mul_val = in_val * weight_val;
                //                         double acc_val = activator_input(n, 0, 0, b) + mul_val;
                //                         activator_input(n, 0, 0, b) = acc_val;
                //                 }
                //         }
                // }

				START_TRACE(); //This will tell moneta where to start tracing

				DUMP_START("weights", (void *) &(weights.data[0]), (void *) &(weights.data[weights.element_count() - 1]), true);

                for ( int b = 0; b < in.size.y; b++ ) {
                        for ( int i = 0; i < in.size.x; i++ ) {
                                for ( int n = 0; n < out.size.x; n++ ) {
                                        double in_val = in(i, b, 0);
                                        double weight_val = weights( i, n, 0 );
                                        double mul_val = in_val * weight_val;
                                        double acc_val = activator_input(n, 0, 0, b) + mul_val;
                                        activator_input(n, 0, 0, b) = acc_val;
                                }
                        }
                }

				DUMP_STOP("weights");

                // finally, apply the activator function.
                for ( unsigned int n = 0; n < activator_input.element_count(); n++ ) {
                        out.data[n] = activator_function( activator_input.data[n] );
                }

                // don't forget to reset the shapes
                in.size = old_size;
                out.size = old_out_size;

				// std::cerr << activator_input.element_count() << std::endl;

        }

			
};

