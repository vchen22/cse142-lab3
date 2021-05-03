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

#define DUMP_TENSOR_START(TAG, T) DUMP_START(TAG, (void *) &((T).data[0]), (void *) &((T).data[(T).element_count() - 1]), true)
#define DUMP_TENSOR_STOP(TAG) DUMP_STOP(TAG)

class opt_fc_layer_t : public fc_layer_t
{
public:
	opt_fc_layer_t( tdsize in_size, int out_size ) : fc_layer_t(in_size, out_size) {

	}

		void activate( tensor_t<double>& in ) {

		START_TRACE();
		DUMP_START_ALL("all", true);
		DUMP_TENSOR_START("weights", weights);
		DUMP_TENSOR_START("activator_input", activator_input);
		DUMP_TENSOR_START("out", out);
		DUMP_TENSOR_START("in", in);
		copy_input(in);

		tdsize old_size = in.size;
		tdsize old_out_size = out.size;

		// cast to correct shape
		in.size.x = old_size.x * old_size.y * old_size.z;
		in.size.y = old_size.b;
		in.size.z = 1;
		in.size.b = 1;

		out.size.x = old_out_size.x * old_out_size.y * old_out_size.z;
		out.size.y = old_out_size.b;
		out.size.z = 1;
		out.size.b = 1;

		for ( int b = 0; b < activator_input.size.b; b++) {
			for ( int n = 0; n < activator_input.size.x; n++ ) {
				activator_input(n, 0, 0, b) = 0;
			}
		}

		// for ( int b = 0; b < in.size.y; b++ ) {
		// 	for ( int i = 0; i < in.size.x; i++ ) {
		// 		for ( int n = 0; n < out.size.x; n++ ) {
		// 			double in_val = in(i, b, 0);
		// 			double weight_val = weights( i, n, 0 );
		// 			double mul_val = in_val * weight_val;
		// 			double acc_val = activator_input(n, 0, 0, b) + mul_val;
		// 			activator_input(n, 0, 0, b) = acc_val;
		// 		}
		// 	}
		// }

		for ( int b = 0; b < in.size.y; b++ ) {
			for ( int n = 0; n < out.size.x; n++ ) {
				for ( int i = 0; i < in.size.x; i++ ) {
					double in_val = in(i, b, 0);
					double weight_val = weights( i, n, 0 );
					double mul_val = in_val * weight_val;
					double acc_val = activator_input(n, 0, 0, b) + mul_val;
					activator_input(n, 0, 0, b) = acc_val;
				}
			}
		}

                // #define TILE_SIZE 16
                // for(int nn = 0; nn < out.size.x; nn+=TILE_SIZE){
		// 	for ( int b = 0; b < in.size.y; b++ ) {
		// 		for ( int n = nn; n < nn + TILE_SIZE && n < out.size.x; n++ ) {
                //                         for ( int i = 0; i < in.size.x; i++ ) {
                //                                 double in_val = in(i, b, 0);
                //                                 double weight_val = weights( i, n, 0 );
                //                                 double mul_val = in_val * weight_val;
                //                                 double acc_val = activator_input(n, 0, 0, b) + mul_val;
                //                                 activator_input(n, 0, 0, b) = acc_val;
                //                         }
                //                 }
                //         }
		// }

		// finally, apply the activator function.
		for ( unsigned int n = 0; n < activator_input.element_count(); n++ ) {
			out.data[n] = activator_function( activator_input.data[n] );
		}

		// don't forget to reset the shapes
		in.size = old_size;
		out.size = old_out_size;
	}


        void calc_grads( const tensor_t<double>& grad_next_layer ) {

                memset( grads_out.data, 0, grads_out.size.x * grads_out.size.y * grads_out.size.z * sizeof( double ) );

                // Using the notation from activate():
                //
                // We do two things in the loop below: 1) compute the
                // gradient (grad.grad). 2) propagate the error to the
                // previous error.

                // The gradient:

                // We are calculating the derivative of F with respect
                // to w.  The derivative, F', is a vector, so it's a
                // gradient (stored in `gradients`)
                //
                // F(x, w)  = L(f(x,w)) // from above
                // F'(x, w) = L'(f(x,w)) * f'(x,w)
                //
                // To compute index, i, of the F', we calculate the
                // derivative with respect to w[i].
                //
                // Note that
                //
                // f(x,w) = x[0]*w[0] + x[1]*w[1] + ... + x[n]*w[n]
                //
                // So the derivative wrt w[i] is 
                //
                // df(x,w)/dw[i] = x[i]
                //
                // L'(x) = activator_derivative(x) // look at the code, if you're curious.

                // The inner loop is responsible for back-propagating
                // the error.  Intuitively, we are assigning 'blame'
                // for the error in this layer's output to the
                // elements of the input tensor.
                //
                // The amount of blame we assign to each input for the
                // error in a particular output is proportional to
                // that input's weight for that output.  If the weight
                // for an input is large, it had a large impact on the
                // output, so it more responsible for the resulting
                // error.

                // The errors attributed to each input is the sum of
                // the error it contributed across all the outputs.

                grads_out.size.x = grads_out.size.x * grads_out.size.y * grads_out.size.z;
                grads_out.size.y = 1;
                grads_out.size.z = 1;

                for ( int b = 0; b < out.size.b; b++ ) {
                        for ( int n = 0; n < activator_input.size.x; n++ ){
                                // In `activate()` we saved the value of
                                // f(x,w) as `activator_input`, so we are
                                // reusing it here to compute L'(f(x,w))
                                double ad = activator_derivative( activator_input(n, 0, 0, b) );
                                //std::cout << ad;
                                double ng = grad_next_layer(n, 0, 0, b);
                                //std::cout << ng;
                                act_grad(n, 0, 0, b) = ad * ng;
                        }
                }

                // We are calculating how much each input
                // contributed to the error.  That
                // contribution is proportional to the
                // weights.

                // for ( int b = 0; b < out.size.b; b++ ) {
                //         for ( int i = 0; i < grads_out.size.x; i++ ) {
                //                 for ( int n = 0; n < out.size.x; n++ ) {
                //                         grads_out(i, 0, 0, b) += act_grad(n, 0, 0, b) * weights( i, n, 0);
                //                 }
                //         }
                // }

                for ( int b = 0; b < out.size.b; b++ ) {
                        for ( int n = 0; n < out.size.x; n++ ) {
                                double calc_act_grad = act_grad(n,0,0,b);
                                for ( int i = 0; i < grads_out.size.x; i++ ) {
                                        grads_out(i, 0, 0, b) += calc_act_grad * weights( i, n, 0);
                                }
                        }
                }

                // #define TILE_SIZE 16
                // for(int nn = 0; nn < out.size.x; nn+=TILE_SIZE){
		// 	for ( int b = 0; b < out.size.b; b++ ) {
		// 		for ( int n = nn; n < nn + TILE_SIZE && n < out.size.x; n++ ) {
                //                         for ( int i = 0; i < grads_out.size.x; i++ ) {
                //                                 grads_out(i, 0, 0, b) += act_grad(n, 0, 0, b) * weights( i, n, 0);
                //                         }
                //                 }
                //         }
		// }

                grads_out.size = in.size;
        }

        void fix_weights() {
                // Here, we are updating the weights.  The amount we
                // change the input primarily depends on the gradient
                // and the input value.  We use gradient decent, which
                // means we follow the gradient downward to minimize
                // error.
                //
                // Recall that during back propagation, the input the
                // layer is the error and the derivatives are with
                // respect to the weights.  This means that the
                // gradient points in the direction we should move the
                // weights to reduce the error.
                //
                // We calculated the gradientt in calc_grads(), and
                // proportional to the error (i.e., grad_next_layer)
                // and the derivative of the activator function.  This
                // means that larger errors or steeper slopes causes
                // bigger changes in the weights.
                //
                // The basic update rule is
                //
                // w_new = w - gradient * input
                //
                // This update rule is too aggressive, however, so we
                // add a learning rate, u:
                //
                // w_new = w - gradient * input * u
                //
                // There is a also problem that can arise when the
                // gradient get small: progress toward the minimum can
                // slow.  So we also have a "momentum" term, M:
                //
                // t = gradient + old_gradient*momentum
                // w_new = w - u * input * t
                //
                // Finally, to smooth out the changes in gradient, we
                // add a 'decay' term governed by a decay, D:
                //
                // t = gradient + old_gradient*momentum
                // w_new = w - (u * input * t + D * w)
                //
                // All this complication lives in update_weight()
                // 
                // Since the above needs old_gradient, the gradient
                // tensor has the old and new gradient values in it.
                // update_gradient() updates the old gradient with the
                // new value.

                tdsize old_in_size = in.size;
                in.size.x = in.size.x * in.size.y * in.size.z;
                in.size.y = 1;
                in.size.z = 1;

                // for ( int b = 0; b < out.size.b; b++ ) {
                // //{ int b = 1;
                //         for ( int n = 0; n < weights.size.y; n++ ) {
                //                 for ( int i = 0; i < weights.size.x; i++ ) {
                //                         double& w = weights( i, n, 0 );
                //                         double m = (act_grad(n, 0, 0, b) + old_act_grad(n, 0, 0, b) * MOMENTUM);
                //                         double g_weight = w - (LEARNING_RATE * m * in(i, 0, 0, b) + LEARNING_RATE * WEIGHT_DECAY * w);
                //                         w = g_weight;
                //                 }
                //                 old_act_grad(n, 0, 0, b) = act_grad(n, 0, 0, b) + old_act_grad(n, 0, 0, b) * MOMENTUM;
                //         }
                // }

                #define TILE_SIZE 4

                for ( int nn = 0; nn < weights.size.x; nn+=TILE_SIZE) {
                        for ( int b = 0; b < out.size.b; b++ ) {
                        //{ int b = 1;
                                for ( int n = nn; n < nn + TILE_SIZE && n < weights.size.y; n++ ) {
                                        for ( int i = 0; i < weights.size.x; i++ ) {
                                                double& w = weights( i, n, 0 );
                                                double m = (act_grad(n, 0, 0, b) + old_act_grad(n, 0, 0, b) * MOMENTUM);
                                                double g_weight = w - (LEARNING_RATE * m * in(i, 0, 0, b) + LEARNING_RATE * WEIGHT_DECAY * w);
                                                w = g_weight;
                                        }
                                        old_act_grad(n, 0, 0, b) = act_grad(n, 0, 0, b) + old_act_grad(n, 0, 0, b) * MOMENTUM;
                                }
                        }
                }

                in.size = old_in_size;
        }
			
};

