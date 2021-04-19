#include "archlab.hpp"
#include "CNN/canela.hpp"
#include "math.h"
#include <fstream>      // std::fstream
std::fstream trace;

#if(1)
// THese three macros are useful for tracing accesses.  See examples
// below for how to use them.  They are disabled by default, but if
// you change the 0 above to 1, they will turn on.
//
// Turning it on will generate a very large file (named `filename`),
// so you should only run it on small datasets (like mnist).
//
// Once you have the trace file, you can use 
#define DUMP_ACCESS(t,x,y,z,b) do {					\
		trace << t.linearize(x,y,z,b) << " "			\
		      << " "						\
		      << &t.get(x,y,z,b) << " ";			\
	} while(0)

#define END_TRACE_LINE() do {trace << "\n";}while(0)
#define OPEN_TRACE(filename)  std::fstream trace; trace.open (filename, std::fstream::out);


// This one is customized for the stabilization code.  It prints out
// the linear index and address of each tensor element that's
// accessed.
#define DUMP_ACCESSES() do {						\
		DUMP_ACCESS(output, offset_x, offset_y, 0, this_frame); \
		DUMP_ACCESS(images, pixel_x, pixel_y, 0, this_frame);	\
		DUMP_ACCESS(images, shifted_x, shifted_y, 0, previous_frame); \
		END_TRACE_LINE();					\
	} while(0)


#else
// By default, you get these versions, which do nothing.
#define DUMP_ACCESS(t,x,y,z,b)
#define END_TRACE_LINE
#define OPEN_TRACE(filename)
#define DUMP_ACCESSES()
#endif

#define MAX_OFFSET 8

void do_stabilize_baseline(const tensor_t<double> & images, tensor_t<double> & output)
{
	OPEN_TRACE("trace.out");
	// The interesting part starts here.

	// We iterate over each frame and compare it to the previous
	// one (so `this_frame` starts at 1)
	//
	// Frames are identified by their index in the batch.  The
	// index is the `b` dimension of the tensor, which always
	// appears last when we access elements of the tensor.
	for (int this_frame = 1; this_frame < images.size.b; this_frame++) {
		int previous_frame = this_frame - 1;

		// We will shift around the previous frame relative to
		// the current frame and compute the "sum of absolute
		// differences" for each different shift amount.

		// Here we are shifting by up to MAX_OFFSET pixels up, down,
		// left, and right.
		for (int offset_x = 0; offset_x < MAX_OFFSET; offset_x++)  {
			for (int offset_y = 0; offset_y < MAX_OFFSET; offset_y++)  {

				// Iterate over the pixels in the two
				// images.  `pixel_x` and `pixel_y`
				// will be use for the current frame.
				for(int pixel_x = 0; pixel_x < images.size.x; pixel_x++) {
					for(int pixel_y = 0; pixel_y < images.size.y; pixel_y++) {

						int shifted_x = pixel_x + offset_x;  // pixel location in the shifted previous frame
						int shifted_y = pixel_y + offset_y;

						if (shifted_x >= images.size.x || // Bounds check.
						    shifted_y >= images.size.y)
							continue;

						// calculate and accumulate the difference between the images at this shifting amount.
						// We add MAX_OFFSET because the offsets can be negative.
						DUMP_ACCESSES();
						output(offset_x, offset_y, 0, this_frame) += 
							fabs(images(pixel_x, pixel_y, 0, this_frame) -
							     images(shifted_x, shifted_y, 0, previous_frame));
					}
				}
			}
		}
	}

}

void do_stabilize_reorder_pixelxy(const tensor_t<double> & images, tensor_t<double> & output)
{
	OPEN_TRACE("trace.out");
	for (int this_frame = 1; this_frame < images.size.b; this_frame++) {
		int previous_frame = this_frame - 1;

		for (int offset_x = 0; offset_x < MAX_OFFSET; offset_x++)  {
			for (int offset_y = 0; offset_y < MAX_OFFSET; offset_y++)  {

				for(int pixel_y = 0; pixel_y < images.size.y; pixel_y++) {
					for(int pixel_x = 0; pixel_x < images.size.x; pixel_x++) {

						int shifted_x = pixel_x + offset_x; 
						int shifted_y = pixel_y + offset_y;

						if (shifted_x >= images.size.x || 
						    shifted_y >= images.size.y)
							continue;
						DUMP_ACCESSES();
						output(offset_x, offset_y, 0, this_frame) += 
							fabs(images(pixel_x, pixel_y, 0, this_frame) -
							     images(shifted_x, shifted_y, 0, previous_frame));
					}
				}
			}
		}
	}
}

void do_stabilize_innerloop_offsets(const tensor_t<double> & images, tensor_t<double> & output) 
{
	OPEN_TRACE("trace.out");
	for (int this_frame = 1; this_frame < images.size.b; this_frame++) {
		int previous_frame = this_frame - 1;

		for(int pixel_y = 0; pixel_y < images.size.y; pixel_y++) {
			for(int pixel_x = 0; pixel_x < images.size.x; pixel_x++) {

				for (int offset_x = 0; offset_x < MAX_OFFSET; offset_x++)  {
					for (int offset_y = 0; offset_y < MAX_OFFSET; offset_y++)  {
						

						int shifted_x = pixel_x + offset_x; 
						int shifted_y = pixel_y + offset_y;

						if (shifted_x >= images.size.x ||
						    shifted_y >= images.size.y)
							continue;
						DUMP_ACCESSES();
						output(offset_x, offset_y, 0, this_frame) += 
							fabs(images(pixel_x, pixel_y, 0, this_frame) -
							     images(shifted_x, shifted_y, 0, previous_frame));
					}
				}
			}
		}
	}
}

//#define TILE_SIZE 2
void do_stabilize_pretile_y(const tensor_t<double> & images, tensor_t<double> & output, int TILE_SIZE)
{
	OPEN_TRACE("trace.out");
	for (int this_frame = 1; this_frame < images.size.b; this_frame++) {
		int previous_frame = this_frame - 1;

		for (int offset_x = 0; offset_x < MAX_OFFSET; offset_x++)  {
			for (int offset_y = 0; offset_y < MAX_OFFSET; offset_y++)  {
				
				for(int pixel_yy = 0; pixel_yy < images.size.y; pixel_yy += TILE_SIZE) {
					for(int pixel_y = pixel_yy; pixel_y < pixel_yy + TILE_SIZE && pixel_y < images.size.y; pixel_y++) {
						for(int pixel_x = 0; pixel_x < images.size.x; pixel_x++) {
							
							int shifted_x = pixel_x + offset_x; 
							int shifted_y = pixel_y + offset_y;
							
							if (shifted_x >= images.size.x ||
							    shifted_y >= images.size.y)
								continue;
							DUMP_ACCESSES();
							output(offset_x, offset_y, 0, this_frame) += 
								fabs(images(pixel_x, pixel_y, 0, this_frame) -
								     images(shifted_x, shifted_y, 0, previous_frame));

						}
					}
				}
			}
		}
	}
}

void do_stabilize_tile_y_1(const tensor_t<double> & images, tensor_t<double> & output, int TILE_SIZE)
{
	OPEN_TRACE("trace.out");
	for (int this_frame = 1; this_frame < images.size.b; this_frame++) {
		int previous_frame = this_frame - 1;

		for(int pixel_yy = 0; pixel_yy < images.size.y; pixel_yy +=  TILE_SIZE) {

			for (int offset_x = 0; offset_x < MAX_OFFSET; offset_x++)  {
				for (int offset_y = 0; offset_y < MAX_OFFSET; offset_y++)  {
	
					for(int pixel_y = pixel_yy; pixel_y < pixel_yy + TILE_SIZE && pixel_y < images.size.y; pixel_y++) {
						for(int pixel_x = 0; pixel_x < images.size.x; pixel_x++) {
							
							int shifted_x = pixel_x + offset_x; 
							int shifted_y = pixel_y + offset_y;
							
							if (shifted_x >= images.size.x ||
							    shifted_y >= images.size.y)
								continue;
							DUMP_ACCESSES();
							output(offset_x, offset_y, 0, this_frame) += 
								fabs(images(pixel_x, pixel_y, 0, this_frame) -
								     images(shifted_x, shifted_y, 0, previous_frame));
						}
					}
				}
			}
		}
	}
}



 
void stabilize(const std::string & version, const dataset_t & test, int frames)
{
        // Declare a 4D tensor to hold frames video frames
	tensor_t<double> batch_data(tdsize(test.data_size.x, test.data_size.y, test.data_size.z, frames));

	int batch_index = 0;
	// Copy the frames video frames into batch_data.  This a bit
	// strange because, the dataset we want might not have enough
	// frames.  So, we copy the dataset over and over until we
	// have enough data.
	while(batch_index < frames) {
		for (auto& t : test.test_cases ) {
			for (int x = 0; x < t.data.size.x; x += 1)
				for (int y = 0; y < t.data.size.y; y += 1){
					int z= 0; // `test` might hold color
					// images so the z dimension
					// would be 3 (RGB).  We are
					// only doing gray scale, so
					// we just take layer 0 (R
					// for RGB images and gray
					// for grayscale).
					batch_data(x, y, z, batch_index) = t.data(x, y, z);
				}
			batch_index += 1;
			if (batch_index >= frames) {
				break;
			}
		}
	}

	tensor_t<double>::diff_prints_deltas = true;
 
	// Tensor to hold the outputs
	tensor_t<double> output(MAX_OFFSET,MAX_OFFSET,1,batch_data.size.b);
	// zero out output
	trace.open ("trace.out", std::fstream::out);
	
	bool verbose = false;
	
#define CHECK_AND_RESET()						\
	do {								\
	if (output != reference) {					\
	if (verbose) {							\
	std::cout << output <<"\n";					\
	std::cout << diff(output, reference)<< "\n";			\
}									\
	assert(0);							\
}									\
	output.clear();							\
} while(0)
	

	output.clear();

	if (version == "all"
	    || version == "baseline"
	    || version == "demo"
		) 
	{
		{
			ArchLabTimer timer; // create it.
			pristine_machine();
			theDataCollector->disable_prefetcher();
			set_cpu_clock_frequency(1900);
			timer.attr("function", "do_stabilize_baseline").go();
			do_stabilize_baseline(batch_data, output);
		}
	}
	tensor_t<double> reference = output;
	CHECK_AND_RESET();
	
	if (version == "all"
	    || version == "reorder_pixelxy"
	    || version == "demo"
		) 
	{
		{
		ArchLabTimer timer; // create it.
		pristine_machine();
		theDataCollector->disable_prefetcher();
		set_cpu_clock_frequency(1900);
		timer.attr("function", "do_stabilize_reorder_pixelxy").go();
		do_stabilize_reorder_pixelxy(batch_data, output);
		}
		CHECK_AND_RESET();
	}

	if (version == "all"
	    || version == "innerloop_offsets"
		) 
	{
		{
		ArchLabTimer timer; // create it.
		pristine_machine();
		theDataCollector->disable_prefetcher();
		set_cpu_clock_frequency(1900);
		timer.attr("function", "do_stabilize_innerloop_offsets").go();
		do_stabilize_innerloop_offsets(batch_data, output);
		}
		CHECK_AND_RESET();
	}

	if (version == "all"
	    || version == "pretile_y"
		) 
	{
		{
		ArchLabTimer timer; // create it.
		pristine_machine();
		theDataCollector->disable_prefetcher();
		set_cpu_clock_frequency(1900);
		timer.attr("function", "do_stabilize_pretile_y").go();
		do_stabilize_pretile_y(batch_data, output, 2);
		}
		CHECK_AND_RESET();
	}

	if (version == "all"
	    || version == "tile_y_1"
	    || version == "demo"
		) 
	{
		{
		ArchLabTimer timer; // create it.
		pristine_machine();
		theDataCollector->disable_prefetcher();
		set_cpu_clock_frequency(1900);
		timer.attr("function", "do_stabilize_tile_y_1").go();
		do_stabilize_tile_y_1(batch_data, output, 2);
		}
		CHECK_AND_RESET();
	}


#if (0)
	{
		pristine_machine();
		theDataCollector->disable_prefetcher();
		set_cpu_clock_frequency(1900);
		do_stabilize_tile_y_1(batch_data, output);
	}

	CHECK_AND_RESET();
#endif
}

