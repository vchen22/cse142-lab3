#include "archlab.hpp"
#include <cstdlib>
#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "CNN/canela.hpp"
#include "math.h"

using namespace std;

void stabilize(const std::string & versin, const dataset_t & dataset, int frames);

int main(int argc, char *argv[])
{
	std::vector<std::string> dataset_s;
	std::vector<std::string> default_set;
	default_set.push_back("mnist");
	uint32_t frames;
	archlab_add_option<uint32_t>("frames",  frames   , 64  ,  "images to process");
	std::string version;
	archlab_add_option<std::vector<std::string> >("dataset",
						      dataset_s,
						      default_set,
						      "mnist",
						      "Which dataset to use: 'mnist', 'emnist', 'cifar10', 'cifar100', or 'imagenet'. "
						      "Pass it multiple times to run multiple datasets.");
	archlab_add_option<std::string>("impl",
					version,
					"baseline",
					"baseline",
					"Which version to run");
	archlab_parse_cmd_line(&argc, argv);

	for(auto & ds: dataset_s) {
		std::cout << "Running " << ds << "\n";
		
		dataset_t *test = new dataset_t;
	
		if (ds == "mnist") {
			*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/mnist-test.dataset", 64);
		} else if (ds == "emnist") {
			*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/emnist-byclass-test.dataset", 64);
		} else if (ds == "imagenet") {
			*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/imagenet/imagenet.dataset", 64);
		} else {
			std::cerr << "unknown (Or incompatible) data set: " << ds << "\n";
			exit(1);
		}
		{
			stabilize(version, *test, frames);
		}

	}
	
	archlab_write_stats();
	return 0;
}

