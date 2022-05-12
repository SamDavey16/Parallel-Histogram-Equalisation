#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

//Implementation explanation: Initially the program takes in a user input for the number of bins to be used in the histogram kernel. The program consists of four kernels: A simple histogram implementation which creates a histogram from the input image, the function is based on an atomic function which negatively impacts its efficiency.
//The second kernel creates a cumulative histogram from the original histogram using scan which is in turn passed into the third kernel (which is based on map) which normalises the cumulative histogram and creates a lookup table from the intensity values.
//The final kernel converts the pixel intensity values from the original image to the intensity values in the lookup table and creates a new image. 
//The C++ file executes the kernels and creates buffers which can be used to pass data to the kernels. The file also queues the kernels in order to execute them. After each kernel is executed the output of the kernel is displayed as well as the execution
//time. The C++ file and kernel file is based on workshop 2 with the histogram and cumulative histogram kernels being based on workshop 3.

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);
	int bins;
	std::cout << "Enter a bin number: " << "\n";
	cin >> bins;

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input"); // Displays the original image

		//a 3x3 convolution mask implementing an averaging filter
		std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - device operations
		typedef int mytype;
		std::vector<mytype> His(bins); // Sets the bin number to the value that the user gave - implementation of variable bin number
		size_t histsize = His.size() * sizeof(mytype); // Histogram size variable
		//device - buffers - Buffers work as ways of communicating data to opencl, below buffers are created to be passed to the kernels
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer histogram_output(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer cumulative_histogram_output(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer Lookup_output(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueFillBuffer(histogram_output, 0, 0, histsize);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_histogram = cl::Kernel(program, "hist_simple"); // This runs the simple histogram kernel which produces a histogram from the image
		kernel_histogram.setArg(0, dev_image_input); // Argument input
		kernel_histogram.setArg(1, histogram_output);
		cl::Event prof_event; // Creates an event to be scheduled
		queue.enqueueNDRangeKernel(kernel_histogram, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event);
		queue.enqueueReadBuffer(histogram_output, CL_TRUE, 0, histsize, &His[0]); // Data to be read

		cout << endl;
		std::cout << "Histogram output: " << His << std::endl; // Displays the histogram values
		std::cout << "Histogram kernel execution time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl; // Displays the execution time
		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		std::vector<mytype> cum_hist(256); // Size of the cumulative histogram

		queue.enqueueFillBuffer(cumulative_histogram_output, 0, 0, histsize);

		cl::Kernel kernel_hist_cum = cl::Kernel(program, "cumulative_histogram"); // Runs the cumulative histogram kernel which is adapted from workshop 3
		kernel_hist_cum.setArg(0, histogram_output);
		kernel_hist_cum.setArg(1, cumulative_histogram_output);

		cl::Event prof_event2;

		queue.enqueueNDRangeKernel(kernel_hist_cum, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &prof_event2);
		queue.enqueueReadBuffer(cumulative_histogram_output, CL_TRUE, 0, histsize, &cum_hist[0]);

		std::cout << "Cumulative histogram output: " << cum_hist << std::endl;
		std::cout << "Cumulative histogram kernel execution time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event2.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event2, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		std::vector<mytype> Lookup(256); // Lookup table size

		queue.enqueueFillBuffer(Lookup_output, 0, 0, histsize);

		cl::Kernel kernel_Lookup = cl::Kernel(program, "normalised"); // Runs the normalised kernel which normalises the cumulative histogram and creates a lookup table
		kernel_Lookup.setArg(0, cumulative_histogram_output);
		kernel_Lookup.setArg(1, Lookup_output);

		cl::Event prof_event3;

		queue.enqueueNDRangeKernel(kernel_Lookup, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &prof_event3);
		queue.enqueueReadBuffer(Lookup_output, CL_TRUE, 0, histsize, &Lookup[0]);

		std::cout << "Look up table output: = " << Lookup << std::endl;
		std::cout << "Look up table kernel execution time [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event3.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event3, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		cl::Kernel kernel_ReProject = cl::Kernel(program, "output"); // Runs the kernel which converts the pixel intensity values from the original image to the intensity values in the lookup table and creates a new image
		kernel_ReProject.setArg(0, dev_image_input);
		kernel_ReProject.setArg(1, Lookup_output);
		kernel_ReProject.setArg(2, image_output);

		cl::Event prof_event4;

		vector<unsigned char> output_buffer(image_input.size());
		queue.enqueueNDRangeKernel(kernel_ReProject, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event4);
		queue.enqueueReadBuffer(image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image,"output");

 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
