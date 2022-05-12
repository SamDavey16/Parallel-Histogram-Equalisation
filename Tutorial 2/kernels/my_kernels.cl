kernel void hist_simple(global const uchar* A, global int* H) {
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

kernel void cumulative_histogram(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N && id < N; i++)
		atomic_add(&B[i], A[id]);
}

kernel void normalised(global int* cum_hist, global int* lookup) {
	int id = get_global_id(0); // Current value
	int N = get_global_size(0); // Size of the cumulative histogram
	double x = 255; // Max value
	lookup[id] = x / cum_hist[N - 1] * cum_hist[id]; //Creates a lookup table from the normalised cumulative histogram. Method adapted from: https://www.itl.nist.gov/div898/handbook/eda/section3/histogra.htm and https://math.stackexchange.com/questions/1154526/histogram-normalization
}

kernel void output(global uchar* A, global int* lookup, global uchar* B) {
	int id = get_global_id(0);
	B[id] = lookup[A[id]]; // Converts the intensity values from the original image to the values from the normalised cumlative histogram
}