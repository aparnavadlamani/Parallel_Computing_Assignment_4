#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>

int main()
{
	// generate 1024 random numbers serially
	thrust::host_vector<int> h_vec(1 << 10);
	std::generate(h_vec.begin(), h_vec.end(), rand);
	std::cout<<"Actual Array: ";
	for(int i=0;i<h_vec.size();i++)
	{
		std::cout<<h_vec[i]<<" ";	
	}
	std::cout<<std::endl;
	// transfer data to the device
	thrust::device_vector<int> d_vec = h_vec;

	// sort data on the device
	thrust::sort(d_vec.begin(), d_vec.end());

	// transfer data back to host
	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

	std::cout<<"Sorted Array: ";
	for(int i=0;i<h_vec.size();i++)
	{
		std::cout<<h_vec[i]<<" ";	
	}
	std::cout<<std::endl;
	return 0;
}