/*
 * CSE 5441 : Lab 4 p2
 * Filename : biswas_rajarshi_part2.cu
 * Author   : Rajarshi Biswas (biswas.91@osu.edu)
 *            The Ohio State University.
 */

#include <math.h>
#include <iostream>
#include "lab4_lib/read_bmp.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define THREADS_X 32
#define THREADS_Y 32

/*
 * Kernel function. Runs on GPU.
 * bmp_data 		- Actual image information.
 * new_bmp_data 	- Modified image information.
 * black_cell_count	- Counts the number of black cells.
 * threshold		- Current threshold.
 * wd			- Width of the image. 
 * ht			- Height of the image.  
 */
__global__  void compute_on_device(uint8_t *bmp_data, uint8_t *new_bmp_img, uint32_t *black_cell_count,
    uint32_t threshold, uint32_t wd, uint32_t ht)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t j = (blockIdx.y * blockDim.y) + threadIdx.y;
 	uint32_t index = i * wd + j;
	
	if (index  > wd*ht) {
		return;
		// Check the boundary. 
	}

	if ( (i >= 1 && i < (ht-1)) && (j >= 1 && j < (wd - 1))) {

		float Gx = bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i-1)*wd + (j-1) ]
			+ 2*bmp_data[ (i)*wd + (j+1) ] - 2*bmp_data[ (i)*wd + (j-1) ]
			+ bmp_data[ (i+1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ];
		float Gy = bmp_data[ (i-1)*wd + (j-1) ] + 2*bmp_data[ (i-1)*wd + (j) ]
			+ bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ]
			- 2*bmp_data[ (i+1)*wd + (j) ] - bmp_data[ (i+1)*wd + (j+1) ];
		float mag = sqrt(Gx * Gx + Gy * Gy);

		if (mag > threshold) {
			new_bmp_img[ index ] = 255;
		} else {
			new_bmp_img[index] = 0;
			// Increment the number of black count cell atomically.
			atomicAdd(black_cell_count, 1);
		}
	} else {
		return;
	}

}

/*
 * Wrapper function that calls the kernel function.
 * in_file 	- The input image file.
 * cuda_out_file- The output image file.
 * Returns the threshold value at convergence. 
 */
int cuda_processing(FILE *in_file, FILE *cuda_out_file) {
	bmp_image img1;
	uint8_t *host_bmp_data = (uint8_t *) img1.read_bmp_file(in_file);
	
	//Get image attributes
	uint32_t wd = img1.image_width;
	uint32_t ht = img1.image_height;
	uint32_t num_pixel = img1.num_pixel;
	
	uint8_t* host_new_bmp_img = (uint8_t*) malloc(num_pixel);
	
	// Initialize the device memory.
	uint8_t *device_bmp_data;
	uint8_t *device_new_bmp_img;
	uint32_t *device_black_cell_count;
	
	cudaMalloc((void**) &device_bmp_data, num_pixel);
	cudaMalloc((void**) &device_new_bmp_img, num_pixel);
	cudaMalloc((void **) &device_black_cell_count, sizeof(uint32_t));
	
	// Initialize the array to 0.
	for (int i = 0 ;i < num_pixel; i++) {
		host_new_bmp_img[i] = 0;
	}
	
	// copy it to cuda mem.
	cudaMemcpy(device_bmp_data, host_bmp_data, num_pixel, cudaMemcpyHostToDevice);
	cudaMemcpy(device_new_bmp_img, host_new_bmp_img, num_pixel, cudaMemcpyHostToDevice);
	
	uint32_t threshold = 0;
	uint32_t black_cell_count = 0;
	
	dim3 threadsPerBlock(THREADS_X, THREADS_Y);
        dim3 blocksPerGrid((ht/THREADS_X) + 1, (wd/THREADS_Y) + 1);

	//Convergence loop
	while (black_cell_count < (75 * wd * ht/100))
	{
		black_cell_count = 0;
		cudaMemcpy(device_black_cell_count, &black_cell_count, sizeof(uint32_t),cudaMemcpyHostToDevice);
		threshold += 1;
		// Call cuda kernel.
		compute_on_device <<< blocksPerGrid, threadsPerBlock >>> (device_bmp_data,
		    device_new_bmp_img, device_black_cell_count, threshold, wd, ht);
		cudaMemcpy(host_new_bmp_img, device_new_bmp_img, num_pixel, cudaMemcpyDeviceToHost);
		cudaMemcpy(&black_cell_count, device_black_cell_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	}

	img1.write_bmp_file(cuda_out_file, host_new_bmp_img);
	// Free all the memory. 
	free(host_bmp_data);
	free(host_new_bmp_img);
	//free(host_black_cell_count);
	cudaFree(device_bmp_data);
	cudaFree(device_new_bmp_img);
	cudaFree(device_black_cell_count);
	//cudaFree(device_black_cell_count);
	return threshold;
}

/*
 * Serial function.
 * in_file 	- The input image file.
 * cuda_out_file- The output image file.
 * Returns the threshold value at convergence. 
 */
int serial_processing(FILE *in_file, FILE *serial_out_file) {
	bmp_image img1;
	uint8_t *bmp_data = (uint8_t *) img1.read_bmp_file(in_file);
	//Allocate new output buffer of same size
	uint8_t* new_bmp_img = (uint8_t*)malloc(img1.num_pixel);
	// Initialize the array to 0.
	for (int i = 0 ;i < img1.num_pixel; i++) {
		new_bmp_img[i] = 0;
	}

	//Get image attributes
	uint32_t wd = img1.image_width;
	uint32_t ht = img1.image_height;

	//Convergence loop
	uint32_t threshold = 0;
	uint32_t black_cell_count = 0;

	// Serial version
	while(black_cell_count < (75*wd*ht/100))
	{
		black_cell_count = 0;
		threshold += 1;
		for(int i=1; i < (ht-1); i++)
		{
			for(int j=1; j < (wd-1); j++)
			{
				float Gx = bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i-1)*wd + (j-1) ]
					+ 2*bmp_data[ (i)*wd + (j+1) ] - 2*bmp_data[ (i)*wd + (j-1) ]
					+ bmp_data[ (i+1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ];
				float Gy = bmp_data[ (i-1)*wd + (j-1) ] + 2*bmp_data[ (i-1)*wd + (j) ]
					+ bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ]
					- 2*bmp_data[ (i+1)*wd + (j) ] - bmp_data[ (i+1)*wd + (j+1) ];
				float  mag = sqrt(Gx * Gx + Gy * Gy);

				if(mag > threshold) {
					new_bmp_img[ i * wd + j] = 255;
				} else {
					new_bmp_img[ i * wd + j] = 0;
					black_cell_count++;
				}
			}
		}

	}

	img1.write_bmp_file(serial_out_file, new_bmp_img);
	free(bmp_data);
	free(new_bmp_img);
	return threshold;
}



int main(int argc, char* argv[]) {
	struct timespec start, end;
	FILE *in_file = fopen(argv[1], "rb");
	FILE *serial_out_file = fopen(argv[2], "wb");
	FILE *cuda_out_file = fopen(argv[3], "wb");

	printf("\n**************************************************\n");

	// Serial version
	printf("\n** Serial version **\n\n");
	clock_gettime(CLOCK_REALTIME, &start);
	uint32_t serial_threshold = serial_processing(in_file, serial_out_file);
	clock_gettime(CLOCK_REALTIME, &end);
	double time_taken_serial = ((double)end.tv_sec + 1.0e-9*end.tv_nsec) - 
	    ((double)start.tv_sec + 1.0e-9*start.tv_nsec);
	printf("\nTime taken for serial sobel operation: %.5f sec",time_taken_serial);
	printf("\nThreshold during convergence: %d", serial_threshold );
	
	printf("\n\n** CUDA version **\n\n");
	// Cuda version
	clock_gettime(CLOCK_REALTIME, &start);
	uint32_t cuda_threshold = cuda_processing(in_file, cuda_out_file);
	clock_gettime(CLOCK_REALTIME, &end);
	double time_taken_cuda = ((double)end.tv_sec + 1.0e-9*end.tv_nsec) -
	    ((double)start.tv_sec + 1.0e-9*start.tv_nsec);

	// Print the result of cuda version.
	printf("\nTime taken for CUDA sobel operation: %.5f sec",time_taken_cuda);
	printf("\nThreshold during convergence: %d", cuda_threshold );
	

	printf("\n**************************************************\n\n");
	
	fclose(in_file);
	return 0;
}
