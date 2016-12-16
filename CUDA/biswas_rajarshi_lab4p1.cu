/*
 * CSE 5441 : Lab 4 part1
 * Filename : biswas_rajarshi_part1.cu
 * Author   : Rajarshi Biswas (biswas.91@osu.edu)
 *            The Ohio State University.
 */

#include <iostream>
#include <cstdlib>
#include <time.h>
#include <stdio.h>
using namespace std;

#define SIZE 1024
#define THREADS_X 32
#define THREADS_Y 32

/*
 * The kernel function. Runs on the device.
 * d_A  - Source matrix.
 * d_B  - Source matrix.
 * d_C  - Destination matrix.
 * dim  - Dimension.
 */
__global__ void multiply_on_device (float *d_A, float* d_B, float *d_C, int dim) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index =  i * dim + j;
	float sum = 0;

	for (int k = 0; k < dim; k++) {
		sum += d_A[i * dim+ k ] * d_B[ k * dim + j ];
	}

	d_C[index] = sum; 
}

/* 
 * Runs on the host.
 * h_A - Source matrix.
 * h_B - Source matrix.
 * h_C - Destination matrix.
 */
void multiply_on_host(float *h_A, float *h_B, float *h_C) {
	
	float sum = 0;
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			for (int k = 0; k < SIZE; k++) {
				sum+= h_A[ i*SIZE+ k ] * h_B[ k*SIZE+j ];
			}
			h_C[ i*SIZE+j ] = sum;
			sum = 0;
		}
	}
}

/*
 * Random float number generator between two numbers.
 * a - The lower bound.
 * b - The upper bound.
 */
float RandomFloat(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

int main() {
	struct timespec start, end;
	int number_of_elements = SIZE * SIZE; 
	size_t memSize = SIZE * SIZE * sizeof(float);

	// Initialize the host memory.
	float* h_A = new float[number_of_elements];
	float* h_B = new float[number_of_elements];
	// Stores the result of serial computation.
	float* h_C1 = new float[number_of_elements];
	// Stores the result of CUDA computation.
	float* h_C2 = new float[number_of_elements];

	// Initialize the matrix.
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			h_A[i * SIZE + j] =  RandomFloat(1.0, 2.0);
		}
	}
	
	// Compute the transpose. 
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			h_B[j * SIZE + i] = h_A[i * SIZE + j];
		}
	}

	cout << "\n**************************************************\n";
	// Call serial version.
	clock_gettime(CLOCK_REALTIME,& start);
	multiply_on_host(h_A, h_B, h_C1);
	clock_gettime(CLOCK_REALTIME,& end);
	double time_taken_serial = ((double)end.tv_sec + 1.0e-9*end.tv_nsec) -
	    ((double)start.tv_sec + 1.0e-9*start.tv_nsec);

	// Initialize the device memory.
	float* d_A;
	float* d_B;
	float* d_C;
	cudaMalloc( (void**) &d_A, memSize);
	cudaMalloc( (void**) &d_B, memSize);
	cudaMalloc( (void**) &d_C, memSize);
	dim3 threadsPerBlock(THREADS_X, THREADS_Y);
	dim3 blocksPerGrid(SIZE/THREADS_X, SIZE/THREADS_Y);

	// call the cuda version.
	clock_gettime(CLOCK_REALTIME,& start);
	// Copy host to device.
	cudaMemcpy(d_A, h_A, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, memSize, cudaMemcpyHostToDevice);
	multiply_on_device <<< blocksPerGrid, threadsPerBlock >>> (d_A, d_B, d_C, SIZE);
	// Copy back the result from device to host memory. 
	cudaMemcpy(h_C2, d_C, memSize, cudaMemcpyDeviceToHost);
	clock_gettime(CLOCK_REALTIME,& end);
	
	double time_taken_cuda = ((double)end.tv_sec + 1.0e-9*end.tv_nsec) -
	    ((double)start.tv_sec + 1.0e-9*start.tv_nsec);

	// Print the result
	cout << "Size of the matrix: "<< SIZE << " x " << SIZE <<"\n"; 
	cout << "Time taken by the serial version: " << time_taken_serial << " seconds\n";
	cout << "Time taken by the CUDA version: " <<  time_taken_cuda << " seconds\n";
	cout << "\n**************************************************\n";
	
	// Free host memory.
	free(h_A);
	free(h_B);
	free(h_C1);
	free(h_C2);
	
	// Free device memory. 
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	return 0;
} 
