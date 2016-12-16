/*
 * CSE 5441 : Lab 5
 * Filename : biswas_rajarshi_lab5.c
 * Author   : Rajarshi Biswas (biswas.91@osu.edu)
 *            The Ohio State University.
 */
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include "lab5_lib/read_bmp.h"
#include <stdlib.h>
#include <time.h>

#define MASTER	0
#define NUM_OF_THREADS 32

/*
 * Serial function.
 * in_file 	- The input image file.
 * Returns the threshold value at convergence. 
 */
int serial_processing(FILE *in_file) {
	int i,j;
	uint8_t *bmp_data = (uint8_t *) read_bmp_file(in_file);
	//Allocate new output buffer of same size
	uint8_t* new_bmp_img = (uint8_t*)malloc(get_num_pixel());
	// Initialize the array to 0.
	for (i = 0 ;i < get_num_pixel(); i++) {
		new_bmp_img[i] = 0;
	}

	//Get image attributes
	uint32_t wd = get_image_width();
	uint32_t ht = get_image_height();

	//Convergence loop
	uint32_t threshold = 0;
	uint32_t black_cell_count = 0;

	// Serial version
	while (black_cell_count < (75 * wd * ht/ 100))
	{
		black_cell_count = 0;
		threshold += 1;
		for (i=1; i < (ht-1); i++)
		{
			for (j=1; j < (wd-1); j++)
			{
				float Gx = bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i-1)*wd + (j-1) ]
					+ 2*bmp_data[ (i)*wd + (j+1) ] - 2*bmp_data[ (i)*wd + (j-1) ]
					+ bmp_data[ (i+1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ];
				float Gy = bmp_data[ (i-1)*wd + (j-1) ] + 2*bmp_data[ (i-1)*wd + (j) ]
					+ bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ]
					- 2*bmp_data[ (i+1)*wd + (j) ] - bmp_data[ (i+1)*wd + (j+1) ];
				float  mag = sqrt(Gx * Gx + Gy * Gy);

				if (mag > threshold) {
					new_bmp_img[ i * wd + j] = 255;
				} else {
					new_bmp_img[ i * wd + j] = 0;
					black_cell_count++;
				}
			}
		}

	}

	free(bmp_data);
	free(new_bmp_img);
	return threshold;
}



int main(int argc, char* argv[]) {
	struct timespec start, end;
	FILE *in_file = fopen(argv[1], "rb");
	FILE *out_file = fopen(argv[2], "wb");
	int rank;
	int workers;
	MPI_Status status;
	uint8_t *bmp_data, *new_bmp_img;
	uint32_t wd, ht, threshold; unsigned int  black_cell_count, global_black_cell_count;
	int i, j, source;
	int type = 1;
	uint32_t avg_rows, extra_rows, rows, start_row, end_row; 

	MPI_Init(NULL, NULL);	
	MPI_Comm_size(MPI_COMM_WORLD, &workers);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// Serial version Master only do.
	if (rank == MASTER) {
		printf("\n**************************************************");
		printf("\n\n** Serial version **\n\n");
		clock_gettime(CLOCK_REALTIME, &start);
		uint32_t serial_threshold = serial_processing(in_file);
		clock_gettime(CLOCK_REALTIME, &end);
		double time_taken_serial = ((double)end.tv_sec + 1.0e-9*end.tv_nsec) - 
			((double)start.tv_sec + 1.0e-9*start.tv_nsec);
		printf("\nTime taken for serial sobel operation: %.5f sec",time_taken_serial);
		printf("\nThreshold during convergence: %d", serial_threshold );
	}
	
	// Barrier Let master finish the Serial version.
	if (rank == MASTER) {	
		printf("\n\n** Parallel version **\n\n");
		printf("%d workers are reading the image\n", workers);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	// Parallel version starts.
	// All the processes reads this.
	bmp_data = (uint8_t *) read_bmp_file(in_file);
	// memory for the processed image.
	new_bmp_img = (uint8_t*) malloc(get_num_pixel());
	for (i = 0; i < get_num_pixel(); i++) {
		new_bmp_img[i] = 0;
	}
	wd = get_image_width();
	ht = get_image_height();
	threshold = 0;
	black_cell_count = 0;
	global_black_cell_count = 0;

	// Let everyone finish reading.
	MPI_Barrier(MPI_COMM_WORLD);	
	

	if (rank == MASTER) {
		clock_gettime(CLOCK_REALTIME, &start);
	}

	// Load distribution.
	avg_rows = ht / workers;
	extra_rows = ht % workers;
	rows  = (rank == workers-1) ? avg_rows + extra_rows : avg_rows;
	start_row = rank * avg_rows;
	end_row = start_row + rows -1;
	float Gx, Gy, mag;
	
	while (global_black_cell_count < (75 * wd * ht/ 100)) {
		black_cell_count = 0;
		MPI_Barrier(MPI_COMM_WORLD);
		threshold += 1;
		#pragma omp parallel num_threads(NUM_OF_THREADS)
		#pragma omp for reduction(+:black_cell_count)
		for (i = start_row; i <=  (end_row); i++)
		{
			if (i == 0 || i == ht-1)
				continue;
			for (j = 1; j < (wd-1); j++)
			{
				Gx = bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i-1)*wd + (j-1) ]
					+ 2*bmp_data[ (i)*wd + (j+1) ] - 2*bmp_data[ (i)*wd + (j-1) ]
					+ bmp_data[ (i+1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ];
				Gy = bmp_data[ (i-1)*wd + (j-1) ] + 2*bmp_data[ (i-1)*wd + (j) ]
					+ bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ]
					- 2*bmp_data[ (i+1)*wd + (j) ] - bmp_data[ (i+1)*wd + (j+1) ];
				mag = sqrt(Gx * Gx + Gy * Gy);
				if (mag > threshold) {
					new_bmp_img[ i * wd + j] = 255;
				} else {
					new_bmp_img[ i * wd + j] = 0;
					black_cell_count++;
				}
			}
		}
		#pragma omp barrier
		// ALL Reduce the black cell count.
		MPI_Allreduce(&black_cell_count, &global_black_cell_count, 1, MPI_UNSIGNED, MPI_SUM,
				MPI_COMM_WORLD);
	}
	
	if (rank != MASTER) {// Send to MASTER
		MPI_Send(&new_bmp_img[start_row * wd], (rows)*wd, MPI_UNSIGNED_CHAR, MASTER,
			type, MPI_COMM_WORLD);
	} else if (rank == MASTER){
		for (source = 1 ; source < workers; source++) {
			rows  = (source== workers-1) ? avg_rows + extra_rows : avg_rows;
			start_row = source * avg_rows;
			// Gather image pixels from slaves.
			MPI_Recv(&new_bmp_img[start_row * wd],  (rows)*wd, MPI_UNSIGNED_CHAR, source, type, 
				MPI_COMM_WORLD, &status);
		}
		write_bmp_file(out_file, new_bmp_img);
		clock_gettime(CLOCK_REALTIME, &end);
		double time_taken = ((double)end.tv_sec + 1.0e-9*end.tv_nsec) - 
			((double)start.tv_sec + 1.0e-9*start.tv_nsec);
		printf("\nTime taken for parallel (MPI+OMP) sobel operation: %.5f sec",time_taken);
		printf("\nThreshold during convergence: %d", threshold );
		printf("\n\n**************************************************\n\n");
	}
	
	fclose(in_file);
	fclose(out_file);
	free(bmp_data);
	free(new_bmp_img);
	MPI_Finalize();
	return 0;
}
