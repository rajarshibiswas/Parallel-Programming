/*
 * CSE 5441 : Lab 3
 * Filename : biswas_rajarshi_disposable.h
 * Author   : Rajarshi Biswas (biswas.91@osu.edu)
 *            The Ohio State University.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <assert.h>
#include <time.h>
#include <omp.h>

using namespace std;
using std::chrono::system_clock;

#define SUCCESS		1
#define ERROR		0
#define HORIZONTAL	1
#define VERTICAL 	0

/*
 * Structure to store neighbor’s information.
 */
typedef struct neighbor {
    int id; /* Stores the neighbor’s ID. */
    int contact_distance; /* Stores the contact distance between two neighbors. */
} neighbor;

/*
 * Structure to store properties of a box.
 */
typedef struct box {
    /*
     * position current box on underlying
     * co-ordinated grid.
     */
    int upper_left_y;
    int upper_left_x;

    /* Height, width and perimeter of the current box. */
    int height;
    int width;
    int perimeter;

    /* Current box DSV (temperature). */
    float box_dsv;

    /* The neighbors of the current box. */
    vector<neighbor> left;
    vector<neighbor> right;
    vector<neighbor> top;
    vector<neighbor> bottom;
} box;

/* Structure contains the thread specific data. */
typedef struct thread_data{
	int thread_id; // Thread ID.
	int start; // Starting box number in the grid. 
	int end; // End box number in the grid. 
} thread_data;

/* Global data shared among all the threads. */
int number_of_boxes; // Number of boxes in the grid. 
int number_of_threads; // Number of threads.
int actual_number_of_threads; // Actual number of threads created by the OMP.
float affect_rate; // Affect rate for the computation. 
float epsilon; // Epsilon for the computation. 
float max_dsv; // Maximum DSV.
float min_dsv; // Minimum DSV.
int convergence_iteration; // Total number of iteration required to converge. 
float *updated_DSVs; // Temporary storage to store the updated DSVs.
box *grid; // The entire grid.  

/* Function prototypes. */
int find_contact_distace(int current_box_id, int neighbor_box_id, int direction);
int compute_contact_distance();
int read_input_file();
int inline commit_dsv_update();
int inline check_for_convergence();
int do_stencil_computation();
