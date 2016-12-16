/*
 * CSE 5441 : Lab 1
 * Rajarshi Biswas
 * biswas.91@osu.edu
 * The Ohio State University.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <assert.h>
#include <time.h>

using namespace std;
using std::chrono::system_clock;

#define SUCCESS 1
#define ERROR 0
#define HORIZONTAL 1
#define VERTICAL 0

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

int find_contact_distace(box grid[], int current_box_id,
	int neighbor_box_id, int direction);
int compute_contact_distance(box grid[], int number_of_boxes);
int read_input_file(box *grid, int number_of_boxes);
int inline commit_dsv_update(box grid[], float updated_DSVs[], int number_of_boxes,
float *max, float *min);
int inline check_for_convergence(box grid[], int number_of_boxes,
	float epsilon, float max, float min);
int do_stencil_computation(box grid[], int number_of_boxes,
    float affect_rate, float epsilon,
    int *convergence_iteration, float *max, float *min);
