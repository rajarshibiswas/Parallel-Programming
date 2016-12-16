/*
 * CSE 5441 : Lab 2
 * Filename : biswas_rajarshi_disposable.cc
 * Author   : Rajarshi Biswas (biswas.91@osu.edu)
 *            The Ohio State University.
 */

#include "biswas_rajarshi_disposable.h"
#include <pthread.h>

/*
 * Finds the contact distance between two boxes.
 * current_box_id	- The ID of the current box.
 * neighbor_box_id	- The ID of the neighbor box.
 * direction		- The direction of the neighbor box in respect with the
 *			  current box. Either vertical or horizontal.
 *
 * Returns the contact distance between the two boxes.
 */
int find_contact_distace(int current_box_id, int neighbor_box_id, int direction) {

	/* Box properties and the contact distance. */
	int contact_distance;
	int current_box_left;
	int current_box_right;
	int current_box_top;
	int current_box_bottom;
	int neighbor_box_left;
	int neighbor_box_right;
	int neighbor_box_top;
	int neighbor_box_bottom;

	/* Neighbors can be in horizontal and vertical direction. */
	switch (direction) {
		case HORIZONTAL:
			current_box_left = grid[current_box_id].upper_left_x;
			current_box_right = grid[current_box_id].upper_left_x +
				grid[current_box_id].width;

			neighbor_box_left = grid[neighbor_box_id].upper_left_x;
			neighbor_box_right = grid[neighbor_box_id].upper_left_x +
				grid[neighbor_box_id].width;

			contact_distance = min(current_box_right, neighbor_box_right) -
				max(current_box_left, neighbor_box_left);
			break;
		case VERTICAL:
			current_box_top = grid[current_box_id].upper_left_y;
			current_box_bottom = grid[current_box_id].upper_left_y +
				grid[current_box_id].height;

			neighbor_box_top = grid[neighbor_box_id].upper_left_y;
			neighbor_box_bottom = grid[neighbor_box_id].upper_left_y +
				grid[neighbor_box_id].height;
			contact_distance = min(current_box_bottom, neighbor_box_bottom) -
				max(current_box_top, neighbor_box_top);
			break;
		default:
		  	/* Error! */
			assert(0);
			break;
	}

	return (contact_distance);
}

/*
 * Computes the contact distances among each neighbor and the current box.
 *
 * Returns SUCCESS in case of success.
 */
int compute_contact_distance() {

	/* Iterator to go through the neighbors of a box. */
	vector<neighbor>::iterator iter;

	// Compute for all the grid.
	for (int current_box_id = 0; current_box_id < number_of_boxes; current_box_id++) {
		// Compute the contact distances of the top neighbours.
		for (iter = grid[current_box_id].top.begin(); iter != grid[current_box_id].top.end(); iter++) {
			iter->contact_distance = find_contact_distace(current_box_id, iter->id, HORIZONTAL);
		}
		// Compute the contact distances of the bottom neighbours.
		for (iter = grid[current_box_id].bottom.begin(); iter != grid[current_box_id].bottom.end(); iter++) {
			iter->contact_distance = find_contact_distace(current_box_id, iter->id, HORIZONTAL);
		}
		// Compute the contact distances of the left neighbors.
		for (iter = grid[current_box_id].left.begin(); iter != grid[current_box_id].left.end(); iter++) {
			iter->contact_distance = find_contact_distace(current_box_id, iter->id, VERTICAL);
		}
		// Compute the contact distances of the right neighbors.
		for (iter = grid[current_box_id].right.begin(); iter != grid[current_box_id].right.end(); iter++) {
			iter->contact_distance = find_contact_distace(current_box_id, iter->id, VERTICAL);
		}
	}

	return (SUCCESS);
}

/*
 * Reads the input and prepare the data structures.
 *
 * Returns SUCCESS in case of successful reading otherwise ERROR.
 */
int read_input_file() {

	int current_box_id;
	int num_grid_rows;
	int num_grid_cols;
	int neighbors;
	int neighbor_box_id;
	int last_line;
	int result;

	cin >> num_grid_rows;
	cin >> num_grid_cols;

	for(int i = 0; i < number_of_boxes; i++) {
		cin >> current_box_id;
		cin >> grid[current_box_id].upper_left_y;
		cin >> grid[current_box_id].upper_left_x;
		cin >> grid[current_box_id].height;
		cin >> grid[current_box_id].width;
		/* compute and store the perimeter for later use. */
		grid[current_box_id].perimeter = 2 *(grid[current_box_id].height +
			grid[current_box_id].width);

		// Scan the neighbors.
		cin >> neighbors;
		neighbor temp;
		for(int j = 0; j < neighbors; j++) {
			cin >> neighbor_box_id;
			temp.id = neighbor_box_id;
			grid[current_box_id].top.push_back(temp);
		}
		cin >> neighbors;
		for(int j = 0; j < neighbors; j++) {
			cin >> neighbor_box_id;
			temp.id = neighbor_box_id;
			grid[current_box_id].bottom.push_back(temp);
		}
		cin >> neighbors;
		for(int j = 0; j < neighbors; j++) {
			cin >> neighbor_box_id;
			temp.id = neighbor_box_id;
			grid[current_box_id].left.push_back(temp);
		}
		cin >> neighbors;
		for(int j = 0; j < neighbors; j++) {
			cin >> neighbor_box_id;
			temp.id = neighbor_box_id;
			grid[current_box_id].right.push_back(temp);
		}

		/* Scan the box DSV (temperature). */
		cin >> grid[current_box_id].box_dsv;
	}

	cin >> last_line;
	if (last_line != -1) {
		/* The last line should always be -1. */
		return (ERROR);
	}

	/*
	 * Compute the contanct distances of
	 * neighbour nodes for later use.
	 */
	result = compute_contact_distance();
	assert (result == SUCCESS);

	return (SUCCESS);
}

/*
 * Commits the updated DSVs in the original boxes in the grid.
 * Also finds the maximum and minimum box DSV (temp).
 *
 * Returns SUCCESS in case of successful grid update.
 */
int inline commit_dsv_update() {

	max_dsv = min_dsv = updated_DSVs[0];

	for(int i = 0; i < number_of_boxes; i++) {
		/* Commit the updated DSVs*/
		grid[i].box_dsv = updated_DSVs[i];

		/* Find the max and min DSVs */
		if (grid[i].box_dsv > max_dsv) {
			max_dsv = grid[i].box_dsv;
		} else if(grid[i].box_dsv < min_dsv) {
			min_dsv = grid[i].box_dsv;
		}
	}

	return (SUCCESS);
}


/*
 * Checks whether the DSVs reached convergence or not.
 *
 * Returns SUCCESS in case of convergence reached, otherwise ERROR.
 */
int inline check_for_convergence() {
	
	if ((max_dsv - min_dsv) <= (max_dsv * epsilon)) {
		return (SUCCESS);
	}

	return (ERROR);
}

/*
 * All the new threads call this function and compute updated DSV.
 *
 * message - Thread specific message that provides information on the
 * 	     grid partitioning.
 */
void*  compute_updated_DSV(void *message) {

	/* Iterator to go through the neighbors of a box. */
	vector<neighbor>::iterator iter;

	/* Used as a temporary variable to store the
	total DSV to aid in computing the weighed average. */
	float total_DSV = 0;

	/* Average adjacent temperature for each current box. */
	float average_DSV;
	
	thread_data *mydata = (thread_data*) message;

	/*
	 * Go through allowed  boxes in the grid and compute updated DSV.
	 */
	for (int i = mydata->start; i <= mydata->end; i++) {
		total_DSV = 0;

		/* Check the top neighbors. */
		if (grid[i].top.empty()) {
			total_DSV += grid[i].box_dsv * grid[i].width;
		} else {
			for (iter = grid[i].top.begin(); iter != grid[i].top.end(); iter++) {
				total_DSV += grid[iter->id].box_dsv * (iter->contact_distance);
			}
		}

		/* Check the bottom neighbors. */
		if (grid[i].bottom.empty()) {
			total_DSV += grid[i].box_dsv * grid[i].width;
		} else {
			for (iter = grid[i].bottom.begin(); iter != grid[i].bottom.end(); iter++) {
				total_DSV += grid[iter->id].box_dsv * (iter->contact_distance);
			}
		}

		/* Check the left neighbors. */
		if (grid[i].left.empty())
		{
			total_DSV += grid[i].box_dsv * grid[i].height;
		} else {
			for (iter = grid[i].left.begin(); iter != grid[i].left.end(); iter++) {
				total_DSV += grid[iter->id].box_dsv * (iter->contact_distance);
			}
		}

		/* Check the right neighbors. */
		if (grid[i].right.empty()) {
			total_DSV += grid[i].box_dsv * grid[i].height;
		} else {
			for (iter = grid[i].right.begin(); iter != grid[i].right.end(); iter++) {
				total_DSV += grid[iter->id].box_dsv * (iter->contact_distance);
			}
		}

		/* Compute the average DSV. */
		average_DSV = (total_DSV / grid[i].perimeter);

		/* Compute the updated values based on the average adjacent DSV. */
		if (average_DSV >= grid[i].box_dsv) {
			updated_DSVs[i] = grid[i].box_dsv +
				((average_DSV - grid[i].box_dsv) * affect_rate);
		} else if (average_DSV < grid[i].box_dsv) {
			updated_DSVs[i] = grid[i].box_dsv -
				((grid[i].box_dsv - average_DSV) * affect_rate);
		}
	}
//	cout << mydata->thread_id << "--" << mydata->start << "--" << mydata->end << "\n"; 
	pthread_exit (0);
}



/*
 * Function that does the stencil computation.
 * The iterative computations are repeated until convergence.
 *
 * Returns SUCCESS in case of successful computation.
 */
int do_stencil_computation() {

	int result;
	convergence_iteration = 0;
	pthread_t *threads;
	pthread_attr_t thread_attr;
	thread_data message[number_of_threads]; //// Message to be passed to each thread.	
	int start_index = 0;
	int number_of_partition = number_of_boxes/number_of_threads;

	// Partition the grid for each Thread.
	for (int i = 0; i < number_of_threads; i++) {
		message[i].thread_id = i;
		message[i].start = start_index;
		if (i == (number_of_threads - 1)) {
			message[i].end = number_of_boxes;
		} else {
			message[i].end = (start_index +
			    number_of_partition - 1);
		}
		start_index = start_index + number_of_partition;
	}

	do { // Convergence Loop.
		/* Create number of threads. */
		threads = new pthread_t[number_of_threads];
		pthread_attr_init(&thread_attr);
		pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);

		for(int i = 0; i < number_of_threads; i++) {
			result = pthread_create(&threads[i], &thread_attr,
			    compute_updated_DSV, (void *) &message[i]);
			assert(result == 0);
		}
		
		// Await Thread Exit.
		for (int i = 0; i < number_of_threads; i++) {
			result = pthread_join (threads[i], NULL);
			assert(result == 0);
		}

		/* Update the grid. */
		result = commit_dsv_update();
		assert(result == SUCCESS);

		convergence_iteration = convergence_iteration + 1;
		/* Check if the convergence reached. */
		result = check_for_convergence();
	} while (result != SUCCESS);

	return (SUCCESS);
}

int main(int argc, char* argv[]) {
	
	int result;

	/* The timers. */
	time_t time1;
	time_t time2;
	clock_t clock1;
	clock_t clock2;
	system_clock::time_point chrono1;
	system_clock::time_point chrono2;
	std::chrono::duration<float> time_taken;

	/* Check whether wrong number of arguments. */
	assert(argc == 4);

	/* Read affect_rate, epsilon and number of threads. */
	affect_rate = atof(argv[1]);
	epsilon = atof(argv[2]);
	number_of_threads = atof(argv[3]);
	
	/* Read the number of grid and create essential objects. */
	cin >> number_of_boxes;
	grid = new box[number_of_boxes];
	updated_DSVs = new float[number_of_boxes];

	/* Read the input file and construct the grid. */
	result = read_input_file();
	assert(result == SUCCESS);

	cout << "\n\n*******************************************************************\n\n";
	time(&time1);
	clock1 = clock();
	chrono1 = system_clock::now();

	result = do_stencil_computation();
	assert(result == SUCCESS);

	time(&time2);
	clock2 = clock();
	chrono2 = system_clock::now();
	time_taken = chrono2 - chrono1;

	/* Print the results. */
	cout << "Dissipation converged in "<< convergence_iteration << " iterations.\n";
	cout << "With max DSV = "<< max_dsv << " and min DSV = " << min_dsv << ".\n";
	cout << "Affect rate = " << affect_rate << ";   Epsilon: " << epsilon << ".\n";
	cout << "Elapsed covergence loop time (clock)  : " << (clock2 - clock1) << "\n";
	cout << "Elapsed covergence loop time (time)   : " << difftime(time2, time1) << ".\n";
	cout << "Elapsed covergence loop time (chrono) : " <<
	chrono::duration<double,std::milli>(time_taken).count() << ".\n";
	cout << "\n*******************************************************************\n\n";

	return (0);
}
