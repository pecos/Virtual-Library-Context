/*
  Fast Artificial Neural Network Library (fann)
  Copyright (C) 2003-2016 Steffen Nissen (steffen.fann@gmail.com)
  
  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.
  
  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.
  
  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <stdio.h>
#include <thread>
#include <vector>

#include "fann.h"

const int REPEAT = 100;

int fann_call_back_nothing(struct fann *ann,
					struct fann_train_data *data,
					unsigned int max_neurons,
					unsigned int neurons_between_reports,
					float desired_error,
					unsigned int epochs) {
						return true;
					}

int cascade_train()
{
	for (int r = 0; r < REPEAT; r++) {
	struct fann *ann;
	struct fann_train_data *train_data, *test_data;
	const float desired_error = (const float)0.0;
	unsigned int max_neurons = 30;
	unsigned int neurons_between_reports = 1;
	unsigned int bit_fail_train, bit_fail_test;
	float mse_train, mse_test;
	unsigned int i = 0;
	fann_type *output;
	fann_type steepness;
	int multi = 0;
	enum fann_activationfunc_enum activation;
	enum fann_train_enum training_algorithm = FANN_TRAIN_RPROP;
	
	printf("Reading data.\n");
	 
	train_data = fann_read_train_from_file("datasets/parity8.train");
	test_data = fann_read_train_from_file("datasets/parity8.test");

	fann_scale_train_data(train_data, -1, 1);
	fann_scale_train_data(test_data, -1, 1);
	
	printf("Creating network.\n");
	
	ann = fann_create_shortcut(2, fann_num_input_train_data(train_data), fann_num_output_train_data(train_data));
		
	fann_set_training_algorithm(ann, training_algorithm);
	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_LINEAR);
	fann_set_train_error_function(ann, FANN_ERRORFUNC_LINEAR);
	
	if(!multi)
	{
		/*steepness = 0.5;*/
		steepness = 1;
		fann_set_cascade_activation_steepnesses(ann, &steepness, 1);
		/*activation = FANN_SIN_SYMMETRIC;*/
		activation = FANN_SIGMOID_SYMMETRIC;
		
		fann_set_cascade_activation_functions(ann, &activation, 1);		
		fann_set_cascade_num_candidate_groups(ann, 8);
	}	
		
	if(training_algorithm == FANN_TRAIN_QUICKPROP)
	{
		fann_set_learning_rate(ann, 0.35f);
		fann_randomize_weights(ann, -2.0f, 2.0f);
	}
	
	fann_set_bit_fail_limit(ann, (fann_type)0.9);
	fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	// fann_print_parameters(ann);
		
	fann_save(ann, "cascade_train2.net");
	
	printf("Training network.\n");

	fann_set_callback(ann, fann_call_back_nothing);
	fann_cascadetrain_on_data(ann, train_data, max_neurons, neurons_between_reports, desired_error);
	
	fann_print_connections(ann);
	
	mse_train = fann_test_data(ann, train_data);
	bit_fail_train = fann_get_bit_fail(ann);
	mse_test = fann_test_data(ann, test_data);
	bit_fail_test = fann_get_bit_fail(ann);
	
	printf("\nTrain error: %f, Train bit-fail: %d, Test error: %f, Test bit-fail: %d\n\n", 
		   mse_train, bit_fail_train, mse_test, bit_fail_test);
	
	for(i = 0; i < train_data->num_data; i++)
	{
		output = fann_run(ann, train_data->input[i]);
		if((train_data->output[i][0] >= 0 && output[0] <= 0) ||
		   (train_data->output[i][0] <= 0 && output[0] >= 0))
		{
			printf("ERROR: %f does not match %f\n", train_data->output[i][0], output[0]);
		}
	}
	
	printf("Saving network.\n");
	
	fann_save(ann, "cascade_train.net");
	
	printf("Cleaning up.\n");
	fann_destroy_train(train_data);
	fann_destroy_train(test_data);
	fann_destroy(ann);
	}
	return 0;
}


void train_on_steepness_file(struct fann *ann, char *filename,
							 unsigned int max_epochs, unsigned int epochs_between_reports,
							 float desired_error, float steepness_start,
							 float steepness_step, float steepness_end)
{
	float error;
	unsigned int i;

	struct fann_train_data *data = fann_read_train_from_file(filename);

	if(epochs_between_reports)
	{
		printf("Max epochs %8d. Desired error: %.10f\n", max_epochs, desired_error);
	}

	fann_set_activation_steepness_hidden(ann, steepness_start);
	fann_set_activation_steepness_output(ann, steepness_start);
	for(i = 1; i <= max_epochs; i++)
	{
		/* train */
		error = fann_train_epoch(ann, data);

		// /* print current output */
		// if(epochs_between_reports &&
		//    (i % epochs_between_reports == 0 || i == max_epochs || i == 1 || error < desired_error))
		// {
		// 	printf("Epochs     %8d. Current error: %.10f\n", i, error);
		// }

		if(error < desired_error)
		{
			steepness_start += steepness_step;
			if(steepness_start <= steepness_end)
			{
				// printf("Steepness: %f\n", steepness_start);
				fann_set_activation_steepness_hidden(ann, steepness_start);
				fann_set_activation_steepness_output(ann, steepness_start);
			}
			else
			{
				break;
			}
		}
	}
	fann_destroy_train(data);
}

int steepness_train()
{
	for (int r = 0; r < REPEAT*16; r++) {
	const unsigned int num_input = 2;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 3;
	const float desired_error = (const float) 0.001;
	const unsigned int max_epochs = 500000;
	const unsigned int epochs_between_reports = 1000;
	unsigned int i;
	fann_type *calc_out;

	struct fann_train_data *data;

	struct fann *ann = fann_create_standard(num_layers,
								   num_input, num_neurons_hidden, num_output);

	data = fann_read_train_from_file("datasets/xor.data");

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_set_training_algorithm(ann, FANN_TRAIN_QUICKPROP);

	train_on_steepness_file(ann, "datasets/xor.data", max_epochs,
							epochs_between_reports, desired_error, (float) 1.0, (float) 0.1,
							(float) 20.0);

	fann_set_activation_function_hidden(ann, FANN_THRESHOLD_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_THRESHOLD_SYMMETRIC);

	for(i = 0; i != fann_length_train_data(data); i++)
	{
		calc_out = fann_run(ann, data->input[i]);
		printf("XOR test (%f, %f) -> %f, should be %f, difference=%f\n",
			   data->input[i][0], data->input[i][1], calc_out[0], data->output[i][0],
			   (float) fann_abs(calc_out[0] - data->output[i][0]));
	}


	fann_save(ann, "xor_float.net");

	fann_destroy(ann);
	fann_destroy_train(data);
	}
	return 0;
}

int xor_train()
{
	for (int r = 0; r < REPEAT*16; r++) {
	fann_type *calc_out;
	const unsigned int num_input = 2;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 3;
	const float desired_error = (const float) 0;
	const unsigned int max_epochs = 1000;
	const unsigned int epochs_between_reports = 10;
	struct fann *ann;
	struct fann_train_data *data;

	unsigned int i = 0;
	unsigned int decimal_point;

	printf("Creating network.\n");
	ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

	data = fann_read_train_from_file("datasets/xor.data");

	fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.01f);

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

	fann_init_weights(ann, data);
	
	printf("Training network.\n");
	fann_set_callback(ann, fann_call_back_nothing);
	fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);

	printf("Testing network. %f\n", fann_test_data(ann, data));

	for(i = 0; i < fann_length_train_data(data); i++)
	{
		calc_out = fann_run(ann, data->input[i]);
		printf("XOR test (%f,%f) -> %f, should be %f, difference=%f\n",
			   data->input[i][0], data->input[i][1], calc_out[0], data->output[i][0],
			   fann_abs(calc_out[0] - data->output[i][0]));
	}

	printf("Saving network.\n");

	fann_save(ann, "xor_float.net");

	decimal_point = fann_save_to_fixed(ann, "xor_fixed.net");
	fann_save_train_to_fixed(data, "xor_fixed.data", decimal_point);

	printf("Cleaning up.\n");
	fann_destroy_train(data);
	fann_destroy(ann);
	}
	return 0;
}


int main() {
	std::vector<std::thread> t(2);
    t[0] = std::thread(steepness_train);
    t[1] = std::thread(xor_train);

    t[0].join();
    t[1].join();
}