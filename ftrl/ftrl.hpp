/*
 * ftrl.hpp
 *
 *  Created on: Jun 8, 2015
 *      Author: florian
 */

#ifndef FTRL_HPP_
#define FTRL_HPP_


#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

#include <string>
#include <string.h>


typedef float ftrl_float;
typedef double ftrl_double;
typedef unsigned short int ftrl_uint8;
typedef unsigned int ftrl_uint16;
typedef unsigned long int ftrl_uint32;
typedef int ftrl_int16;

/* Sparse representation of feature */
struct ftrl_node{
	/* Index of the feature */
	ftrl_uint16 index;
	/* Value of the feature at index */
	ftrl_float value;
};

/* Defines the task to solve */
struct ftrl_task{

	/* Number of different features in the input */
	ftrl_uint16 num_features;
	/* Number of data instances */
	ftrl_uint16 num_instances;
	/* If data should be shuffled before training (if data is ordered by label) */
	bool shuffle_data;
	/* Pick a random datapoint for holdout */
	ftrl_float random_holdout_rate;
	/* Number complete trainingdata passes */
	ftrl_uint16 epochs;
	/* Pointer to data ID's */
	std::string* ID;
	/* Pointer to the training labels */
	ftrl_uint8* Y;
	/* Pointer to the data */
	ftrl_node* X;
	/* Pointer to beginning of the rows in data */
	ftrl_uint32* row_ptr;

};

/* Hyper-parameter set for the classifier to use */
struct ftrl_parameter{

	/* Learning rate and learning-rate-decay */
	ftrl_float alpha;
	ftrl_float beta;

	/* L1-norm weighting parameters */
	ftrl_float l1_lambda_w;

	/* L2-norm weighting parameters */
	ftrl_float l2_lambda_w;

};

class ftrl{
public:

	/* Constructs ftrl-Model based on given "problem" and "parameters" */
	ftrl(std::string conf_path);
	virtual ~ftrl();

	/*
	 * Read all parameters from extern file
	 */
	void read_conf(std::string conf_path);

	/*
	 * Read input dimension from avito-data config file
	 */
	void avito_init_params(std::string data_conf_path);

	/*
	 * Save trained model weights
	 */
	const void saveModel(std::string outputPath) const;

	/*
	 * Load weights of a trained model
	 */
	void loadModel(std::string inputPath);

	/*
	 * Predicts array of probabilities dim(1, num_classes)
	 * and saves them in "p".
	 */
	void predict(std::vector<ftrl_node> &instance);

	/*
	 * Preforms update on the parameters by SGD
	 */
	void update(std::vector<ftrl_node> &instance, ftrl_uint8 y);

	/* Task to be solved */
	ftrl_task task;


	/* Used Hyper-parameter-set */
	ftrl_parameter parameters;

	/* Linear weights */
	std::vector< ftrl_float > w;

	/* Lazy weights */
	std::vector< ftrl_float > z;
	std::vector< ftrl_float > n;

	/* Bias */
	ftrl_float b;

	/* stores the predicted probabilities of the current instance */
	ftrl_float p;

private:

	/* Coefficient to scale the initialized weights */
	ftrl_float init_coeff;

	/* Output error: (p - y) */
	ftrl_float diff;

};


ftrl_float accuracy(const ftrl_float &p, const ftrl_uint8 &y);

ftrl_float logloss(const ftrl_float &p, const ftrl_uint8 &y);

ftrl_float l1(const ftrl_float& w);


#endif /* FTRL_HPP_ */
