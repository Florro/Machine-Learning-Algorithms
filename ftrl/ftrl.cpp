/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include "ftrl.hpp"

ftrl_float accuracy(const ftrl_float &p, const ftrl_uint8 &y){

	ftrl_uint8 pred = (p <= 0.5) ? 0 : 1;
	return (pred == y) ? 1.0 : 0.0;
}

ftrl_float logloss(const ftrl_float &p, const ftrl_uint8 &y){
	return (y == 1) ? -log(p) : -log(1.0 - p);
}

ftrl_float l1(const ftrl_float &w){
	ftrl_float l1_term = 0.0;
	if(w != 0.0)
		l1_term = (w > 0)? 1.0 : 0.0;

	return l1_term;
}

void ftrl::read_conf(std::string conf_path)
{
	bool debug = false;

	std::ifstream config_stream((char*)conf_path.c_str(), std::ios::in);
	std::istringstream line_stream;

	std::string line;
	std::string token;

	// Read header:
	getline(config_stream, line);

	//Read parameters:
	getline(config_stream, line);
	line_stream.str(line);
	while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
	parameters.alpha = atof(token.c_str());

	line_stream.str(std::string());
	line_stream.clear();

	if(debug) std::cout << token << " " <<  parameters.alpha << std::endl;

	//Read parameters:
	getline(config_stream, line);
	line_stream.str(line);
	while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
	parameters.beta = atof(token.c_str());

	line_stream.str(std::string());
	line_stream.clear();

	if(debug) std::cout << token << " " <<  parameters.beta << std::endl;

	//Read parameters:
	getline(config_stream, line);
	line_stream.str(line);
	while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
	parameters.l1_lambda_w = atof(token.c_str());

	line_stream.str(std::string());
	line_stream.clear();

	if(debug) std::cout << token << " " <<  parameters.l1_lambda_w << std::endl;

	//Read parameters:
	getline(config_stream, line);
	line_stream.str(line);
	while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
	parameters.l2_lambda_w = atof(token.c_str());

	line_stream.str(std::string());
	line_stream.clear();

	if(debug) std::cout << token << " " <<  parameters.l2_lambda_w << std::endl;

	//Read parameters:
	getline(config_stream, line);
	line_stream.str(line);
	while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
	task.epochs = atoi(token.c_str());

	line_stream.str(std::string());
	line_stream.clear();

	if(debug) std::cout << token << " " <<  task.epochs << std::endl;

	//Read parameters:
	getline(config_stream, line);
	line_stream.str(line);
	while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
	task.num_features = atoi(token.c_str());

	line_stream.str(std::string());
	line_stream.clear();

	if(debug) std::cout << token << " " <<  task.num_features << std::endl;

	task.num_instances = 1;
	task.random_holdout_rate = 0.0;
	task.shuffle_data = false;
}

ftrl::ftrl(std::string conf_path)
{

	this->read_conf(conf_path);

	this->init_coeff = 0.01;
	this->b = 0.0;

	this->p = -999.;
	this->diff = -999.;

}

ftrl::~ftrl(){}

void ftrl::avito_init_params(std::string data_conf_path){

	bool debug = false;

	std::ifstream config_stream((char*)data_conf_path.c_str(), std::ios::in);

	std::string line;
	std::string token;
	std::string header;

	// Read header:
	for(int i = 0; i < 8; i++){
		getline(config_stream, line);
	}

	std::istringstream line_stream(line);
	while(line_stream){	if (!getline( line_stream, token, ':' )) break;	}
	if(debug) std::cout << token << std::endl;

	bool poly2 = ( token == "true" ) ? true : false;
	//if ( token == "true" )//bool(token.c_str());

	getline(config_stream, line);
	getline(config_stream, line);

	task.num_features = 1; // 0 index for CTR-numerical value
	int feature_count = 0;
	while(getline(config_stream, line)){
		std::istringstream hash_stream(line);
		getline( hash_stream, header, ':' );
		getline( hash_stream, token, ':' );

		task.num_features += atoi(token.c_str());

		if(debug) std::cout << header << ": " << token << " Current inputDim: " << task.num_features << std::endl;
		feature_count++;
	}

	if(debug) std::cout << "Num-features: " << task.num_features << std::endl;

	if(poly2){
		for(int i = 0; i < feature_count; i++){
			for(int j = 0; j < feature_count; j++){
				task.num_features += 100000;
			}
		}
	}

	for(ftrl_uint16 i = 0; i < task.num_features; i++){

		w.push_back(0.0);//init_coeff * (drand48())
		z.push_back(0.0);
		n.push_back(0.0);
	}

	std::cout << "ftrl input Dimension: " << task.num_features << std::endl;
}

const void ftrl::saveModel(std::string outputPath) const
{
	std::ofstream output_stream((char*)outputPath.c_str(), std::ios::out);
	for( unsigned int i = 0; i < task.num_features; ++i )
	{
		output_stream << w[i] << std::endl;
	}
	output_stream.close();
}

void ftrl::loadModel(std::string inputPath)
{
	std::ifstream input_stream((char*)inputPath.c_str(), std::ios::out);
	std::string tmp;
	unsigned int i = 0;
	while ( input_stream >> tmp)
	{
		w[i] = atof(tmp.c_str());
		++i;
	}
	input_stream.close();
}

void ftrl::predict(std::vector<ftrl_node> &instance){

	ftrl_float sum = 0.0;
	//#pragma omp parallel for reduction(+:sum)
	for( int i = 0; i < instance.size(); i++){
		sum += w[instance[i].index];
	}

	p =  1.0 / (1.0 + exp(-std::max(std::min(sum, ftrl_float(15.0)), ftrl_float(-15.))));

	//printf("prediction: %f\n", p);
	/*
	p = 0.0;
	#pragma omp parallel for reduction(*:p)
	for(std::vector<ftrl_node>::iterator node = instance.begin(); node != instance.end(); ++node){
		p += w[node->index];
	}

	p =  1.0 / (1.0 + exp(-std::max(std::min(p, ftrl_float(15.0)), ftrl_float(-15.))));
	*/
}

void ftrl::update(std::vector<ftrl_node> &instance, ftrl_uint8 y){

	diff = p - y;

	//#pragma omp parallel for
	for( int nIdx = 0; nIdx < instance.size(); nIdx++)
	{
	//for(std::vector<ftrl_node>::iterator node = instance.begin(); node != instance.end(); ++node){

		ftrl_float sigma = (sqrt(n[instance[nIdx].index] + diff * diff) - sqrt(n[instance[nIdx].index])) / parameters.alpha;

		z[instance[nIdx].index] += diff - sigma * w[instance[nIdx].index];
		n[instance[nIdx].index] += diff * diff;

		ftrl_int16 sign = (z[instance[nIdx].index] < 0.0) ? -1 : 1;

		if(sign * z[instance[nIdx].index] <= parameters.l1_lambda_w){
			w[instance[nIdx].index] = 0.0;
		}
		else{
			w[instance[nIdx].index] = (sign * parameters.l1_lambda_w - z[instance[nIdx].index]) / (((parameters.beta + sqrt(n[instance[nIdx].index]))/ parameters.alpha) + parameters.l2_lambda_w);
		}
	}

	/*
	for(std::vector<ftrl_node>::iterator node = instance.begin(); node != instance.end(); ++node){

		ftrl_float sigma = (sqrt(n[node->index] + diff * diff) - sqrt(n[node->index])) / parameters.alpha;

		z[node->index] += diff - sigma * w[node->index];
		n[node->index] += diff * diff;

		ftrl_int16 sign = (z[node->index] < 0.0) ? -1 : 1;

		if(sign * z[node->index] <= parameters.l1_lambda_w){
			w[node->index] = 0.0;
		}
		else{
			w[node->index] = (sign * parameters.l1_lambda_w - z[node->index]) / (((parameters.beta + sqrt(n[node->index]))/ parameters.alpha) + parameters.l2_lambda_w);
		}
	}
	*/

}
