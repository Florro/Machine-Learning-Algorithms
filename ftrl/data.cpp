/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include "data.hpp"

void avito_data::read_data_conf(std::string config_path)
{

	bool debug = false;

	std::ifstream config_stream((char*)config_path.c_str(), std::ios::in);
	std::istringstream line_stream;

	std::string line;
	std::string token;

	// Read header:
	getline(config_stream, line);

	//Read traindata path:
	getline(config_stream, line);
	line_stream.str(line);
	while(line_stream){	if (!getline( line_stream, token, ':' )) break;	}
	data_parameters.traindata_path = token;

	line_stream.str(std::string());
	line_stream.clear();

	if(debug) std::cout << token << std::endl;

	//Read testdata path:
	getline(config_stream, line);
	line_stream.str(line);
	while(line_stream){	if (!getline( line_stream, token, ':' )) break;	}
	data_parameters.testdata_path = token;

	line_stream.str(std::string());
	line_stream.clear();

	if(debug) std::cout << token << std::endl;

	//Read holdout_traindata path:
	getline(config_stream, line);
	line_stream.str(line);
	while(line_stream){	if (!getline( line_stream, token, ':' )) break;	}
	data_parameters.holdout_traindata_path = token;

	line_stream.str(std::string());
	line_stream.clear();

	if(debug) std::cout << token << std::endl;

	//Read holdout_testdata path:
	getline(config_stream, line);
	line_stream.str(line);
	while(line_stream){	if (!getline( line_stream, token, ':' )) break;	}
	data_parameters.holdout_testdata_path = token;

	line_stream.str(std::string());
	line_stream.clear();

	if(debug) std::cout << token << std::endl;

	// Read header and empty line:
	getline(config_stream, line);
	getline(config_stream, line);

	//Read Poly2 boolean:
	getline(config_stream, line);
	line_stream.str(line);
	while(line_stream){	if (!getline( line_stream, token, ':' )) break;	}
	data_parameters.poly2_features = false;
	if(token == "true"){
		data_parameters.poly2_features = true;
	}

	line_stream.str(std::string());
	line_stream.clear();

	if(debug) std::cout << token << std::endl;

	// Read header and empty line:
	getline(config_stream, line);
	getline(config_stream, line);

	std::string hash_dim;
	while(getline(config_stream, line)){
		std::istringstream hash_stream(line);
		getline( hash_stream, token, ':' );
		getline( hash_stream, hash_dim, ':' );

		data_parameters.hash_list[token] = atoi(hash_dim.c_str());

		if(debug) std::cout << token << "  " << hash_dim << std::endl;
	}

	data_parameters.default_hash = 100000;

}

avito_data::avito_data(std::string data_conf_path)
{
	this->read_data_conf(data_conf_path);
	this->traindata_impression.is_open = false;
	this->testdata_impression.is_open = false;
	this->holdout_traindata_impression.is_open = false;
	this->holdout_testdata_impression.is_open = false;
}

std::vector<std::string> avito_data::read_header(std::ifstream &data_stream)
{

	std::vector<std::string> header;
	std::string header_line;
	std::string header_instance;

	getline(data_stream, header_line);

	// Remove '\n' from header line
	header_line.erase(header_line.length()-1);

	std::istringstream line_stream( header_line );

	while(line_stream){

		if (!getline( line_stream, header_instance, ',' )) break;
		header.push_back(header_instance);
	}

	return header;
}


void avito_data::open_traindata_stream(void)
{
	traindata_impression.data_stream.open((char*)data_parameters.traindata_path.c_str(), std::ios::in);
	traindata_impression.header = this->read_header(traindata_impression.data_stream);
	unsigned int size = data_parameters.hash_list.size();
	if(data_parameters.poly2_features) size += size * (size - 1);

	//traindata_impression.line.resize(traindata_impression.header.size() - MISSING_FEATURES - 1);
	traindata_impression.line.resize(size);
	traindata_impression.is_open = true;
}

void avito_data::open_testdata_stream(void)
{
	testdata_impression.data_stream.open((char*)data_parameters.testdata_path.c_str(), std::ios::in);
	testdata_impression.header = this->read_header(testdata_impression.data_stream);
	unsigned int size = data_parameters.hash_list.size();
	if(data_parameters.poly2_features) size += size * (size - 1);

	//testdata_impression.line.resize(testdata_impression.header.size() - MISSING_FEATURES);
	testdata_impression.line.resize(size);
	testdata_impression.is_open = true;
}

void avito_data::open_holdout_traindata_stream(void)
{
	holdout_traindata_impression.data_stream.open((char*)data_parameters.holdout_traindata_path.c_str(), std::ios::in);
	holdout_traindata_impression.header = this->read_header(holdout_traindata_impression.data_stream);
	unsigned int size = data_parameters.hash_list.size();
	if(data_parameters.poly2_features) size += size * (size - 1);

	std::cout << "size " << size << std::endl;
	//holdout_traindata_impression.line.resize(holdout_traindata_impression.header.size() - MISSING_FEATURES - 1);
	holdout_traindata_impression.line.resize(size);
	holdout_traindata_impression.is_open = true;
}

void avito_data::open_holdout_testdata_stream(void)
{
	holdout_testdata_impression.data_stream.open((char*)data_parameters.holdout_testdata_path.c_str(), std::ios::in);
	holdout_testdata_impression.header = this->read_header(holdout_testdata_impression.data_stream);
	//holdout_testdata_impression.line.resize(holdout_testdata_impression.header.size() - MISSING_FEATURES);
	unsigned int size = data_parameters.hash_list.size();
	if(data_parameters.poly2_features) size += size * (size - 1);

	holdout_testdata_impression.line.resize(size);
	holdout_testdata_impression.is_open = true;
}



void avito_data::read_line(avito_impression &data_impression)
{

	/* If line is not empty read one line from the file */
	std::string data_line;
	if (!getline( data_impression.data_stream, data_line)){
		data_impression.data_stream.close();
		data_impression.is_open = false;
		return;
	}

	/* Open a linestream to iterate through the line */
	std::istringstream line_stream( data_line );

	unsigned int count = 0;
	unsigned int idx = 0;
	unsigned long int stride = 1;
	std::string feature;

	for(;; count++, idx++){

		/* read token from line until new line begins */
		if (!getline( line_stream, feature, ',' )) break;

		if(data_impression.header[count] == "ID")
		{
			data_impression.id = feature.c_str();
			idx--;
		}
		else if(data_impression.header[count] == "IsClick")
		{
			data_impression.label = atoi(feature.c_str());
			idx--;
		}
		else if(!data_parameters.hash_list.count(data_impression.header[count]))
		{
			idx--;
		}
		else
		{

			std::string key = data_impression.header[count] + "_" + feature.c_str();

			// Here the hash-function will be applied
			boost::hash<std::string> string_hash;

			unsigned long int index = string_hash(key) % data_parameters.hash_list[data_impression.header[count]] + stride;

			//unsigned long int index = string_hash(key) % (data_parameters.default_hash - 1) + 1;

			//std::cout << data_parameters.hash_list[data_impression.header[count]] << " " << data_impression.header[count] << "  " << feature << std::endl;

			data_impression.line[idx].index = index;
			data_impression.line[idx].value = 1.0f;

			stride += data_parameters.hash_list[data_impression.header[count]];
		}

	}

	// append poly2 features:
	if(data_parameters.poly2_features){
		for(unsigned int i = 0; i < data_parameters.hash_list.size(); i++){
			for(unsigned int j = i+1; j < data_parameters.hash_list.size(); j++){

				idx = i * data_parameters.hash_list.size() + j;
				boost::hash<std::string> string_hash;
				std::string key = data_impression.header[i] + "_" + data_impression.header[j] + "_";
				key += boost::lexical_cast<std::string>(data_impression.line[i].index) + boost::lexical_cast<std::string>(data_impression.line[j].index);

				unsigned long int index = string_hash(key) % data_parameters.default_hash + data_parameters.default_hash;
				data_impression.line[idx].index = index;
				data_impression.line[idx].value = 1.0f;
				stride += data_parameters.default_hash;
			}
		}
	}

}

void avito_data::read_traindata_line(void)
{
	this->read_line(traindata_impression);
}
void avito_data::read_testdata_line(void)
{
	this->read_line(testdata_impression);
}
void avito_data::read_holdout_traindata_line(void)
{
	this->read_line(holdout_traindata_impression);
}
void avito_data::read_holdout_testdata_line(void)
{
	this->read_line(holdout_testdata_impression);
}


avito_data::~avito_data(void){
	if(traindata_impression.data_stream.is_open()){
		traindata_impression.data_stream.close();
	}
	else if(testdata_impression.data_stream.is_open()){
		testdata_impression.data_stream.close();
	}
	else if(holdout_traindata_impression.data_stream.is_open()){
		holdout_traindata_impression.data_stream.close();
	}
	else if(holdout_testdata_impression.data_stream.is_open()){
		holdout_testdata_impression.data_stream.close();
	}
}
