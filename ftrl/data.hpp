/*
 * data.hpp
 *
 *  Created on: Jun 18, 2015
 *      Author: florian
 */

#ifndef DATA_HPP_
#define DATA_HPP_

#include "ftrl.hpp"

#include <boost/functional/hash.hpp>
#include <boost/lexical_cast.hpp>

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

#define MISSING_FEATURES 4

struct task_paths{

	std::string traindata_path;
	std::string testdata_path;
	std::string holdout_traindata_path;
	std::string holdout_testdata_path;

	unsigned long int default_hash;
	std::map <std::string, unsigned long int> hash_list;
	bool poly2_features;

};

struct avito_impression{

	std::ifstream data_stream;
	bool is_open;

	std::vector<std::string> header;

	std::string id;
	std::vector<ftrl_node> line;
	unsigned short int label;

};

class avito_data{
public:

	void read_data_conf(std::string config_path);
	avito_data(std::string data_conf_path);

	void open_traindata_stream(void);
	void open_testdata_stream(void);
	void open_holdout_traindata_stream(void);
	void open_holdout_testdata_stream(void);

	void read_line(avito_impression &data_impression);
	void read_traindata_line(void);
	void read_testdata_line(void);
	void read_holdout_traindata_line(void);
	void read_holdout_testdata_line(void);

	~avito_data();

	task_paths data_parameters;

	avito_impression traindata_impression;
	avito_impression testdata_impression;
	avito_impression holdout_traindata_impression;
	avito_impression holdout_testdata_impression;

private:

	std::vector<std::string> read_header(std::ifstream &data_stream);

};

#endif /* DATA_HPP_ */
