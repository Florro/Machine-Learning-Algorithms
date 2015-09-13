#include "fmdata.hpp"

#define DEBUG false

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

  if(debug) std::cout << "Test-data path: " << token << std::endl;

  //Read testdata path:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, ':' )) break;	}
  data_parameters.testdata_path = token;

  line_stream.str(std::string());
  line_stream.clear();

  if(debug) std::cout << "Train-data path: " << token << std::endl;

  //Read holdout_traindata path:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, ':' )) break;	}
  data_parameters.holdout_traindata_path = token;

  line_stream.str(std::string());
  line_stream.clear();

  if(debug) std::cout << "Holdout Train-data path: " << token << std::endl;

  //Read holdout_testdata path:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, ':' )) break;	}
  data_parameters.holdout_testdata_path = token;

  line_stream.str(std::string());
  line_stream.clear();

  if(debug) std::cout << "Holdout Test-data path: " << token << std::endl;

  // Read header and empty line:
  getline(config_stream, line);
  getline(config_stream, line);

  //Read Default hash boolean:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, ':' )) break;	}
  data_parameters.useDefaultHash = ( token != "false" ) ? true : false;
  data_parameters.default_hash = 0;
  if( data_parameters.useDefaultHash ) {
    data_parameters.default_hash = atoi(token.c_str());
  }

  line_stream.str(std::string());
  line_stream.clear();

  if(debug) std::cout << "Default hashdim: " << token << std::endl;

  // Read header and empty line:
  getline(config_stream, line);
  getline(config_stream, line);

  std::string hash_dim;
  while( getline(config_stream, line) )
  {
    std::istringstream hash_stream(line);
    getline( hash_stream, token, ':' );
    getline( hash_stream, hash_dim, ':' );

    data_parameters.hash_list[token] = atoi(hash_dim.c_str());

    if(debug) std::cout << token << "  " << hash_dim << std::endl;
  }
}

avito_data::avito_data(std::string data_conf_path)
{
  read_data_conf(data_conf_path);
  traindata_impression.is_open = false;
  testdata_impression.is_open = false;
  holdout_traindata_impression.is_open = false;
  holdout_testdata_impression.is_open = false;
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

  while( line_stream )
  {
    if (!getline( line_stream, header_instance, ',' )) break;
    header.push_back(header_instance);
  }

  return header;
}

void avito_data::open_traindata_stream(void)
{
  traindata_impression.data_stream.open((char*)data_parameters.traindata_path.c_str(), std::ios::in);
  assert(traindata_impression.data_stream);
  traindata_impression.header = this->read_header(traindata_impression.data_stream);
  unsigned int size = data_parameters.hash_list.size();

  //traindata_impression.line.resize(traindata_impression.header.size() - MISSING_FEATURES - 1);
  traindata_impression.line.resize(size);
  traindata_impression.is_open = true;
  if(DEBUG) std::cout << "open_traindata_stream: size: " << size << std::endl;
}

void avito_data::open_testdata_stream(void)
{
  testdata_impression.data_stream.open((char*)data_parameters.testdata_path.c_str(), std::ios::in);
  assert(testdata_impression.data_stream);
  testdata_impression.header = this->read_header(testdata_impression.data_stream);
  unsigned int size = data_parameters.hash_list.size();

  //testdata_impression.line.resize(testdata_impression.header.size() - MISSING_FEATURES);
  testdata_impression.line.resize(size);
  testdata_impression.is_open = true;
  if(DEBUG) std::cout << "open_testdata_stream: size: " << size << std::endl;
}

void avito_data::open_holdout_traindata_stream(void)
{
  holdout_traindata_impression.data_stream.open((char*)data_parameters.holdout_traindata_path.c_str(), std::ios::in);
  assert(holdout_traindata_impression.data_stream);
  holdout_traindata_impression.header = this->read_header(holdout_traindata_impression.data_stream);
  unsigned int size = data_parameters.hash_list.size();

  //holdout_traindata_impression.line.resize(holdout_traindata_impression.header.size() - MISSING_FEATURES - 1);
  holdout_traindata_impression.line.resize(size);
  holdout_traindata_impression.is_open = true;
  if(DEBUG) std::cout << "open_holdout_traindata_stream: size: " << size << std::endl;
}

void avito_data::open_holdout_testdata_stream(void)
{
  holdout_testdata_impression.data_stream.open((char*)data_parameters.holdout_testdata_path.c_str(), std::ios::in);
  assert(holdout_testdata_impression.data_stream);
  holdout_testdata_impression.header = this->read_header(holdout_testdata_impression.data_stream);
   
  //holdout_testdata_impression.line.resize(holdout_testdata_impression.header.size() - MISSING_FEATURES);
  unsigned int size = data_parameters.hash_list.size();

  holdout_testdata_impression.line.resize(size);
  holdout_testdata_impression.is_open = true;
  if(DEBUG) std::cout << "open_holdout_testdata_stream: size: " << size << std::endl;
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
  
  if(DEBUG) std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
  for(;; count++, idx++){

    /* read token from line until new line begins */
    if (!getline( line_stream, feature, ',' )) break;

    if(data_impression.header[count] == "ID")
    {
      if(DEBUG) std::cout << "ID " << feature << " " << std::endl;
      data_impression.id = feature.c_str();
      idx--;
    }
    else if(data_impression.header[count] == "IsClick")
    {
      if(DEBUG) std::cout << "IsClick " << feature << " " << std::endl;
      data_impression.label = atoi(feature.c_str());
      idx--;
    }
    else if(!data_parameters.hash_list.count(data_impression.header[count]))
    {
      //std::cout << data_impression.header[count] << " skip idx "  << idx << " count " << count << std::endl;
      idx--;
    }
    else
    {
      std::string key = data_impression.header[count] + "_" + feature.c_str();

      // Here the hash-function will be applied
      boost::hash<std::string> string_hash;

      unsigned long int index;
      if ( data_parameters.useDefaultHash ) {
	index = string_hash(key) % data_parameters.default_hash;
	//std::cout << idx << " " << data_impression.line.size() << std::endl;
      }
      else {
	index = string_hash(key) % data_parameters.hash_list[data_impression.header[count]] + stride;
	stride += data_parameters.hash_list[data_impression.header[count]];
      }
      
      data_impression.line[idx].field_index = idx;
      data_impression.line[idx].index = index;
      data_impression.line[idx].value = 1.0f;
      if(DEBUG) std::cout << data_impression.header[count] << " " << feature << " index: " << index << " field " << idx << std::endl;
    }

  }
  if(DEBUG) std::cout << "-----------------------------------" << std::endl;

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

void avito_data::showInfo(void) const
{
  printf("############## DATA INFO ##################\n");
  printf("# Train-data path:          	%s\n", data_parameters.traindata_path.c_str());
  printf("# Test-data path: 		%s\n", data_parameters.testdata_path.c_str());
  printf("# Holdout Train-data path: 	%s\n", data_parameters.holdout_traindata_path.c_str());
  printf("# Holdout Test-data path: 	%s\n", data_parameters.holdout_testdata_path.c_str());
  printf("# Use default_hash: 		%s\n", data_parameters.useDefaultHash ? "True" : "False");
  printf("# Default hash: 		%lu\n", data_parameters.default_hash);
  printf("############## END DATA INFO ##################\n");
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
