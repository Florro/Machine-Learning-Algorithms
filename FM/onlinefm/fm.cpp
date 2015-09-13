#include "fm.hpp"

FM::FM(fm_parameter inParameters, const unsigned int & numFeatures) : parameters(inParameters), D(numFeatures), b(0.0), diff(0.0)
{
  init_coeff = 0.5/sqrt(parameters.num_factors);  
}

FM::FM(const std::string & conf_path) : b(0.0), diff(0.0)
{
  read_conf(conf_path);
  
  //w.reserve(D);
  //v.reserve(D * parameters.num_factors);
  
  //v_sums.reserve(parameters.num_factors);
  //v_sums_sq.reserve(parameters.num_factors);

  init_coeff = 0.5/sqrt(parameters.num_factors);
  
}

void FM::read_conf(const std::string &conf_path)
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
  parameters.use_b = false;
  if ( token == "true") { 
    parameters.use_b = true;
  };

  line_stream.str(std::string());
  line_stream.clear();

  if(debug) std::cout << "use_b: " << token << " " <<  parameters.use_b << std::endl;

  //Read parameters:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
  parameters.use_w = false;
  if ( token == "true") { 
    parameters.use_w = true;
  };
  line_stream.str(std::string());
  line_stream.clear();

  if(debug) std::cout << "use_w: " << token << " " <<  parameters.use_w << std::endl;

  //Read parameters:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
  parameters.num_factors = atoi(token.c_str());

  line_stream.str(std::string());
  line_stream.clear();

  if(debug) std::cout << "num_factors: " << token << " " <<  parameters.num_factors << std::endl;

  //Read parameters:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
  parameters.stepsize = atof(token.c_str());

  line_stream.str(std::string());
  line_stream.clear();

  if(debug) std::cout << "stepsize: " << token << " " <<  parameters.stepsize << std::endl;

  //Read parameters:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
  parameters.stepsize_decay = atof(token.c_str());

  line_stream.str(std::string());
  line_stream.clear();

  if(debug) std::cout << "stepsize_decay: " << token << " " <<  parameters.stepsize_decay << std::endl;

  //Read parameters:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
  parameters.l1_lambda_b = atof(token.c_str());

  line_stream.str(std::string());
  line_stream.clear();
  
  if(debug) std::cout << "l1_lambda_b:" << token << " " <<  parameters.l1_lambda_b << std::endl;

  //Read parameters:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
  parameters.l1_lambda_w = atof(token.c_str());

  line_stream.str(std::string());
  line_stream.clear();
  
  if(debug) std::cout << "l1_lambda_w:" << token << " " <<  parameters.l1_lambda_w << std::endl;

  //Read parameters:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
  parameters.l1_lambda_v = atof(token.c_str());

  line_stream.str(std::string());
  line_stream.clear();
  
  if(debug) std::cout << "l1_lambda_v:" << token << " " <<  parameters.l1_lambda_v << std::endl;

  //Read parameters:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
  parameters.l2_lambda_b = atof(token.c_str());

  line_stream.str(std::string());
  line_stream.clear();

  if(debug) std::cout << "l2_lambda_b:" << token << " " <<  parameters.l2_lambda_b << std::endl;

  //Read parameters:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
  parameters.l2_lambda_w = atof(token.c_str());

  line_stream.str(std::string());
  line_stream.clear();

  
  if(debug) std::cout << "l2_lambda_w:" << token << " " <<  parameters.l2_lambda_w << std::endl;

  //Read parameters:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
  parameters.l2_lambda_v = atof(token.c_str());

  line_stream.str(std::string());
  line_stream.clear();

  if(debug) std::cout << "l2_lambda_v:" << token << " " <<  parameters.l2_lambda_v << std::endl;

  //Read parameters:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
  parameters.epochs = atoi(token.c_str());

  line_stream.str(std::string());
  line_stream.clear();

  if(debug) std::cout << "epochs:" << token << " " <<  parameters.epochs << std::endl;
  
  //Read parameters:
  getline(config_stream, line);
  line_stream.str(line);
  while(line_stream){	if (!getline( line_stream, token, '=' )) break;	}
  D = atoi(token.c_str());

  line_stream.str(std::string());
  line_stream.clear();
  
  if(debug) std::cout << "D:" << token << " " <<  D << std::endl;

}

void FM::avito_init_params(const std::string & data_conf_path)
{
    bool debug = false;

    std::ifstream config_stream((char*)data_conf_path.c_str(), std::ios::in);

    std::string line;
    std::string token;
    std::string header;

    // Read header:
    for(unsigned int i = 0; i < 8; i++){
      getline(config_stream, line);
    }

    std::istringstream line_stream(line);
    while(line_stream){	if (!getline( line_stream, token, ':' )) break;	}
    if(debug) std::cout << "Poly2: " << token << std::endl;

    bool poly2 = ( token == "true" ) ? true : false;
    //if ( token == "true" )//bool(token.c_str());

    getline(config_stream, line);
    getline(config_stream, line);

    D = 1; // 0 index for CTR-numerical value
    unsigned int feature_count = 0;
    while(getline(config_stream, line)){
      std::istringstream hash_stream(line);
      getline( hash_stream, header, ':' );
      getline( hash_stream, token, ':' );

      D += atoi(token.c_str());

      if(debug) std::cout << header << ": " << token << " Current inputDim: " << D << std::endl;
      feature_count++;
    }

    if(debug) std::cout << "Num-features: " << D << std::endl;
    
    initParams();    
}

FM::~FM(void) {}


void FM::initParams(void)
{ 
  for(unsigned int s = 0; s < D * parameters.num_factors; s++)
  {
    if ( s < parameters.num_factors ) {
      v_sums.push_back(0.0);
      v_sums_sq.push_back(0.0);
    }
    
    if( s < D and parameters.use_w ){
      w.push_back(0.01);//-0.5//init_coeff * drand48()
    }
    
    v.push_back(0.01);//-0.5
  }
}

float FM::getPrediction(void) const
{
  return p;
}

const fm_parameter & FM::getParameters(void) const
{
  return parameters;
}

void FM::saveModel(std::string outputDir) const
{
  std::ofstream w_output_stream((char*)(outputDir +"bw.csv").c_str(), std::ios::out);
  w_output_stream << b << std::endl;
  for( auto & i : w )
  {
    w_output_stream << i << std::endl;
  }
  w_output_stream.close();
  
  std::ofstream v_output_stream((char*)(outputDir +"v.csv").c_str(), std::ios::out);
  for( auto & i : v )
  {
    v_output_stream << i << std::endl;
  }
  v_output_stream.close();
}

void FM::loadModel(std::string inputDir)
{
  std::ifstream wb_input_stream((char*)(inputDir + "bw.csv").c_str(), std::ios::out);
  std::string tmp;
  
  wb_input_stream >> tmp;
  b = atof(tmp.c_str());
  
  unsigned int i = 0;
  while ( wb_input_stream >> tmp)
  {
	  w[i] = atof(tmp.c_str());
	  ++i;
  }
  wb_input_stream.close();
  
  std::ifstream v_input_stream((char*)(inputDir + "v.csv").c_str(), std::ios::out);
  i = 0;
  while ( v_input_stream >> tmp)
  {
	  v[i] = atof(tmp.c_str());
	  ++i;
  }
  v_input_stream.close();
  
}

void FM::showInfo(void) const
{
  printf("################# INFO #####################\n");
  printf("# Use Bias: %s\n", !parameters.use_b ? "True" : "False");
  printf("# Use Linear weights: %s\n", !parameters.use_w ? "True" : "False");
  printf("# w size: %lu\n", w.size());
  printf("# Number of latent factors: %u\n", parameters.num_factors);
  printf("# v size: %lu\n", v.size());
  printf("# Stepsize: %f\n", parameters.stepsize);
  printf("# Stepsize decay: %f\n", parameters.stepsize_decay);
  printf("# l1_lambda_b: %f\n", parameters.l1_lambda_b);
  printf("# l1_lambda_w: %f\n", parameters.l1_lambda_w);
  printf("# l1_lambda_v: %f\n", parameters.l1_lambda_v);
  printf("# l2_lambda_b: %f\n", parameters.l2_lambda_b);
  printf("# l2_lambda_w: %f\n", parameters.l2_lambda_w);
  printf("# l2_lambda_v: %f\n", parameters.l2_lambda_v);
  printf("# Number of epochs: %u\n", parameters.epochs);
  printf("# Number of features: %u\n", D);
  printf("# Parameter init_coeff: %f\n", init_coeff);
  printf("################# END INFO #################\n");
}

void FM::predict(const std::vector< fm_node > & instance)
{
  p = (!parameters.use_b) ? 0.0 : b;
  
  if ( parameters.use_w ) {
    for ( auto & i : instance )
    {
      p += w[i.index] * i.value;
    }
  }
  
  for(unsigned int j = 0; j < parameters.num_factors; j++){
    v_sums[j] = 0.0;
    v_sums_sq[j] = 0.0;
    for ( auto & i : instance )
    {
      unsigned int  idx = j*D + i.index;

      float d = v[idx] * i.value;

      v_sums[j] += d;
      v_sums_sq[j] += d*d;
    }
    p += 0.5 * (v_sums[j]*v_sums[j] - v_sums_sq[j]);
  }
 
  p =  1.0 / (1.0 + exp(-std::max(std::min(p, float(15.0)), float(-15.))));
 
}

void FM::update(const std::vector< fm_node > & instance, const unsigned int & y){

  //diff = -(y - 1./(1. + exp(-p)));
  diff = p - y;

  if( parameters.use_b ) {
    b -= parameters.stepsize * (diff + parameters.l2_lambda_b * b + parameters.l1_lambda_b * l1(b));
  }

  if( parameters.use_w ) {
    for ( auto & n : instance )
    {
      w[n.index] -= parameters.stepsize * (diff * n.value + parameters.l2_lambda_w * w[n.index] + parameters.l1_lambda_w * l1(w[n.index]));
    }
  }

  for( unsigned int j = 0; j < parameters.num_factors; ++j ){
    for ( auto & n : instance )
    {
      //if ( n.index > D ) std::cout << "asdf" << std::endl;
      unsigned int idx = j*D + n.index;
      //std::cout << diff << " " << idx << " " << v[idx] << " " << n.index << " " << n.value << std::endl;

      float grad = (v_sums[j] - v[idx] * n.value) * n.value;
      v[idx] -= parameters.stepsize * (diff * grad + parameters.l2_lambda_v * v[idx] + parameters.l1_lambda_v * l1(v[idx]));
      
      //if (abs(v[idx]) > 10. ) std::cout << v[idx] << " error " << std::endl;
      
    }
  }
}

float FM::l1(const float & w){
  float l1_term = 0.0;
  if( w != 0.0 ) {
    l1_term = (w > 0)? 1.0 : 0.0;
  }
  return l1_term;
}

const float & FM::get_b(void) const
{
  return b;
}
const std::vector< float > & FM::get_w(void) const
{
  return w;
}
const std::vector< float > & FM::get_v(void) const
{
  return v;
}