#include "ffm.hpp"

float* FFM::malloc_aligned_float(long long size)
{
    void *ptr;

#ifdef _WIN32
    ptr = _aligned_malloc(size*sizeof(float), kALIGNByte);
    if(ptr == nullptr)
        throw std::bad_alloc();
#else
    int status = posix_memalign(&ptr, kALIGNByte, size*sizeof(float));
    if(status != 0)
        throw std::bad_alloc();
#endif
    
    return (float*)ptr;
}

float FFM::wTx(const ffm_node *instance, const unsigned int & size, float kappa, float eta, float lambda, bool do_update)
{
  long long align0 = (long long)parameters.num_factors*2;
  long long align1 = (long long)num_fields*align0;

  __m128 XMMkappa = _mm_set1_ps(kappa);
  __m128 XMMeta = _mm_set1_ps(eta);
  __m128 XMMlambda = _mm_set1_ps(lambda);

  __m128 XMMt = _mm_setzero_ps();

  for( unsigned int n1 = 0; n1 < size; n1++ )
  {
    int j1 = instance[n1].index;
    int f1 = instance[n1].field_index;
    float v1 = instance[n1].value;
    if(j1 >= num_features || f1 >= num_fields)
	continue;

    for( unsigned int n2 = n1 + 1; n2 < size; n2++ )
    {
      int j2 = instance[n2].index;
      int f2 = instance[n2].field_index;
      float v2 = instance[n2].value;
      if(j2 >= num_features || f2 >= num_fields)
	  continue;

      float *w1 = W + j1*align1 + f2*align0;
      float *w2 = W + j2*align1 + f1*align0;

      __m128 XMMv = _mm_set1_ps(v1*v2);

      if(do_update)
      {
	__m128 XMMkappav = _mm_mul_ps(XMMkappa, XMMv);

	float *wg1 = w1 + parameters.num_factors;
	float *wg2 = w2 + parameters.num_factors;
	for(int d = 0; d < parameters.num_factors; d += 4)
	{
	  __m128 XMMw1 = _mm_load_ps(w1+d);
	  __m128 XMMw2 = _mm_load_ps(w2+d);

	  __m128 XMMwg1 = _mm_load_ps(wg1+d);
	  __m128 XMMwg2 = _mm_load_ps(wg2+d);

	  __m128 XMMg1 = _mm_add_ps(_mm_mul_ps(XMMlambda, XMMw1), _mm_mul_ps(XMMkappav, XMMw2));
	  __m128 XMMg2 = _mm_add_ps(_mm_mul_ps(XMMlambda, XMMw2), _mm_mul_ps(XMMkappav, XMMw1));

	  XMMwg1 = _mm_add_ps(XMMwg1, _mm_mul_ps(XMMg1, XMMg1));
	  XMMwg2 = _mm_add_ps(XMMwg2, _mm_mul_ps(XMMg2, XMMg2));

	  XMMw1 = _mm_sub_ps(XMMw1, _mm_mul_ps(XMMeta, _mm_mul_ps(_mm_rsqrt_ps(XMMwg1), XMMg1)));
	  XMMw2 = _mm_sub_ps(XMMw2, _mm_mul_ps(XMMeta, _mm_mul_ps(_mm_rsqrt_ps(XMMwg2), XMMg2)));

	  _mm_store_ps(w1+d, XMMw1);
	  _mm_store_ps(w2+d, XMMw2);

	  _mm_store_ps(wg1+d, XMMwg1);
	  _mm_store_ps(wg2+d, XMMwg2);
	}
      }
      else
      {
	for(int d = 0; d < parameters.num_factors; d += 4)
	{
	  __m128  XMMw1 = _mm_load_ps(w1+d);
	  __m128  XMMw2 = _mm_load_ps(w2+d);

	  XMMt = _mm_add_ps(XMMt, _mm_mul_ps(_mm_mul_ps(XMMw1, XMMw2), XMMv));
	}
      }
    }
  }

  if(do_update)
      return 0;

  XMMt = _mm_hadd_ps(XMMt, XMMt);
  XMMt = _mm_hadd_ps(XMMt, XMMt);
  float t;
  _mm_store_ss(&t, XMMt);

  return t;
}

FFM::FFM(ffm_parameter inParameters, const unsigned int & numFeatures) : parameters(inParameters), num_features(0), num_fields(0), p(0.0), init_coeff(0.0), diff(0.0)
{
  init_coeff = 0.5/sqrt(parameters.num_factors);  
  W = nullptr;
}

FFM::FFM(const std::string & conf_path) : num_features(0), num_fields(0), p(0.0), init_coeff(0.0), diff(0.0)
{
  W = nullptr;
  read_conf(conf_path);
}

void FFM::read_conf(const std::string &conf_path)
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
  parameters.l1_lambda_v = atof(token.c_str());

  line_stream.str(std::string());
  line_stream.clear();
  
  if(debug) std::cout << "l1_lambda_v:" << token << " " <<  parameters.l1_lambda_v << std::endl;

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
  num_features = atoi(token.c_str());

  line_stream.str(std::string());
  line_stream.clear();
  
  if(debug) std::cout << "num_features:" << token << " " <<  num_features << std::endl;

}

void FFM::avito_init_params(const std::string & data_conf_path)
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
    if(debug) std::cout << "Default hash: " << token << std::endl;

    bool useDefaultHash = ( token != "false" ) ? true : false;
    if ( useDefaultHash ){
      num_features = atoi(token.c_str());
    }

    getline(config_stream, line);
    getline(config_stream, line);

    unsigned int feature_count = 0;
    while(getline(config_stream, line)){
      std::istringstream hash_stream(line);
      getline( hash_stream, header, ':' );
      getline( hash_stream, token, ':' );
      
      num_fields++;
      if ( !useDefaultHash ) {
	num_features += atoi(token.c_str());
      }
      if(debug) std::cout << header << ": " << token << " Current inputDim: " << num_features << std::endl;
      feature_count++;
    }

    if(debug) std::cout << "Num-features: " << num_features << std::endl;
    
    initParams();    
}

FFM::~FFM(void) 
{
  free(W);
  W = nullptr;
}

void FFM::initParams(void)
{ 
  int k_aligned = (int)ceil((double)parameters.num_factors/kALIGN)*kALIGN;
  parameters.num_factors = k_aligned;
    
  try
  {
    W = malloc_aligned_float((long long)num_features*num_fields*k_aligned*2);
  }
  catch(std::bad_alloc const &e)
  {
    throw;
  }

  init_coeff = 0.5/sqrt(parameters.num_factors);
  float *w = W;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  for(int j = 0; j < num_features; j++)
  {
    for(int f = 0; f < num_fields; f++)
    {
      for(int d = 0; d < parameters.num_factors; d++, w++)
	  *w = init_coeff*distribution(generator);
      for(int d = parameters.num_factors; d < k_aligned; d++, w++)
	  *w = 0;
      for(int d = k_aligned; d < 2*k_aligned; d++, w++)
	  *w = 1;
    }
  }
}

float FFM::getPrediction(void) const
{
  return p;
}

const ffm_parameter & FFM::getParameters(void) const
{
  return parameters;
}

void FFM::saveModel(std::string outputDir) const
{
  std::ofstream output_stream((char*)(outputDir +"bw.csv").c_str(), std::ios::out);
  for ( int i = 0; i < num_features*num_fields*parameters.num_factors*2; i++ )
  {
    output_stream << W[i] << std::endl;
  }
  output_stream.close();
}

void FFM::loadModel(std::string inputDir)
{
  std::ifstream input_stream((char*)(inputDir + "bw.csv").c_str(), std::ios::out);
  std::string tmp;
  int count = 0;
  while ( input_stream >> tmp )
  {
    W[count] = atof(tmp.c_str());
    count++;
  }
  input_stream.close();
}

void FFM::showInfo(void) const
{
  printf("################# FFM INFO #####################\n");
  printf("# Number of latent factors: %u\n", parameters.num_factors);
  printf("# Stepsize: %f\n", parameters.stepsize);
  printf("# Stepsize decay: %f\n", parameters.stepsize_decay);
  printf("# l1_lambda_v: %f\n", parameters.l1_lambda_v);
  printf("# l2_lambda_v: %f\n", parameters.l2_lambda_v);
  printf("# Number of epochs: %u\n", parameters.epochs);
  printf("# Number of features: %u\n", num_features);
  printf("# Number of fields: %u\n", num_fields);
  printf("# Parameter init_coeff: %f\n", init_coeff);
  printf("################# END FFM INFO #################\n");
}

void FFM::predict(const std::vector< ffm_node > & instance)
{
  p = wTx(&instance[0], instance.size());
  p =  1.0 / (1.0 + exp(-std::max(std::min(p , float(15.0)), float(-15.))));
}

void FFM::update(const std::vector< ffm_node > & instance, const unsigned int & y)
{
  diff = p - static_cast<float>(y);
  wTx(&instance[0], instance.size(), diff, parameters.stepsize, parameters.l1_lambda_v, true);
}
