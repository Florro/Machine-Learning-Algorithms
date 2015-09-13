#ifndef FFM_HPP
#define FFM_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <new>
#include <memory>
#include <random>
#include <stdexcept>
#include <cstring>
#include <pmmintrin.h>

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sstream>

#include <string>
#include <string.h>
#include <sys/stat.h>

#if defined USEOMP
#include <omp.h>
#endif


int const kALIGNByte = 16;
int const kALIGN = kALIGNByte/sizeof(float);

/* Sparse representation of feature */
struct ffm_node{
  /* Index of the feature field */
  unsigned int field_index;
  /* Index of the feature */
  unsigned int index;
  /* Value of the feature at index */
  float value;
};

/* Hyper-parameter set for the classifier to use */
struct ffm_parameter{

  unsigned int epochs;
  /* Learning rate and learning-rate-decay */
  float stepsize;
  float stepsize_decay;

  /* L1-norm weighting parameters */
  float l1_lambda_v;

  /* L2-norm weighting parameters */
  float l2_lambda_v;

  /* Number of latent factors */
  int num_factors;
};


class FFM{
public:
  FFM(const std::string & conf_path);
  FFM(ffm_parameter inParameters, const unsigned int & numFeatures);
  virtual ~FFM(void);
  
  void initParams(void);
  void avito_init_params(const std::string & data_conf_path);
  void saveModel(std::string outputDir) const;
  void loadModel(std::string outputDir);
  void showInfo(void) const;
  
  float getPrediction(void) const;
  const ffm_parameter & getParameters(void) const;
  
  void predict(const std::vector< ffm_node > & instance);
  void update(const std::vector< ffm_node > & instance, const unsigned int & y);

private:
  
  /* Used Hyper-parameter-set */
  ffm_parameter parameters;
  
  /* Number of Features */
  int num_features;
  
  /* Number of data-fields */
  int num_fields;
  
  /* Current prediction */
  float p;
  
  /* Scaling factor for random initialization */
  float init_coeff;
  
  /* Current external error*/
  float diff;
  
  /* parameters */
  float* W;
  
  void read_conf(const std::string &conf_path);
  
  float* malloc_aligned_float(long long size);
  float wTx(const ffm_node *instance, const unsigned int & size, float kappa=0, float eta=0, float lambda=0, bool do_update=false);
};

#endif