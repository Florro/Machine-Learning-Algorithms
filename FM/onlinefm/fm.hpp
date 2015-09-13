#ifndef FM_HPP
#define FM_HPP

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include <string>
#include <string.h>

/* Sparse representation of feature */
struct fm_node{
  /* Index of the feature */
  unsigned int index;
  /* Value of the feature at index */
  float value;
};

/* Hyper-parameter set for the classifier to use */
struct fm_parameter{

  unsigned int epochs;
  /* Learning rate and learning-rate-decay */
  float stepsize;
  float stepsize_decay;

  /* L1-norm weighting parameters */
  float l1_lambda_b;
  float l1_lambda_w;
  float l1_lambda_v;

  /* L2-norm weighting parameters */
  float l2_lambda_b;
  float l2_lambda_w;
  float l2_lambda_v;

  /* Number of latent factors */
  unsigned int num_factors;

  /* Use bias */
  bool use_b;
  /* Use linear combinations */
  bool use_w;
};


class FM{
public:
  FM(const std::string & conf_path);
  FM(fm_parameter inParameters, const unsigned int & numFeatures);
  virtual ~FM(void);
  
  void initParams(void);
  void avito_init_params(const std::string & data_conf_path);
  void saveModel(std::string outputDir) const;
  void loadModel(std::string outputDir);
  void showInfo(void) const;
  
  float getPrediction(void) const;
  const fm_parameter & getParameters(void) const;
  
  void predict(const std::vector< fm_node > & instance);
  void update(const std::vector< fm_node > & instance, const unsigned int & y);
  
  const float & get_b(void) const;
  const std::vector< float > & get_w(void) const;
  const std::vector< float > & get_v(void) const;

private:
  
  /* Used Hyper-parameter-set */
  fm_parameter parameters;
  
  /* Number of Features */
  unsigned int D;
  /* Bias */
  float b; 
  /* Linear weights */
  std::vector< float > w;
  /* Latent factors */
  std::vector< float > v;
  
  float p;
  
  float init_coeff;
  
  float diff;
  
  std::vector< float > v_sums;
  std::vector< float > v_sums_sq;
  
  float l1(const float & w);
  void read_conf(const std::string &conf_path);
};

#endif