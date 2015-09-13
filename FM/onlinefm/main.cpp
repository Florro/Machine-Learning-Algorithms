
#include "fmdata.hpp"

#define TRAIN_FILE_SIZE 190157735
#define TEST_FILE_SIZE 7816362

using namespace std;

float logloss(const float & p, const unsigned int & y)
{
  return (y == 1) ? -log(p) : -log(1.0 - p);
}

float accuracy(const float & p, const unsigned int & y)
{
  unsigned int pred = (p <= 0.5) ? 0 : 1;
  return (pred == y) ? 1 : 0;
}

void predict(FM* fmClassifier, avito_data* data, string prediction_path, bool is_cv)
{
  std::ofstream prediction_stream((char*)prediction_path.c_str(), std::ios::out);
  assert(prediction_stream);

  prediction_stream << "ID,IsClick" << std::endl;

  unsigned int test_count = 0;
  float loss = 0.0;
  float acc = 0.0;

  bool* is_open;
  avito_impression* test_impression;
  if( is_cv ) {
    data->open_holdout_testdata_stream();
    is_open = &data->holdout_testdata_impression.is_open;
    test_impression = &data->holdout_testdata_impression;
  }
  else {
    data->open_testdata_stream();
    is_open = &data->testdata_impression.is_open;
    test_impression = &data->testdata_impression;
  }
  
  while ( *is_open )
  {
    if( is_cv ) {
      data->read_holdout_testdata_line();
      if(!*is_open) break;
    }
    else {
      data->read_testdata_line();
      if( !*is_open ) break;
    }
    
    fmClassifier->predict(test_impression->line);

    if( is_cv ) {
      loss += logloss(fmClassifier->getPrediction(), test_impression->label);
      acc += accuracy(fmClassifier->getPrediction(), test_impression->label);
    }

    prediction_stream << test_impression->id << "," << fmClassifier->getPrediction() << std::endl;

    test_count++;
  }
  if( is_cv ) {
    printf("(Holdout-Test-loss: %f, Holdout-Test-Acc: %f %%)\n", loss / test_count, 100 * acc / test_count);
  }

  prediction_stream.close();
}

void train(FM* fmClassifier, string data_conf_path, string prediction_path, bool is_cv, float sub_sample_rate){

  for(unsigned int e = 0; e < fmClassifier->getParameters().epochs; e++)
  {
    avito_data* data = new avito_data(data_conf_path);
    
    bool* is_open;
    avito_impression* train_impression;
    if( is_cv ) {
      data->open_holdout_traindata_stream();
      is_open = &data->holdout_traindata_impression.is_open;
      train_impression = &data->holdout_traindata_impression;
    }
    else {
      data->open_traindata_stream();
      is_open = &data->traindata_impression.is_open;
      train_impression = &data->traindata_impression;
    }
    
    unsigned int train_count = 1;
    float train_loss = 0.0;
    float train_acc = 0.0;

    while( *is_open )
    {     
      if( is_cv ) {
	data->read_holdout_traindata_line();
	if(!*is_open) break;
      }
      else {
	data->read_traindata_line();
	if(!*is_open) break;
      }

      if( drand48() < sub_sample_rate ) {

	fmClassifier->predict(train_impression->line);
	
	train_loss += logloss(fmClassifier->getPrediction(), train_impression->label);
	train_acc += accuracy(fmClassifier->getPrediction(), train_impression->label);

	if( isinf(train_loss) or isnan(train_loss) ){
	  std::cout << fmClassifier->getPrediction() << "  " << train_impression->label << "  " << train_loss << std::endl;
	  const std::vector< float> & x = fmClassifier->get_v();
	  for ( auto & xx : train_impression->line)
	  {
	    std::cout << xx.index << " " << xx.value << " " << x[xx.index] << " " << x[0] << " " << x[1] << std::endl;
	  }
	  break;
	}

	fmClassifier->update(train_impression->line, train_impression->label);
	//std::cout << train_impression->line[0].value << "   " << train_impression->line.back().value << std::endl;

      }

      if( train_count % 5000000 == 0 and train_count > 0 ) {
	printf("Epoch: %i, Samples Seen: %f %% (Train-loss: %f, Train-Acc: %f %%)\n", e, 100. * float(train_count)/(TRAIN_FILE_SIZE), train_loss / train_count, 100 * train_acc / train_count);
      }

      if( (train_count % 10000000 == 0 or train_count == 0) and is_cv ) {
	predict(fmClassifier, data, prediction_path, is_cv);
      }

      train_count++;
      
    }

    printf("###############\n");
    printf("Epoch: %i finished\n", e);
    printf("(Train-loss: %f, Train-Acc: %f %%)\n", train_loss / train_count, 100 * train_acc / train_count);
    printf("###############\n");

    predict(fmClassifier, data, prediction_path, is_cv);

    delete data;

  }

}


void holdout(string model, float sub_sample_rate){

	printf("Holdout validation started...\n");

	string model_path = "/home/florian/Data/projects/AVITO/models/" + model + "/";
	string holdout_prediction_path = model_path + "holdout_prediction.csv";

	string data_conf_path = model_path + "data_conf";
	string ftrl_conf_path = model_path + "fm_config";


	FM* fmClassifier = new FM(ftrl_conf_path);
	fmClassifier->avito_init_params(data_conf_path);
	fmClassifier->showInfo();

	train(fmClassifier, data_conf_path, holdout_prediction_path, true, sub_sample_rate);

	delete fmClassifier;
}

void submit(string model, float sub_sample_rate){

	printf("Submission procedure started...\n");

	string model_path = "/home/florian/Data/projects/AVITO/models/" + model + "/";
	string submission_path = model_path + "submission.csv";

	string data_conf_path = model_path + "data_conf";
	string ftrl_conf_path = model_path + "fm_config";


	FM* fmClassifier = new FM(ftrl_conf_path);
	fmClassifier->avito_init_params(data_conf_path);
	fmClassifier->showInfo();

	train(fmClassifier, data_conf_path, submission_path, false, sub_sample_rate);

	fmClassifier->saveModel(model_path);

	delete fmClassifier;
}

int main(void) {

//#pragma omp parallel
//	printf("Hello OpenMP\n");

	float sub_sample_rate = 1.0;

	holdout("fm4", sub_sample_rate);

	std::cout << "###############################################" << std::endl;

	submit("fm4", sub_sample_rate);

	printf("Exit Success");
	return 0;
}

