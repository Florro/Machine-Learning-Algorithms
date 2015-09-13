/*
 * train.cpp
 *
 *  Created on: Jun 8, 2015
 *      Author: florian
 */

#ifndef TRAIN_CPP_
#define TRAIN_CPP_

#include "data.hpp"

#define TRAIN_FILE_SIZE 190157735
#define TEST_FILE_SIZE 7816362

using namespace std;

void predict(ftrl* ftrl_classifier, avito_data* data, string prediction_path, bool is_cv)
{

	std::ofstream prediction_stream((char*)prediction_path.c_str(), ios::out);
	assert(prediction_stream);

	prediction_stream << "ID,IsClick" << std::endl;

	ftrl_uint32 test_count = 0;
	ftrl_float loss = 0.0;
	ftrl_float acc = 0.0;

	bool* is_open;
	avito_impression* test_impression;
	if(is_cv){
		data->open_holdout_testdata_stream();
		is_open = &data->holdout_testdata_impression.is_open;
		test_impression = &data->holdout_testdata_impression;
	}
	else{
		data->open_testdata_stream();
		is_open = &data->testdata_impression.is_open;
		test_impression = &data->testdata_impression;
	}

	while (*is_open){

		if(is_cv){
			data->read_holdout_testdata_line();
			if(!*is_open) break;
		}
		else{
			data->read_testdata_line();
			if(!*is_open) break;
		}

		ftrl_classifier->predict(test_impression->line);

		if(is_cv){
			loss += logloss(ftrl_classifier->p, test_impression->label);
			acc += accuracy(ftrl_classifier->p, test_impression->label);
		}


		prediction_stream << test_impression->id << "," << ftrl_classifier->p << std::endl;

		test_count++;
	}
	if(is_cv){
		printf("(Holdout-Test-loss: %f, Holdout-Test-Acc: %f %%)\n", loss / test_count, 100 * acc / test_count);
	}

	prediction_stream.close();
}

void train(ftrl* ftrl_classifier, string data_conf_path, string prediction_path, bool is_cv, ftrl_float sub_sample_rate){

	for(ftrl_uint16 e = 0; e < ftrl_classifier->task.epochs; e++){

		avito_data* data = new avito_data(data_conf_path);

		bool* is_open;
		avito_impression* train_impression;
		if(is_cv){
			data->open_holdout_traindata_stream();
			is_open = &data->holdout_traindata_impression.is_open;
			train_impression = &data->holdout_traindata_impression;
		}
		else{
			data->open_traindata_stream();
			is_open = &data->traindata_impression.is_open;
			train_impression = &data->traindata_impression;
		}

		//ftrl_uint32 train_count = 1;
		ftrl_float train_loss = 0.0;
		ftrl_float train_acc = 0.0;
		data->read_holdout_traindata_line();
#pragma omp parallel for schedule(static) reduction(+: train_loss) 
		for ( int train_count = 1; train_count < 200000000; train_count++ ){
		//while(*is_open){

			if(is_cv){
				//data->read_holdout_traindata_line();
				//if(!*is_open) break;
			}
			else{
				//data->read_traindata_line();
				//if(!*is_open) break;
			}

			//if(drand48() < sub_sample_rate){

				ftrl_classifier->predict(train_impression->line);

				train_loss += 1;//logloss(ftrl_classifier->p, train_impression->label);
				//train_acc += accuracy(ftrl_classifier->p, train_impression->label);

				//if(isinf(train_loss) or isnan(train_loss)){
					//std::cout << ftrl_classifier->p << "  " << train_impression->label << "  " << train_loss << std::endl;
				//}


				ftrl_classifier->update(train_impression->line, train_impression->label);
				//std::cout << train_impression->line[0].value << "   " << train_impression->line.back().value << std::endl;

			//}

			if(train_count % 10000000 == 0 and train_count > 0)
			{
				printf("Epoch: %i, Samples Seen: %f %% (Train-loss: %f, Train-Acc: %f %%)\n", e, 100. * ftrl_float(train_count)/(TRAIN_FILE_SIZE), train_loss / train_count, 100 * train_acc / train_count);
			}


			//if((train_count % 50000000 == 0 or train_count == 0) and is_cv )
			//{
			//	predict(ftrl_classifier, data, prediction_path, is_cv);
			//}

			//train_count++;

		}

		printf("###############\n");
		printf("Epoch: %i finished\n", e);
		//printf("(Train-loss: %f, Train-Acc: %f %%)\n", train_loss / train_count, 100 * train_acc / train_count);
		printf("###############\n");

		predict(ftrl_classifier, data, prediction_path, is_cv);

		delete data;

	}

}


void holdout(string model, ftrl_float sub_sample_rate){

	printf("Holdout validation started...\n");

	string model_path = "/home/florian/Data/projects/AVITO/models/" + model + "/";
	string holdout_prediction_path = model_path + "holdout_prediction.csv";

	string data_conf_path = model_path + "data_conf";
	string ftrl_conf_path = model_path + "ftrl_conf";


	ftrl* my_ftrl = new ftrl(ftrl_conf_path);
	my_ftrl->avito_init_params(data_conf_path);

	train(my_ftrl, data_conf_path, holdout_prediction_path, true, sub_sample_rate);

	delete my_ftrl;
}

void submit(string model, ftrl_float sub_sample_rate){

	printf("Submission procedure started...\n");

	string model_path = "/home/florian/Data/projects/AVITO/models/" + model + "/";
	string submission_path = model_path + "submission.csv";

	string data_conf_path = model_path + "data_conf";
	string ftrl_conf_path = model_path + "ftrl_conf";


	ftrl* my_ftrl = new ftrl(ftrl_conf_path);
	my_ftrl->avito_init_params(data_conf_path);

	train(my_ftrl, data_conf_path, submission_path, false, sub_sample_rate);

	my_ftrl->saveModel(model_path + "parameters.csv");

	delete my_ftrl;
}

int main(void) {

//#pragma omp parallel
//	printf("Hello OpenMP\n");

	ftrl_float sub_sample_rate = 1.0;

	holdout("ftrl11", sub_sample_rate);

	std::cout << "###############################################" << std::endl;

	submit("ftrl11", sub_sample_rate);

	printf("Exit Success");
	return 0;
}



#endif /* TRAIN_CPP_ */
