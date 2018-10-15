#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <Eigen/Dense>
#include <valarray>
#include <omp.h>


using namespace Eigen;
using namespace std;

#ifndef  Neural_Network_H
#define Neural_Network_H

class Neural_Network
{
public:
	// Variables
	vector<int> Network_shape;
	// Methoden
	Neural_Network(vector<int> Network_shape, string activation_func);
	void print();
	int ff(MatrixXd input);
	MatrixXd read_txt(const string file, int rows, int cols) const;
	void train(MatrixXd input, MatrixXd output, double learning_rate, int epochs, bool learning_rate_adaption, MatrixXd input_test, MatrixXd output_test);
	double test(MatrixXd input, MatrixXd output);


private:
	// Variables
	vector<MatrixXd, aligned_allocator<MatrixXd>> weights;
	vector<MatrixXd, aligned_allocator<MatrixXd>> bias;
	vector<MatrixXd, aligned_allocator<MatrixXd>> neurons;
	vector<MatrixXd, aligned_allocator<MatrixXd>> neurons_no_activation;
	function<double(double)> activation;
	function<double(double)> activation_ddx;
	function<double(double)> activation_inv;
	// Methods
	MatrixXd feed_forward(MatrixXd input);
	MatrixXd feed_forward_train(MatrixXd input);
	void single_training(MatrixXd input, int output, double learning_rate);
	MatrixXd calc_error_vector(int output);
	vector<MatrixXd> calc_delta(MatrixXd error_vector);
	void update_weights(vector<MatrixXd> delta, double learning_rate);
	void update_bias(vector<MatrixXd> delta, double learning_rate);
};

inline Neural_Network::Neural_Network(vector<int> Network_shape, string activation_func)
{	
	this->Network_shape = Network_shape;
	for (int i = 0; i < Network_shape.size(); i++) {
		if (i < Network_shape.size() - 1) {
			weights.push_back(sqrt(200 / Network_shape[i]) * MatrixXd::Random(Network_shape[i + 1], Network_shape[i]));
			bias.push_back(sqrt(200 / Network_shape[i]) * MatrixXd::Random(Network_shape[i + 1], 1));
			neurons.push_back(MatrixXd::Constant(Network_shape[i], 1, 0));
			neurons_no_activation.push_back(MatrixXd::Constant(Network_shape[i], 1, 0));
		}
		else
		{
			neurons.push_back(MatrixXd::Constant(Network_shape[i], 1, 0));
			neurons_no_activation.push_back(MatrixXd::Constant(Network_shape[i], 1, 0));
		}
	}
	string activation_name;
	if (activation_func == "relu")
	{
		activation = [](double x) { if (x < 0) { return 0.01*x; } else { return x; }; };
		activation_ddx = [](double x) { if (x < 0) { return 0.01; } else { return 1.0; }; };
		activation_inv = [](double x) { if (x < 0) { return 100 * x; } else { return x; }; };
		activation_name = "relu";
	}
	else if (activation_func == "sigmoid")
	{
		activation = [](double x) {return 1 / (1 + exp(-x)); };
		activation_ddx = [](double x) {return (1 / (1 + exp(-x))) * (1 - (1 / (1 + exp(-x)))); };
		activation_inv = [](double x) {return 2 * atanh(2 * x - 1); };
		activation_name = "sigmoid";
	}
	else if (activation_func == "identity")
	{
		activation = [](double x) {return x; };
		activation_ddx = [](double x) {return 1.0; };
		activation_inv = [](double x) {return x; };
		activation_name = "identity";
	}
	else if (activation_func == "softplus")
	{
		activation = [](double x) {return log(1 + exp(x)); };
		activation_ddx = [](double x) {return 1.0; };
		activation_inv = [](double x) {return 1 / (1 + exp(-x)); };
		activation_name = "softplus";
	}
	else
	{
		activation = [](double x) {return 1 / (1 + exp(-x)); };
		activation_ddx = [](double x) {return (exp(-x) / ((1 + exp(-x))*(1 + exp(-x)))); };
		activation_inv = [](double x) {return 2 * atanh(2 * x - 1); };
		activation_name = "sigmoid (default)";
	}
	cout << "Network activation: " << activation_name << endl;
	cout << "Newtork Shape: ";
	for (int i = 0; i < Network_shape.size(); i++) {
		cout << Network_shape[i] << " ";
	}
	cout << endl;
}

inline void Neural_Network::print() {

	cout << "Network shape: ";
	for (int i = 0; i < Network_shape.size(); i++) 
	{
		cout << Network_shape[i] << ", ";
	}

	cout << endl << "________" << endl;

	cout << "Weights:" << endl;
	if (weights[0].size() < 300) {
		for (int i = 0; i < Network_shape.size() - 1; i++) {
			cout << weights[i] << endl << endl;
		}
	}
	else {
		for (int i = 0; i < Network_shape.size() - 1; i++) 
		{
			cout << weights[i].rows() << " x " << weights[i].cols() << endl;
		}
	}

	cout << "________" << endl;

	cout << "Biases:" << endl;
	if (bias[0].size() < 1000) {
		for (int i = 0; i < Network_shape.size() - 1; i++) {
			cout << bias[i] << endl << endl;
		}
	}
	else {
		for (int i = 0; i < Network_shape.size() - 1; i++) 
		{
			cout << bias[i].rows() << " x " << bias[i].cols() << endl;
		}
	}
}

inline int Neural_Network::ff(MatrixXd input) {
	Neural_Network::feed_forward(input);
	int result;
	for (int i = 0; i < Network_shape[Network_shape.size() - 1]; i++) {
		if (i == 0) {
			result = i;
		}
		else {
			if (neurons[neurons.size() - 1](result, 0) < neurons[neurons.size() - 1](i, 0)) {
				result = i;
			}
		}
	}
	return result;
}

inline MatrixXd Neural_Network::feed_forward(MatrixXd input) {
	neurons[0] = input;
	neurons_no_activation[0] = neurons[0];
	for (int i = 0; i < Network_shape.size() - 1; i++) {
		if (i < Network_shape.size() - 2) {
			neurons[i + 1] = (weights[i] * neurons[i] + bias[i]).unaryExpr(activation);
		}
		else {
			neurons[i + 1] = (weights[i] * neurons[i] + bias[i]).unaryExpr(activation);
		}
	}
	return neurons[neurons.size() - 1];
}

inline MatrixXd Neural_Network::feed_forward_train(MatrixXd input) {
	neurons_no_activation[0] = input;
	neurons[0] = neurons_no_activation[0];
	for (int i = 0; i < Network_shape.size() - 1; i++) {
		if (i < Network_shape.size() - 2) {
			neurons_no_activation[i + 1] = weights[i] * neurons[i] + bias[i];
			neurons[i + 1] = neurons_no_activation[i + 1].unaryExpr(activation);
		}
		else {
			neurons_no_activation[i + 1] = weights[i] * neurons[i] + bias[i];
			neurons[i + 1] = neurons_no_activation[i + 1].unaryExpr(activation);
		}
	}
	return neurons[neurons.size() - 1];
}

inline MatrixXd Neural_Network::read_txt(string file, int rows, int cols) const
{
	ifstream in(file);
	string line;

	int row = 0;
	int col = 0;

	MatrixXd matrix = MatrixXd(rows, cols);

	if (in.is_open()) {
		while (getline(in, line))
		{
			char *ptr = (char *)line.c_str();
			int len = line.length();

			col = 0;

			char *start = ptr;
			for (int i = 0; i < len; i++) {
				if (ptr[i] == ',') {
					matrix(row, col++) = atof(start);
					start = ptr + i + 1;
				}
			}
			matrix(row, col) = atof(start);
			row++;
		}
		in.close();
	}
	else
	{
		throw invalid_argument("Fehler beim oeffnen");
	}
	return matrix;
}

inline void Neural_Network::train(MatrixXd input, MatrixXd output, double learning_rate, int epochs, bool learning_rate_adaption, MatrixXd input_test, MatrixXd output_test) {
	cout << "Training start" << endl;
	double test = 0;
	double test_new = 0;
	for (int epoch = 0; epoch < epochs; epoch++)
	{	
		if(learning_rate_adaption)
		{
			learning_rate = learning_rate * exp(-epoch / ( 1.1 * epochs));
		}
		if(epoch % 50 == 0)
		{
			Neural_Network::print();
		}
		if(epoch % 10 == 0)
		{
			cout << "Training ";
			test_new = Neural_Network::test(input, output);
			Neural_Network::test(input_test, output_test);
		}
		if(test_new > test)
		{
			test = test_new;
		}
		else if(test_new <= test)
		{
			test = test_new;
			learning_rate = 1.25 * learning_rate;
		}
		cout << "epoch " << epoch + 1 << endl;
		for (int i = 0; i < input.rows(); i++) {
			single_training(input.row(i).transpose(), (int) output(i, 0), learning_rate);
		}
	}
	
}

inline void Neural_Network::single_training(MatrixXd input, int output, double learning_rate) {
	feed_forward_train(input);
	MatrixXd error_vector = calc_error_vector(output);
	vector<MatrixXd> delta = calc_delta(error_vector);
	update_weights(delta, learning_rate);
	update_bias(delta, learning_rate);
}

inline vector<MatrixXd> Neural_Network::calc_delta(MatrixXd error_vector) {
	vector<MatrixXd> delta;
	for (int i = 0; i < Network_shape.size() - 1; i++) {
		delta.push_back(MatrixXd::Constant(Network_shape[i + 1], 1, 0.0));
	}
	for (int i = Network_shape.size() - 1; i > 0; i--) {
		if (i == Network_shape.size() - 1) {
			delta[i - 1] = (neurons_no_activation[i]).unaryExpr(activation_ddx).cwiseProduct(error_vector);
		}
		else
		{	
			for (int j = 0; j < delta[i - 1].size(); j++)
			{
				double sum = 0.0;
				for (int k = 0; k < delta[i].size(); k++)
				{
					sum = sum + delta[i](k, 0) * weights[i](k, j);
				}
				delta[i - 1](j, 0) = sum * activation_ddx(neurons_no_activation[i](j, 0));
			}
		}
	}
	return delta;
}

inline MatrixXd Neural_Network::calc_error_vector(int output) {
	MatrixXd vector = MatrixXd::Constant(Network_shape[Network_shape.size() - 1], 1, 0.0);
	vector(output, 0) = 1.0;
	vector = neurons[neurons.size() - 1] - vector;
	return vector;
}

inline void Neural_Network::update_weights(vector<MatrixXd> delta, double learning_rate)
{
	vector<MatrixXd, aligned_allocator<MatrixXd>> delta_weights;
	for (int i = 0; i < Network_shape.size() - 1; i++) {
		delta_weights.push_back(MatrixXd::Constant(Network_shape[i + 1], Network_shape[i], 0.0));
	}
	for (int i = 0; i < Network_shape.size() - 1; i++)
	{
		for (int j = 0; j < Network_shape[i]; j++)
		{	
			for (int k = 0; k < Network_shape[i + 1]; k++)
			{
				delta_weights[i](k, j) = learning_rate * delta[i](k, 0) * neurons[i](j, 0);
			}
		}
	}
	for (int i = 0; i < Network_shape.size() - 1; i++)
	{	
		weights[i] = weights[i] - delta_weights[i];
	}
}

inline void Neural_Network::update_bias(vector<MatrixXd> delta, double learning_rate)
{
	for (int i = 0; i < bias.size(); i++)
	{
		bias[i] = bias[i] - (learning_rate * delta[i]);
	}
}

inline double Neural_Network::test(MatrixXd input, MatrixXd output)
{
	double counter = 0;
	for (int i = 0; i < input.rows(); i++)
	{
		if(Neural_Network::ff(input.row(i).transpose()) == output(i, 0))
		{
			counter++;
		}
	}
	cout << "Error: " << counter / input.rows() << endl;
	return counter / input.rows();
}



#endif