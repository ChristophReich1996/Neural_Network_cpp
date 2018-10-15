// g++ -O3 -fopenmp -lstdc++ -Ic:\eigen main.cpp
#include <omp.h>
#include <Eigen/Dense>
#include <iostream>
#include "Neural_Network.h"

using namespace Eigen;
using namespace std;

int main(){

	initParallel();
	setNbThreads(4);

	int nthreads = nbThreads();
	cout << "Num of Threads: " << nthreads << endl;

	Neural_Network NN(vector<int> {28 * 28, 64, 32, 10}, "");

	NN.print();

	const string address_test_out = "C:/mnist/mnist_small_test_out.txt";
	MatrixXd matrix_test_out = NN.read_txt(address_test_out, 1004, 1);

	const string address_test_in = "C:/mnist/mnist_small_test_in.txt";
	MatrixXd matrix_test_in = NN.read_txt(address_test_in, 1004, 28*28);

	const string address_train_out = "C:/mnist/mnist_small_train_out.txt";
	MatrixXd matrix_train_out = NN.read_txt(address_train_out, 6006, 1);

	const string address_train_in = "C:/mnist/mnist_small_train_in.txt";
	MatrixXd matrix_train_in = NN.read_txt(address_train_in, 6006, 28*28);

	cout << "Load data successful" << endl;

	system("pause");

	NN.train(matrix_train_in, matrix_train_out, 0.3, 500, true, matrix_test_in, matrix_test_out);
	NN.test(matrix_test_in, matrix_test_out);

	system("pause");

	return 0;
}