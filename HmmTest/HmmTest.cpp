#include "pch.h"
#include <iostream>
#include "hmm.h"
#include "HmmTrain.h"

using namespace std;
int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		cout << "参数不足\n";
		exit(0);
	}
	/*
	char *modellist = argv[1];
	char *testFile = argv[2];
	char *resultFile = argv[3];
	*/
	char modellist[] = "modellist.txt";
	char testFile[] = "testing_data1.txt";
	char resultFile[] = "result.txt";
	char trureFile[] = "testing_answer.txt";
	HMM model[5] = { NULL };
	
	//加载HMM
	load_models(modellist, model, 5);
	test(model, testFile, resultFile);
	double accuracy = computeAccuracy(resultFile, trureFile);
	cout << "Accuracy:" << accuracy << "\n";
}