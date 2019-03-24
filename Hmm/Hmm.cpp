// Hmm.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include "hmm.h"
#include "HmmTrain.h"

using namespace std;
int main(int argc, char *argv[])
{
	if (argc < 5)
	{
		cout << "参数不足！\n";
	}
	int interation = atoi(argv[1]);
	char *initFile = argv[2];
	char *inputFile = argv[3];
	char *outputFile = argv[4];
	HMM hmm;
	loadHMM(&hmm, initFile);
	dumpHMM(stderr, &hmm);
	train(&hmm, inputFile, outputFile, interation);
	
}

