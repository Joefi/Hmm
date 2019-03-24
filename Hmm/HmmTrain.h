#pragma once
#include "hmm.h"
void initHmm(HMM *hmm, char* initFile);
double** forward(HMM *hmm, char* observation);
double** backward(HMM *hmm, char* observation);
double** computeGamma(double **alpha, double **beta, int T);
double*** computeEpsilon(HMM *hmm, char* observation, double **alpha, double **beta);
double computeTransition(double **gamma, double ***epsilon, int i, int j);
double computeObservation(char* observation, double **gamma, int i, int o);
int getTotalLine(char *fileName);
char * readOneSample(char *fileName, int no);
void train(HMM *hmm, char* fileName, char *outputFile, int iteration);
void viterbi(HMM *hmm, char *observation, int *path, double *probability);
void test(HMM *hmm, char *testFile, char *resultFile);