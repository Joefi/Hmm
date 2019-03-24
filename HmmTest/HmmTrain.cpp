#include "pch.h"
#include <iostream>
#include <cstring>
#include "hmm.h"

#include "HmmTrain.h"

void initHmm(HMM *hmm, char* initFile)
{
	std::cout << "初始化HMM\n";
	loadHMM(hmm, initFile);
	dumpHMM(stderr, hmm);
	std::cout << "初始化完成\n";
}

double** forward(HMM *hmm, char* observation)
{
	double **alpha = (double **)malloc(sizeof(double *) * strlen(observation));
	int t = 0, i = 0, j = 0;
	double sum = 0.0;
	//初始化
	alpha[0] = (double *)malloc(sizeof(double) * 6);
	for (i = 0; i < 6; i++)
	{
		alpha[0][i] = hmm->initial[i] * hmm->observation[observation[0] - 'A'][i];
	}
	for (t = 1; t < strlen(observation); t++)
	{
		alpha[t] = (double *)malloc(sizeof(double) * 6);
		for (i = 0; i < 6; i++)
		{
			sum = 0.0;
			for (j = 0; j < 6; j++)
			{
				sum += alpha[t - 1][j] * hmm->transition[j][i];

			}
			alpha[t][i] = sum * hmm->observation[observation[t] - 'A'][i];
		}
	}
	return alpha;
}

double** backward(HMM *hmm, char* observation)
{
	double** beta = (double **)malloc(sizeof(double *) * strlen(observation));
	int T = strlen(observation) - 1;
	int t = 0, i = 0, j = 0;
	double sum = 0.0;
	beta[T] = (double *)malloc(sizeof(double) * 6);
	for (i = 0; i < 6; i++)
	{
		beta[T][i] = 1;
	}
	for (t = T - 1; t >= 0; t--)
	{
		beta[t] = (double *)malloc(sizeof(double) * 6);
		for (i = 0; i < 6; i++)
		{
			sum = 0.0;
			for (j = 0; j < 6; j++)
			{
				sum += hmm->transition[i][j] * hmm->observation[observation[t + 1] - 'A'][j] * beta[t + 1][j];

			}
			beta[t][i] = sum;
		}
	}
	return beta;
}

double** computeGamma(double **alpha, double **beta, int T)
{
	int t = 0, i = 0;
	double sum = 0.0;
	double** gamma = (double **)malloc(sizeof(double *) * T);
	for (t = 0; t < 50; t++)
	{
		gamma[t] = (double *)malloc(sizeof(double) * 6);
		sum = 0.0;
		for (i = 0; i < 6; i++)
		{
			sum += alpha[t][i] * beta[t][i];
		}
		for (i = 0; i < 6; i++)
		{
			gamma[t][i] = alpha[t][i] * beta[t][i] / sum;
		}
	}
	return gamma;
}

double*** computeEpsilon(HMM *hmm, char* observation, double **alpha, double **beta)
{
	int t = 0, i = 0, j = 0;
	double sum = 0.0;
	double*** epsilon = (double ***)malloc(sizeof(double **) * (strlen(observation) - 1));
	for (t = 0; t < (strlen(observation) - 1); t++)
	{
		epsilon[t] = (double **)malloc(sizeof(double *) * 6);
		sum = 0.0;
		for (i = 0; i < 6; i++)
		{
			for (j = 0; j < 6; j++)
			{
				sum += alpha[t][i] * hmm->transition[i][j] * hmm->observation[observation[t + 1] - 'A'][j] * beta[t + 1][j];
			}
		}
		for (i = 0; i < 6; i++)
		{
			epsilon[t][i] = (double *)malloc(sizeof(double) * 6);
			for (j = 0; j < 6; j++)
			{
				epsilon[t][i][j] = (alpha[t][i] * hmm->transition[i][j] * hmm->observation[observation[t + 1] - 'A'][j] * beta[t + 1][j]) / sum;
			}
		}
	}
	return epsilon;
}

double computeTransition(double **gamma, double ***epsilon, int i, int j)
{
	int t = 0;
	double sum1 = 0.0;
	double sum2 = 0.0;
	for (t = 0; t < 49; t++)
	{
		sum1 += epsilon[t][i][j];
		sum2 += gamma[t][i];
	}
	return sum1 / sum2;
}

double computeObservation(char* observation, double **gamma, int i, int o)
{
	int t = 0;
	double sum1 = 0.0;
	double sum2 = 0.0;
	for (t = 0; t < 50; t++)
	{
		if (observation[t] - 'A' == o)
		{
			sum1 += gamma[t][i];
		}
		sum2 += gamma[t][i];
	}
	return sum1 / sum2;
}

int getTotalLine(char *fileName)
{
	int lines = 0;
	int ch = 0;
	//读取文件行数
	FILE* fp = fopen(fileName, "r");
	if (NULL == fp)
	{
		perror(fileName);
		exit(1);
	}
	while ((ch = fgetc(fp)) != EOF)
	{
		if (ch == '\n')
		{
			lines++;
		}
	}
	fclose(fp);
	return lines;
}
char * readOneSample(char *fileName, int no)
{
	FILE* fp = fopen(fileName, "r");
	char* buf = (char *)malloc(sizeof(char) * 1024);
	if (NULL == fp)
	{
		perror(fileName);
		exit(1);
	}
	fseek(fp, 51 * no, SEEK_SET);
	fgets(buf, 1024, fp);
	fclose(fp);
	buf[50] = '\0';
	return buf;
}

void train(HMM *hmm, char* fileName, char *outputFile, int iteration)
{
	double **alpha = NULL;
	double **beta = NULL;
	double **gamma = NULL;
	double ***epsilon = NULL;
	char* observation = NULL;
	FILE *fp = NULL;
	int ch = 0;
	int i = 0, j = 0, t = 0, o = 0, lines = 0;
	int line = 0;
	int iter = 0;
	lines = getTotalLine(fileName);
	std::cout << "总共" << lines << "条数据\n";
	for (iter = 0; iter < iteration; iter++)
	{
		double totalPi[6] = { 0.0 };

		double totalA1[6][6] = { 0.0 };
		double totalA2[6] = { 0.0 };

		double totalB1[6][6] = { 0.0 };
		double totalB2[6] = { 0.0 };
		for (line = 0; line < lines; line++)
		{
			observation = readOneSample(fileName, line);
			alpha = forward(hmm, observation);
			beta = backward(hmm, observation);
			gamma = computeGamma(alpha, beta, strlen(observation));
			epsilon = computeEpsilon(hmm, observation, alpha, beta);
			for (i = 0; i < 6; i++)
			{
				totalPi[i] += gamma[0][i];
			}
			for (i = 0; i < 6; i++)
			{
				for (j = 0; j < 6; j++)
				{
					for (t = 0; t < 49; t++)
					{
						totalA1[i][j] += epsilon[t][i][j];
					}
				}
			}
			for (i = 0; i < 6; i++)
			{
				for (t = 0; t < 49; t++)
				{
					totalA2[i] += gamma[t][i];
				}
			}

			for (i = 0; i < 6; i++)
			{
				for (o = 0; o < 6; o++)
				{
					for (t = 0; t < 49; t++)
					{
						if (observation[t] - 'A' == o)
						{
							totalB1[o][i] += gamma[t][i];
						}
					}
				}
			}
			for (i = 0; i < 6; i++)
			{
				for (t = 0; t < 49; t++)
				{
					totalB2[i] += gamma[t][i];
				}
			}
			//释放
			for (t = 0; t < 50; t++)
			{
				free(alpha[t]);
				free(beta[t]);
				free(gamma[t]);
			}
			free(alpha);
			free(beta);
			free(gamma);
			for (t = 0; t < 49; t++)
			{
				for (i = 0; i < 6; i++)
				{
					free(epsilon[t][i]);
				}
				free(epsilon[t]);
			}
			free(epsilon);
			free(observation);
		}
		for (i = 0; i < 6; i++)
		{
			hmm->initial[i] = totalPi[i] / lines;
		}
		for (i = 0; i < 6; i++)
		{
			for (j = 0; j < 6; j++)
			{
				hmm->transition[i][j] = totalA1[i][j] / totalA2[i];
			}
		}
		for (i = 0; i < 6; i++)
		{
			for (o = 0; o < 6; o++)
			{
				hmm->observation[o][i] = totalB1[o][i] / totalB2[i];
			}
		}
		std::cout << "第" << iter + 1 << "次迭代完成\n";
	}

	fp = fopen(outputFile, "w");
	dumpHMM(fp, hmm);
	fclose(fp);
	std::cout << "训练完成\n";
}

void viterbi(HMM *hmm, char *observation, int *path, double *probability)
{
	double delta[50][6] = { 0.0 };
	int psi[50][6] = { 0 };
	//初始化
	int i = 0, j = 0, t = 0;
	double maximum = 0.0;
	int maximumIndex = 0;
	int T = strlen(observation);
	for (i = 0; i < 6; i++)
	{
		delta[0][i] = hmm->initial[i] * hmm->observation[observation[0] - 'A'][i];
		psi[0][i] = 0;
	}
	//递推
	for (t = 1; t < T; t++)
	{
		for (i = 0; i < 6; i++)
		{
			maximum = 0.0;
			maximumIndex = 0;
			for (j = 0; j < 6; j++)
			{
				if (delta[t - 1][j] * hmm->transition[j][i] > maximum)
				{
					maximum = delta[t - 1][j] * hmm->transition[j][i];
					maximumIndex = j;
				}
			}
			delta[t][i] = maximum * hmm->observation[observation[t] - 'A'][i];
			psi[t][i] = maximumIndex;
		}
	}
	//终止
	maximum = 0.0;
	maximumIndex = 0;
	for (i = 0; i < 6; i++)
	{
		if (delta[T - 1][i] > maximum)
		{
			maximum = delta[T - 1][i];
			maximumIndex = i;
		}
	}
	*probability = maximum;
	path[T - 1] = maximumIndex;
	//回溯
	for (t = T - 2; t >= 0; t--)
	{
		path[t] = psi[t][path[t + 1]];
	}
}

void test(HMM *hmm, char *testFile, char *resultFile)
{
	int lines = 0;
	int line = 0, i = 0;
	char *observation = NULL;
	double probability = 0.0;
	double maximum = 0.0;
	char modelsName[5][50] = { "model_01.txt","model_02.txt","model_03.txt","model_04.txt","model_05.txt" };
	int *path = NULL;
	int optimalIndex = 0;
	int *optimalPath = NULL;
	FILE *fp = NULL;
	lines = getTotalLine(testFile);
	fp = fopen(resultFile, "w");
	for (line = 0; line < lines; line++)
	{
		observation = readOneSample(testFile, line);
		maximum = 0.0;
		optimalIndex = 0;
		optimalPath = NULL;
		for (i = 0; i < 5; i++)
		{
			path = (int *)malloc(sizeof(int) * strlen(observation));
			viterbi(&hmm[i], observation, path, &probability);
			if (probability > maximum) {
				if (optimalPath != NULL)
				{
					free(optimalPath);
				}
				optimalIndex = i;
				maximum = probability;
				optimalPath = path;
			}
			else {
				free(path);
			}
		}
		fprintf(fp, "%s\n", modelsName[optimalIndex]);
	}
	fclose(fp);
}

double computeAccuracy(char *predictFile, char *trueFile)
{
	int totalLines = getTotalLine(predictFile);
	FILE *fp1 = NULL;
	FILE *fp2 = NULL;
	int sum = 0;
	char buf1[1024] = { '\0' };
	char buf2[1024] = { '\0' };
	fp1 = fopen(predictFile, "r");
	fp2 = fopen(trueFile, "r");
	while (!feof(fp1) && !feof(fp2))
	{
		if (NULL == fgets(buf1, 1024, fp1))
		{
			break;
		}
		if (NULL == fgets(buf2, 1024, fp2))
		{
			break;
		}
		if (strcmp(buf1, buf2) == 0)
		{
			sum++;
		}
	}
	fclose(fp1);
	fclose(fp2);
	return double(sum) / totalLines;
	
}