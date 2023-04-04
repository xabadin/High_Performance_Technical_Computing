#include <iostream>
#include <fstream>
#include <mpi.h>
#include <random>

int main(int argc, char* argv[])
{
	int npes, rank;
	double startRunTime, endRunTime, runTime;
	double startComputationTime, endComputationTime, computationTime;

	MPI_Init(&argc, &argv);
	startRunTime = MPI_Wtime();
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/*
	* Initialization of the size of the sparse matrix (M) and the fat vector (v).
	* m is the number of rows of the sparse matrix
	* n is the number of columns of the sparse matrix and also the number of rows of the fat vector
	* k is the number of columns of the fat vector
	*/
	int const m = 10;
	int const n = 15;
	int const k = 12;

	/*
	* Initilization of the sparse matrix (M), the fat vector (v) and the result matrix (result)
	*/
	static int M[m][n];
	static int v[n][k];
	static int result[m][k];

	/*
	* Populating the sparse matrix with random values between 0-9.
	* Approximately one third of the values will be non zero values.
	*/
	int randomNumber = 0;
	std::random_device random; // It is used to obtain a seed to generate random numbers
	std::mt19937 generator(1); // Uses the same seed to compare the values
	//std::mt19937 generator(random());
	std::uniform_int_distribution<> sparseMatrixDistribution(1, 27);
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			randomNumber = sparseMatrixDistribution(generator);
			if (randomNumber >= 1 && randomNumber <= 9)
			{
				M[i][j] = randomNumber;
			}
			else
			{
				M[i][j] = 0;
			}
		}
	}

	/*
	* Populating the fat vector with random values between 0-9.
	*/
	std::uniform_int_distribution<> fatVectorDistribution(0, 9);
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			v[i][j] = fatVectorDistribution(generator);
		}
	}

	startComputationTime = MPI_Wtime();
	/*
	* Computes the value of the result matrix
	*/
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			result[i][j] = 0;
			for (int x = 0; x < n; ++x)
			{
				result[i][j] += M[i][x] * v[x][j];
			}
		}
	}
	endComputationTime = MPI_Wtime();
	computationTime = endComputationTime - startComputationTime;

	/*
	* Calculates the run time
	*/
	endRunTime = MPI_Wtime();
	runTime = endRunTime - startRunTime;

	/*
	* Creates a .csv to obtain the runtime and to be able to check if the values of the result matrix are correct
	* Comment this section for big matrices
	*/
	/*
	std::ofstream ofs("resultserial.csv");
	ofs << "Run Time : ;" << runTime << std::endl;
	ofs << "Computation Time : ;" << computationTime << std::endl;

	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			ofs << result[i][j] << ";";
		}
		ofs << "\n";
	}
	*/
	printf("m = %d, n = %d, k = %d\n", m, n, k);
	printf("Run Time is %f\n", runTime);
	printf("Computation Time is %f\n", computationTime);
	MPI_Finalize();
	return 0;
}