#include <iostream>
#include <fstream>
#include <mpi.h>
#include <random>

int main(int argc, char* argv[])
{
	int npes, rank;
	double startRunTime, endRunTime, runTime, maxRunTime;
	double startComputationTime, endComputationTime, computationTime, maxComputationTime;
	double startCommunicationTime, endCommunicationTime, communicationTime, maxCommunicationTime;

	MPI_Init(&argc, &argv);
	startRunTime = MPI_Wtime();
	MPI_Status status;
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
	* sendCount describes the amount of data that the root process needs to send to each process
	*/
	int sendCount[npes] = { 0 };

	/*
	* positions describes the displacements
	*/
	int positions[npes] = { 0 };

	int maxData = (m / npes + 1) * n;
	/*
	* the buffer which is going to receive the data in the different processes
	*/
	int receiveBuffer[maxData] = { 0 };

	/*
	* the buffer which is going to send the data back to the root node
	*/
	int sendBuffer[maxData] = { 0 };

	/*
	* Populating the sparse matrix with random values between 0-9.
	* Approximately one third of the values will be non zero values.
	* Process 0 executes this task
	*/
	if (rank == 0)
	{
		int randomNumber = 0;
		std::random_device random; // It is used to obtain a seed to generate random numbers
		std::mt19937 generator(1); // Used to generate the same values with the same seed across the different parallel and serial methods
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
	}

	/*
	* Broadcasts the fat vector to all the processes and calculates the communication time.
	*/
	startCommunicationTime = MPI_Wtime();
	MPI_Bcast(&v, n * k, MPI_INT, 0, MPI_COMM_WORLD);
	endCommunicationTime = MPI_Wtime();
	communicationTime = endCommunicationTime - startCommunicationTime;

	startComputationTime = MPI_Wtime();

	/*
	* Splits the sparse matrix by row and multiplies the number of rows by n
	* It gives the amount of data that the root process needs to send
	*/
	int remaining = m % npes;;
	int temp = 0;
	for (int i = 0; i < npes; ++i)
	{
		sendCount[i] = m / npes * n;
		if (remaining > 0)
		{
			sendCount[i] += n;
			remaining--;
		}
		positions[i] = temp;;
		temp += sendCount[i];
	}

	endComputationTime = MPI_Wtime();
	computationTime = endComputationTime - startComputationTime;

	/*
	* Scatter the sparse matrix between all the processes.
	*/
	if (rank == 0)
	{
		startCommunicationTime = MPI_Wtime();
		//MPI_IN_PLACE is used for the root process not to send data to itself
		MPI_Scatterv(M, sendCount, positions, MPI_INT, MPI_IN_PLACE, sendCount[rank], MPI_INT, 0, MPI_COMM_WORLD);
		endCommunicationTime = MPI_Wtime();
		communicationTime += endCommunicationTime - startCommunicationTime;
	}
	else
	{
		startCommunicationTime = MPI_Wtime();
		MPI_Scatterv(NULL, NULL, NULL, MPI_INT, receiveBuffer, sendCount[rank], MPI_INT, 0, MPI_COMM_WORLD);
		endCommunicationTime = MPI_Wtime();
		communicationTime += endCommunicationTime - startCommunicationTime;
	}

	startCommunicationTime = MPI_Wtime();

	if (rank == 0)
	{
		for (int i = 0; i < sendCount[rank] / n; ++i)
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
	}
	else
	{
		int index = 0;
		int lineNb = sendCount[rank] / n;
		for (int i = 0; i < lineNb; ++i)
		{
			for (int j = 0; j < k; ++j)
			{
				for (int x = 0; x < n; ++x)
				{
					sendBuffer[index] += receiveBuffer[i * n + x] * v[x][j];
				}
				index++;
			}
		}
	}

	/*
	* Recalculates the positions of the data in the result matrix and the number that needs to be send
	*/
	for (int i = 0; i < npes; ++i)
	{
		positions[i] = positions[i] / n * k;
		sendCount[i] = sendCount[i] / n * k;
	}

	endComputationTime = MPI_Wtime();
	computationTime += endComputationTime - startComputationTime;

	/*
	* Gather the data in the root process
	*/
	if (rank == 0)
	{
		startCommunicationTime = MPI_Wtime();
		MPI_Gatherv(MPI_IN_PLACE, sendCount[rank], MPI_INT, result, sendCount, positions, MPI_INT, 0, MPI_COMM_WORLD);
		endCommunicationTime = MPI_Wtime();
		communicationTime += endCommunicationTime - startCommunicationTime;
	}
	else
	{
		startCommunicationTime = MPI_Wtime();
		MPI_Gatherv(sendBuffer, sendCount[rank], MPI_INT, result, sendCount, positions, MPI_INT, 0, MPI_COMM_WORLD);
		endCommunicationTime = MPI_Wtime();
		communicationTime += endCommunicationTime - startCommunicationTime;
	}

	/*
	* Calculates the run time
	*/
	endRunTime = MPI_Wtime();
	runTime = endRunTime - startRunTime;

	/*
	* MPI_REDUCE to get the maximum communication, computation and run times
	*/
	MPI_Reduce(&computationTime, &maxComputationTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&communicationTime, &maxCommunicationTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&runTime, &maxRunTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	/*
	* Creates a .csv to obtain the runtime and to be able to check if the values of the result matrix are correct.
	* Only executed by process 0, comment this part for big matrix multiplications
	*/
	if (rank == 0)
	{
		std::ofstream ofs("resultFirstMethod.csv");
		ofs << "Run Time : " << maxRunTime << std::endl;
		ofs << "Computation Time : " << maxComputationTime << std::endl;
		ofs << "Communication Time : " << maxCommunicationTime << std::endl;
		ofs << "Sparse:" << std::endl;
		for (int i = 0; i < m; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				ofs << M[i][j] << ";";
			}
			ofs << "\n";
		}

		ofs << "Fat vector:" << std::endl;
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < k; ++j)
			{
				ofs << v[i][j] << ";";
			}
			ofs << "\n";
		}

		ofs << "Result:" << std::endl;
		for (int i = 0; i < m; ++i)
		{
			for (int j = 0; j < k; ++j)
			{
				ofs << result[i][j] << ";";
			}
			ofs << "\n";
		}
		printf("Run Time is %f\n", maxRunTime);
		printf("Computation Time is %f\n", maxComputationTime);
		printf("Communication Time is %f\n", maxCommunicationTime);
	}
	MPI_Finalize();
	return 0;
}