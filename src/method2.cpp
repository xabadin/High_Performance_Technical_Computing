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
	int const m = 464;
	int const n = 464;
	int const k = 464;

	/*
	* Initilization of the sparse matrix (M), the fat vector (v) and the result matrix (result)
	*/
	static int M[m][n];
	static int v[n][k];
	static int result[m][k];

	/*
	* sendCount and sendCountBis are two arrays describing the number of data that the root process is going to send with scatterv
	*/
	int sendCount[npes] = { 0 };
	int sendCountBis[npes] = { 0 };

	/*
	* positions and positionsBis are two arrays used for MPI_Scatterv, it describes how to scatter the data in the different processes
	*/
	int positions[npes] = { 0 };
	int positionsBis[npes] = { 0 };

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
	* Broadcast the fat vector to all the processes and calculates the communication time.
	*/
	startCommunicationTime = MPI_Wtime();
	MPI_Bcast(&v, n * k, MPI_INT, 0, MPI_COMM_WORLD);
	endCommunicationTime = MPI_Wtime();
	communicationTime = endCommunicationTime - startCommunicationTime;

	startComputationTime = MPI_Wtime();

	int nonZero = 0;

	/*
	* Counts the number of non zero values in the sparse matrix
	*/
	if (rank == 0)
	{
		for (int i = 0; i < m; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (M[i][j] != 0)
				{
					nonZero++;
				}
			}
		}
	}

	endComputationTime = MPI_Wtime();
	computationTime = endComputationTime - startComputationTime;

	startCommunicationTime = MPI_Wtime();
	MPI_Bcast(&nonZero, 1, MPI_INT, 0, MPI_COMM_WORLD);
	endCommunicationTime = MPI_Wtime();
	communicationTime += endCommunicationTime - startCommunicationTime;

	startComputationTime = MPI_Wtime();

	/*
	* the buffers which are going to receive the data
	*/
	int receiveBufferNZ[nonZero] = { 0 };
	int receiveBufferCI[nonZero] = { 0 };

	/*
	* Creates three arrays and populates them according to the CSR format.
	* nonZeros which contains the non zero values
	* columnIndex which contains the index of the column in which the value is stored
	* endRowIndex which contains the index to mark the boundary of the rows
	*/
	int nonZeros[nonZero] = { 0 };
	int columnIndex[nonZero] = { 0 };
	int endRowIndex[m + 1] = { 0 };

	/*
	* Populating the three arrays
	*/
	if (rank == 0)
	{
		int index = 0;
		int x = 1;
		for (int i = 0; i < m; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (M[i][j] != 0)
				{
					nonZeros[index] = M[i][j];
					columnIndex[index] = j;
					index++;
				}
				if (j == n - 1)
				{
					endRowIndex[x] = index;
					x++;
				}
			}
		}
	}

	endComputationTime = MPI_Wtime();
	computationTime += endComputationTime - startComputationTime;

	startCommunicationTime = MPI_Wtime();
	MPI_Bcast(endRowIndex, m + 1, MPI_INT, 0, MPI_COMM_WORLD);
	endCommunicationTime = MPI_Wtime();
	communicationTime += endCommunicationTime - startCommunicationTime;

	startComputationTime = MPI_Wtime();
	/*
	* Calculates how to split the sparse-matrix between all the processes
	*/
	int remaining = m % npes;
	int temp = 0;
	for (int i = 0; i < npes; ++i)
	{
		sendCount[i] = m / npes;
		if (remaining > 0)
		{
			sendCount[i] += 1;
			remaining--;
		}
		positions[i] = temp;
		temp += sendCount[i];
	}

	int sendData = sendCount[rank] * k;
	/*
	* the buffer which is going to send the data back to the root node
	*/
	int sendBuffer[sendData] = { 0 };

	/*
	* Calculates the data that we need to send to each process and the positions for the arrays "nonZeros" and "columnIndex"
	*/
	for (int i = 0; i < npes; ++i)
	{
		positionsBis[i] = endRowIndex[positions[i]];

		if (i != npes - 1)
		{
			sendCountBis[i] = endRowIndex[positions[i + 1]] - endRowIndex[positions[i]];
		}
		else
		{
			sendCountBis[i] = nonZero - endRowIndex[positions[i]];
		}
	}

	endComputationTime = MPI_Wtime();
	computationTime += endComputationTime - startComputationTime;

	startCommunicationTime = MPI_Wtime();
	/*
	* Scatter the two arrays between all the processes
	*/
	if (rank == 0)
	{
		//MPI_IN_PLACE is used for the root process not to send data to itself
		MPI_Scatterv(nonZeros, sendCountBis, positionsBis, MPI_INT, MPI_IN_PLACE, sendCountBis[rank], MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatterv(columnIndex, sendCountBis, positionsBis, MPI_INT, MPI_IN_PLACE, sendCountBis[rank], MPI_INT, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Scatterv(NULL, NULL, NULL, MPI_INT, receiveBufferNZ, sendCountBis[rank], MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatterv(NULL, NULL, NULL, MPI_INT, receiveBufferCI, sendCountBis[rank], MPI_INT, 0, MPI_COMM_WORLD);
	}
	endCommunicationTime = MPI_Wtime();
	communicationTime += endCommunicationTime - startCommunicationTime;

	startComputationTime = MPI_Wtime();
	/*
	* Computes the value of the result matrix with the CSR method
	*/
	if (rank == 0)
	{
		for (int i = 0; i < k; ++i)
		{
			for (int x = 0; x < sendCount[rank]; ++x)
			{
				int temp = 0;
				for (int j = endRowIndex[x]; j < endRowIndex[x + 1]; ++j)
				{
					temp += nonZeros[j] * v[columnIndex[j]][i];
				}
				result[x][i] = temp;
			}
		}
	}
	else
	{
		int indexBuffer = 0;
		for (int i = 0; i < k; ++i)
		{
			int index = 0;
			for (int x = positions[rank]; x < (sendCount[rank] + positions[rank]); ++x)
			{
				int temp = 0;
				for (int j = endRowIndex[x]; j < endRowIndex[x + 1]; ++j)
				{
					temp += receiveBufferNZ[index] * v[receiveBufferCI[index]][i];
					index++;
				}
				sendBuffer[indexBuffer] = temp;
				indexBuffer++;
			}
		}
	}

	int sum = 0;
	/*
	* Recalculates the positions of the data in the result matrix and the number that needs to be send
	*/
	for (int i = 0; i < npes; ++i)
	{
		positionsBis[i] = sum;
		sum += sendCount[i] * k;
		sendCountBis[i] = sendCount[i] * k;
	}

	int sorted[sendCountBis[rank]] = { 0 };
	/*
	* Before sending the data we need to sort it because it is not sorted by row
	*/
	if (rank != 0)
	{
		int index = 0;
		for (int i = 0; i < sendCountBis[rank]; ++i)
		{
			if (index >= sendCountBis[rank])
			{
				index++;
				index = index % sendCount[rank];
			}
			sorted[i] = sendBuffer[index];
			index += sendCount[rank];
		}
	}

	endComputationTime = MPI_Wtime();
	computationTime += endComputationTime - startComputationTime;

	startCommunicationTime = MPI_Wtime();
	/*
	* Gather the data in the root process
	*/
	if (rank == 0)
	{
		MPI_Gatherv(MPI_IN_PLACE, sendCountBis[rank], MPI_INT, result, sendCountBis, positionsBis, MPI_INT, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Gatherv(sorted, sendCountBis[rank], MPI_INT, result, sendCountBis, positionsBis, MPI_INT, 0, MPI_COMM_WORLD);
	}
	endCommunicationTime = MPI_Wtime();
	communicationTime += endCommunicationTime - startCommunicationTime;

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
		std::ofstream ofs("resultSecondMethod.csv");
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