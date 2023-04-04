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
	int const m = 15;
	int const n = 20;
	int const k = 25;

	/*
	* Initilization of the sparse matrix (M), the fat vector (v) and the result matrix (result)
	*/
	static int M[m][n];
	static int v[n][k];
	static int result[m][k];

	/*
	* the array sendCount describes the number of non zero values that each process needs to send
	* sendCountLines describes the number of line of the sparse matrix that each process will receive
	*/
	int sendCount[npes] = { 0 };
	int sendCountLines[npes] = { 0 };

	/*
	* the array positions describes how to split the non zero array between the processes
	* positionLines describes how the lines are distributed between the processes
	*/
	int positions[npes] = { 0 };
	int positionsLines[npes] = { 0 };

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

	/*
	* Calculates the number of lines that we need to send to each process and the positions
	*/
	int remaining = m % npes;
	int temp = 0;
	for (int i = 0; i < npes; ++i)
	{
		sendCountLines[i] = m / npes;
		if (remaining > 0)
		{
			sendCountLines[i] += 1;
			remaining--;
		}
		positionsLines[i] = temp;
		temp += sendCountLines[i];
	}

	int nonZero = 0;
	/*
	* Counts the number of non zero values in the sparse matrix
	* Populates the sendCount and positions arrays at the same time
	*/
	if (rank == 0)
	{
		int index = 0;
		int count = 0;
		int position = 0;
		for (int i = 0; i < m; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (i == positionsLines[index] && j == 0)
				{
					count = 0;
				}
				if (M[i][j] != 0)
				{
					nonZero++;
					count++;
				}
				if (index < npes && i == positionsLines[index] + sendCountLines[index] - 1 && j == n - 1)
				{
					sendCount[index] = count;
					positions[index] = position;
					position += count;
					index++;
				}
			}
		}
	}

	endComputationTime = MPI_Wtime();
	computationTime = endComputationTime - startComputationTime;

	startCommunicationTime = MPI_Wtime();
	MPI_Bcast(&nonZero, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(sendCount, npes, MPI_INT, 0, MPI_COMM_WORLD);
	endCommunicationTime = MPI_Wtime();
	communicationTime += endCommunicationTime - startCommunicationTime;

	startComputationTime = MPI_Wtime();

	/*
	* Creates three arrays and populates them according to the COO format.
	* nonZeros which contains the non zero values
	* columnIndex which contains the index of the column of the corresponding non zero value
	* rowIndex which contains the index of the row of the corresponding non zero value
	*/
	int nonZeros[nonZero] = { 0 };
	int columnIndex[nonZero] = { 0 };
	int rowIndex[nonZero] = { 0 };

	/*
	* Populating the three arrays
	*/
	if (rank == 0)
	{
		int index = 0;
		for (int i = 0; i < m; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				if (M[i][j] != 0)
				{
					nonZeros[index] = M[i][j];
					columnIndex[index] = j;
					rowIndex[index] = i;
					index++;
				}
			}
		}
	}

	/*
	* the buffers which are going the send the data from the root to the other processes
	*/
	int sendBuffer[nonZero] = { 0 };
	int sendBufferCol[nonZero] = { 0 };
	int sendBufferRow[nonZero] = { 0 };

	int recvBuffer[nonZero] = { 0 };
	int recvBufferCol[nonZero] = { 0 };
	int recvBufferRow[nonZero] = { 0 };
	
	if (rank == 0)
	{
		for (int j = 1; j < npes; ++j)
		{
			int index = positions[j];
			for (int i = 0; i < sendCount[j]; ++i)
			{
				sendBuffer[i] = nonZeros[index];
				sendBufferCol[i] = columnIndex[index];
				sendBufferRow[i] = rowIndex[index];
				index++;
			}
			endComputationTime = MPI_Wtime();
			computationTime += endComputationTime - startComputationTime;

			startCommunicationTime = MPI_Wtime();
			MPI_Send(sendBuffer, sendCount[j], MPI_INT, j, 0, MPI_COMM_WORLD);
			MPI_Send(sendBufferCol, sendCount[j], MPI_INT, j, 1, MPI_COMM_WORLD);
			MPI_Send(sendBufferRow, sendCount[j], MPI_INT, j, 2, MPI_COMM_WORLD);
			endCommunicationTime = MPI_Wtime();
			communicationTime += endCommunicationTime - startCommunicationTime;

			startComputationTime = MPI_Wtime();
		}
	}
	else {
		endComputationTime = MPI_Wtime();
		computationTime += endComputationTime - startComputationTime;

		startCommunicationTime = MPI_Wtime();
		MPI_Recv(recvBuffer, sendCount[rank], MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(recvBufferCol, sendCount[rank], MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(recvBufferRow, sendCount[rank], MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
		endCommunicationTime = MPI_Wtime();
		communicationTime += endCommunicationTime - startCommunicationTime;

		startComputationTime = MPI_Wtime();
	}


	int sum = 0;
	/*
	* Buffer which contains the number of data that each process needs to send to the root process
	*/
	int sendCountRoot[npes] = { 0 };
	for (int i = 0; i < npes; ++i)
	{
		positions[i] = sum;
		sum += sendCountLines[i] * k;
		sendCountRoot[i] = sendCountLines[i] * k;
	}

	/*
	* Buffer to send the data back to the root process
	*/
	int sendBufferRoot[sendCountRoot[rank]] = { 0 };
	/*
	* Computes the value of the result matrix with the COO method
	*/
	if (rank == 0)
	{
		for (int i = 0; i < k; ++i)
		{
			for (int j = 0; j < sendCount[rank]; ++j)
			{
				int row = rowIndex[j];
				int col = columnIndex[j];
				result[row][i] = result[row][i] + nonZeros[j] * v[col][i];
			}
		}
	}
	else
	{
		int index = -1;
		for (int i = 0; i < k; ++i)
		{
			for (int j = 0; j < sendCount[rank]; ++j)
			{
				int row = recvBufferRow[j];
				int col = recvBufferCol[j];
				if (row == recvBufferRow[j - 1])
				{
					sendBufferRoot[index] = sendBufferRoot[index] + recvBuffer[j] * v[col][i];
				}
				else
				{
					index++;
					sendBufferRoot[index] = 0;
					sendBufferRoot[index] = sendBufferRoot[index] + recvBuffer[j] * v[col][i];
				}
			}
		}
	}

	int sorted[sendCountRoot[rank]] = { 0 };
	/*
	* Before sending the data we need to sort it 
	*/
	if (rank != 0)
	{
		int index = 0;
		for (int i = 0; i < sendCountRoot[rank]; ++i)
		{
			if (index >= sendCountRoot[rank])
			{
				index++;
				index = index % sendCountLines[rank];
			}
			sorted[i] = sendBufferRoot[index];
			index += sendCountLines[rank];
		}
	}

	endComputationTime = MPI_Wtime();
	computationTime += endComputationTime - startComputationTime;

	/*
	* Gather the data in the root process
	*/
	startCommunicationTime = MPI_Wtime();
	if (rank == 0)
	{
		MPI_Gatherv(MPI_IN_PLACE, sendCountRoot[rank], MPI_INT, result, sendCountRoot, positions, MPI_INT, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Gatherv(sorted, sendCountRoot[rank], MPI_INT, result, sendCountRoot, positions, MPI_INT, 0, MPI_COMM_WORLD);
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
		std::ofstream ofs("resultThirdMethod.csv");
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