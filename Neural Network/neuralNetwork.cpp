//standard includes
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

//include definition file
#include "neuralNetwork.h"

using namespace std;

/*******************************************************************
* Constructor
********************************************************************/
neuralNetwork::neuralNetwork(int nI, int nH, int nO) : nInput(nI), nHidden(nH), nOutput(nO)
{				
	//create neuron lists
	//--------------------------------------------------------------------------------------------------------
	inputNeurons = new( double[nInput + 1] );
	for ( int i=0; i < nInput; i++ ) inputNeurons[i] = 0;

	//create input bias neuron
	inputNeurons[nInput] = -1;

	hiddenNeurons = new( double[nHidden + 1] );
	for ( int i=0; i < nHidden; i++ ) hiddenNeurons[i] = 0;

	//create hidden bias neuron
	hiddenNeurons[nHidden] = -1;

	outputNeurons = new( double[nOutput] );
	for ( int i=0; i < nOutput; i++ ) outputNeurons[i] = 0;

	//create weight lists (include bias neuron weights)
	//--------------------------------------------------------------------------------------------------------
	wInputHidden = new( double*[nInput + 1] );
	for ( int i=0; i <= nInput; i++ ) 
	{
		wInputHidden[i] = new (double[nHidden]);
		for ( int j=0; j < nHidden; j++ ) wInputHidden[i][j] = 0;		
	}

	wHiddenOutput = new( double*[nHidden + 1] );
	for ( int i=0; i <= nHidden; i++ ) 
	{
		wHiddenOutput[i] = new (double[nOutput]);			
		for ( int j=0; j < nOutput; j++ ) wHiddenOutput[i][j] = 0;		
	}	
	
	//initialize weights
	//--------------------------------------------------------------------------------------------------------
	initializeWeights();			
}

/*******************************************************************
* Destructor
********************************************************************/
neuralNetwork::~neuralNetwork()
{
	//delete neurons
	delete[] inputNeurons;
	delete[] hiddenNeurons;
	delete[] outputNeurons;

	//delete weight storage
	for (int i=0; i <= nInput; i++) delete[] wInputHidden[i];
	delete[] wInputHidden;

	for (int j=0; j <= nHidden; j++) delete[] wHiddenOutput[j];
	delete[] wHiddenOutput;	
}
/*******************************************************************
* Load Neuron Weights
********************************************************************/
bool neuralNetwork::loadWeights(char* filename)
{
	//open file for reading
	fstream inputFile;
	inputFile.open(filename, ios::in);

	if ( inputFile.is_open() )
	{
		vector<double> weights;
		string line = "";
		
		//read data
		while ( !inputFile.eof() )
		{
			getline(inputFile, line);				
			
			//process line
			if (line.length() > 2 ) 
			{				
				//store inputs		
				char* cstr = new char[line.size()+1];
				char* t;
				strcpy_s(cstr, line.size() + 1, line.c_str());

				//tokenise
				int i = 0;
                char* nextToken = NULL;
				t=strtok_s(cstr,",", &nextToken );
				
				while ( t!=NULL )
				{	
					weights.push_back( atof(t) );
				
					//move token onwards
					t = strtok_s(NULL,",", &nextToken );
					i++;			
				}

				//free memory
				delete[] cstr;
			}
		}	
		
		//check if sufficient weights were loaded
		if ( weights.size() != ( (nInput + 1) * nHidden + (nHidden +  1) * nOutput ) ) 
		{
			cout << endl << "Error - Incorrect number of weights in input file: " << filename << endl;
			
			//close file
			inputFile.close();

			return false;
		}
		else
		{
			//set weights
			int pos = 0;

			for ( int i=0; i <= nInput; i++ ) 
			{
				for ( int j=0; j < nHidden; j++ ) 
				{
					wInputHidden[i][j] = weights[pos++];					
				}
			}
			
			for ( int i=0; i <= nHidden; i++ ) 
			{		
				for ( int j=0; j < nOutput; j++ ) 
				{
					wHiddenOutput[i][j] = weights[pos++];						
				}
			}	

			//print success
			cout << endl << "Neuron weights loaded successfuly from '" << filename << "'" << endl;

			//close file
			inputFile.close();
			
			return true;
		}		
	}
	else 
	{
		cout << endl << "Error - Weight input file '" << filename << "' could not be opened: " << endl;
		return false;
	}
}
/*******************************************************************
* Save Neuron Weights
********************************************************************/
bool neuralNetwork::saveWeights(char* filename)
{
	//open file for reading
	fstream outputFile;
	outputFile.open(filename, ios::out);

	if ( outputFile.is_open() )
	{
		outputFile.precision(50);		

		//output weights
		for ( int i=0; i <= nInput; i++ ) 
		{
			for ( int j=0; j < nHidden; j++ ) 
			{
				outputFile << wInputHidden[i][j] << ",";				
			}
		}
		
		for ( int i=0; i <= nHidden; i++ ) 
		{		
			for ( int j=0; j < nOutput; j++ ) 
			{
				outputFile << wHiddenOutput[i][j];					
				if ( i * nOutput + j + 1 != (nHidden + 1) * nOutput ) outputFile << ",";
			}
		}

		//print success
		cout << endl << "Neuron weights saved to '" << filename << "'" << endl;

		//close file
		outputFile.close();
		
		return true;
	}
	else 
	{
		cout << endl << "Error - Weight output file '" << filename << "' could not be created: " << endl;
		return false;
	}
}
/*******************************************************************
* Feed pattern through network and return results
********************************************************************/
int* neuralNetwork::feedForwardPattern(double *pattern)
{
	feedForward(pattern);

	//create copy of output results
	int* results = new int[nOutput];
	for (int i=0; i < nOutput; i++ ) results[i] = clampOutput(outputNeurons[i]);

	return results;
}
/*******************************************************************
* Return the NN accuracy on the set
********************************************************************/
double neuralNetwork::getSetAccuracy( std::vector<dataEntry*>& set )
{
	double incorrectResults = 0;
		
	//for every training input array
	for ( int tp = 0; tp < (int) set.size(); tp++)
	{						
		//feed inputs through network and backpropagate errors
		feedForward( set[tp]->pattern );
		
		//correct pattern flag
		bool correctResult = true;

		//check all outputs against desired output values
		for ( int k = 0; k < nOutput; k++ )
		{					
			//set flag to false if desired and output differ
			if ( clampOutput(outputNeurons[k]) != set[tp]->target[k] ) correctResult = false;
		}
		
		//inc training error for a incorrect result
		if ( !correctResult ) incorrectResults++;	
		
	}//end for
	
	//calculate error and return as percentage
	return 100 - (incorrectResults/set.size() * 100);
}
/*******************************************************************
* Return the NN mean squared error on the set
********************************************************************/
double neuralNetwork::getSetMSE( std::vector<dataEntry*>& set )
{
	double mse = 0;
		
	//for every training input array
	for ( int tp = 0; tp < (int) set.size(); tp++)
	{						
		//feed inputs through network and backpropagate errors
		feedForward( set[tp]->pattern );
		
		//check all outputs against desired output values
		for ( int k = 0; k < nOutput; k++ )
		{					
			//sum all the MSEs together
			mse += pow((outputNeurons[k] - set[tp]->target[k]), 2);
		}		
		
	}//end for
	
	//calculate error and return as percentage
	return mse/(nOutput * set.size());
}
/*******************************************************************
* Initialize Neuron Weights
********************************************************************/
void neuralNetwork::initializeWeights()
{
	//set range
	double rH = 1/sqrt( (double) nInput);
	double rO = 1/sqrt( (double) nHidden);
	
	//set weights between input and hidden 		
	//--------------------------------------------------------------------------------------------------------
	for(int i = 0; i <= nInput; i++)
	{		
		for(int j = 0; j < nHidden; j++) 
		{
			//set weights to random values
			wInputHidden[i][j] = ( ( (double)(rand()%100)+1)/100  * 2 * rH ) - rH;			
		}
	}
	
	//set weights between input and hidden
	//--------------------------------------------------------------------------------------------------------
	for(int i = 0; i <= nHidden; i++)
	{		
		for(int j = 0; j < nOutput; j++) 
		{
			//set weights to random values
			wHiddenOutput[i][j] = ( ( (double)(rand()%100)+1)/100 * 2 * rO ) - rO;
		}
	}
}
/*******************************************************************
* Activation Function
********************************************************************/
inline double neuralNetwork::activationFunction( double x )
{
	//sigmoid function
	return 1/(1+exp(-x));
}	
/*******************************************************************
* Output Clamping
********************************************************************/
inline int neuralNetwork::clampOutput( double x )
{
	if ( x < 0.1 ) return 0;
	else if ( x > 0.9 ) return 1;
	else return -1;
}
/*******************************************************************
* Feed Forward Operation
********************************************************************/
void neuralNetwork::feedForward(double* pattern)
{
	//set input neurons to input values
	for(int i = 0; i < nInput; i++) inputNeurons[i] = pattern[i];
	
	//Calculate Hidden Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------
	for(int j=0; j < nHidden; j++)
	{
		//clear value
		hiddenNeurons[j] = 0;				
		
		//get weighted sum of pattern and bias neuron
		for( int i=0; i <= nInput; i++ ) hiddenNeurons[j] += inputNeurons[i] * wInputHidden[i][j];
		
		//set to result of sigmoid
		hiddenNeurons[j] = activationFunction( hiddenNeurons[j] );			
	}
	
	//Calculating Output Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------
	for(int k=0; k < nOutput; k++)
	{
		//clear value
		outputNeurons[k] = 0;				
		
		//get weighted sum of pattern and bias neuron
		for( int j=0; j <= nHidden; j++ ) outputNeurons[k] += hiddenNeurons[j] * wHiddenOutput[j][k];
		
		//set to result of sigmoid
		outputNeurons[k] = activationFunction( outputNeurons[k] );
	}
}


