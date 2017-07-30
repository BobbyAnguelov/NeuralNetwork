//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include "NeuralNetwork/neuralNetworkTrainer.h"
#include "NeuralNetwork/TrainingDataReader.h"
#include <iostream>

#if _MSC_VER
#pragma warning(push, 0)
#pragma warning(disable: 4702)
#endif

#include "cmdParser.h"

#if _MSC_VER
#pragma warning(pop)
#endif

//-------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    cli::Parser cmdParser( argc, argv );
    cmdParser.set_required<std::string>( "d", "DataFile", "Path to training data csv file." );
    cmdParser.set_required<uint32_t>( "in", "NumInputs", "Num Input neurons." );
    cmdParser.set_required<uint32_t>( "hidden", "NumHidden", "Num Hidden neurons." );
    cmdParser.set_required<uint32_t>( "out", "NumOutputs", "Num Output neurons." );

    if ( !cmdParser.run() )
    {
        std::cout << "Invalid command line arguments";
        return 1;
    }

    std::string trainingDataPath = cmdParser.get<std::string>( "d" ).c_str();
    uint32_t const numInputs = cmdParser.get<uint32_t>( "in" );
    uint32_t const numHidden = cmdParser.get<uint32_t>( "hidden" );
    uint32_t const numOutputs = cmdParser.get<uint32_t>( "out" );

    BPN::TrainingDataReader dataReader( trainingDataPath, numInputs, numOutputs );
    if ( !dataReader.ReadData() )
    {
        return 1;
    }

    // Create neural network
    BPN::Network::Settings networkSettings{ numInputs, numHidden, numOutputs };
    BPN::Network nn( networkSettings );

    // Create neural network trainer
    BPN::NetworkTrainer::Settings trainerSettings;
    trainerSettings.m_learningRate = 0.001;
    trainerSettings.m_momentum = 0.9;
    trainerSettings.m_useBatchLearning = false;
    trainerSettings.m_maxEpochs = 200;
    trainerSettings.m_desiredAccuracy = 90;

    BPN::NetworkTrainer trainer( trainerSettings, &nn );
    trainer.Train( dataReader.GetTrainingData() );

    return 0;
}
