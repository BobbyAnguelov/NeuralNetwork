//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include "NeuralNetworkTrainer.h"
#include <assert.h>
#include <iostream>

//-------------------------------------------------------------------------

namespace BPN
{
    NetworkTrainer::NetworkTrainer( Settings const& settings, Network* pNetwork )
        : m_pNetwork( pNetwork )
        , m_learningRate( settings.m_learningRate )
        , m_momentum( settings.m_momentum )
        , m_desiredAccuracy( settings.m_desiredAccuracy )
        , m_maxEpochs( settings.m_maxEpochs )
        , m_useBatchLearning( settings.m_useBatchLearning )
        , m_currentEpoch( 0 )
        , m_trainingSetAccuracy( 0 )
        , m_validationSetAccuracy( 0 )
        , m_generalizationSetAccuracy( 0 )
        , m_trainingSetMSE( 0 )
        , m_validationSetMSE( 0 )
        , m_generalizationSetMSE( 0 )
    {
        assert( pNetwork != nullptr );

        m_deltaInputHidden.resize( pNetwork->m_weightsInputHidden.size() );
        m_deltaHiddenOutput.resize( pNetwork->m_weightsHiddenOutput.size() );
        m_errorGradientsHidden.resize( pNetwork->m_hiddenNeurons.size() );
        m_errorGradientsOutput.resize( pNetwork->m_outputNeurons.size() );

        memset( m_deltaInputHidden.data(), 0, sizeof( double ) * m_deltaInputHidden.size() );
        memset( m_deltaHiddenOutput.data(), 0, sizeof( double ) * m_deltaHiddenOutput.size() );
        memset( m_errorGradientsHidden.data(), 0, sizeof( double ) * m_errorGradientsHidden.size() );
        memset( m_errorGradientsOutput.data(), 0, sizeof( double ) * m_errorGradientsOutput.size() );
    }

    void NetworkTrainer::Train( TrainingData const& trainingData )
    {
        // Reset training state
        m_currentEpoch = 0;
        m_trainingSetAccuracy = 0;
        m_validationSetAccuracy = 0;
        m_generalizationSetAccuracy = 0;
        m_trainingSetMSE = 0;
        m_validationSetMSE = 0;
        m_generalizationSetMSE = 0;

        // Print header
        //-------------------------------------------------------------------------

        std::cout	<< std::endl << " Neural Network Training Starting: " << std::endl
                    << "==========================================================================" << std::endl
                    << " LR: " << m_learningRate << ", Momentum: " << m_momentum << ", Max Epochs: " << m_maxEpochs << std::endl
                    << " " << m_pNetwork->m_numInputs<< " Input Neurons, " << m_pNetwork->m_numHidden << " Hidden Neurons, " << m_pNetwork->m_numOutputs<< " Output Neurons" << std::endl
                    << "==========================================================================" << std::endl << std::endl;

        // Train network using training dataset for training and generalization dataset for testing
        //--------------------------------------------------------------------------------------------------------

        while ( ( m_trainingSetAccuracy < m_desiredAccuracy || m_generalizationSetAccuracy < m_desiredAccuracy ) && m_currentEpoch < m_maxEpochs )
        {
            // Use training set to train network
            RunEpoch( trainingData.m_trainingSet );

            // Get generalization set accuracy and MSE
            GetSetAccuracyAndMSE( trainingData.m_generalizationSet, m_generalizationSetAccuracy, m_generalizationSetMSE );

            std::cout << "Epoch :" << m_currentEpoch;
            std::cout << " Training Set Accuracy:" << m_trainingSetAccuracy << "%, MSE: " << m_trainingSetMSE;
            std::cout << " Generalization Set Accuracy:" << m_generalizationSetAccuracy << "%, MSE: " << m_generalizationSetMSE << std::endl;

            m_currentEpoch++;
        }

        // Get validation set accuracy and MSE
        GetSetAccuracyAndMSE( trainingData.m_validationSet, m_validationSetAccuracy, m_validationSetMSE );

        // Print validation accuracy and MSE
        std::cout << std::endl << "Training Complete!!! - > Elapsed Epochs: " << m_currentEpoch << std::endl;
        std::cout << " Validation Set Accuracy: " << m_validationSetAccuracy << std::endl;
        std::cout << " Validation Set MSE: " << m_validationSetMSE << std::endl << std::endl;
    }

    double NetworkTrainer::GetHiddenErrorGradient( int32_t hiddenIdx ) const
    {
        // Get sum of hidden->output weights * output error gradients
        double weightedSum = 0;
        for ( auto outputIdx = 0; outputIdx < m_pNetwork->m_numOutputs; outputIdx++ )
        {
            int32_t const weightIdx = m_pNetwork->GetHiddenOutputWeightIndex( hiddenIdx, outputIdx );
            weightedSum += m_pNetwork->m_weightsHiddenOutput[weightIdx] * m_errorGradientsOutput[outputIdx];
        }
        
        // Return error gradient
        return m_pNetwork->m_hiddenNeurons[hiddenIdx] * ( 1.0 - m_pNetwork->m_hiddenNeurons[hiddenIdx] ) * weightedSum;
    }

    void NetworkTrainer::RunEpoch( TrainingSet const& trainingSet )
    {
        double incorrectEntries = 0;
        double MSE = 0;

        for ( auto const& trainingEntry : trainingSet )
        {
            // Feed inputs through network and back propagate errors
            m_pNetwork->Evaluate( trainingEntry.m_inputs );
            Backpropagate( trainingEntry.m_expectedOutputs );

            // Check all outputs from neural network against desired values
            bool resultCorrect = true;
            for ( int outputIdx = 0; outputIdx < m_pNetwork->m_numOutputs; outputIdx++ )
            {
                if ( m_pNetwork->m_clampedOutputs[outputIdx] != trainingEntry.m_expectedOutputs[outputIdx] )
                {
                    resultCorrect = false;
                }

                // Calculate MSE
                MSE += pow( ( m_pNetwork->m_outputNeurons[outputIdx] - trainingEntry.m_expectedOutputs[outputIdx] ), 2);
            }

            if ( !resultCorrect )
            {
                incorrectEntries++;
            }
        }
        
        // If using batch learning - update the weights
        if ( m_useBatchLearning )
        {
            UpdateWeights();
        }

        // Update training accuracy and MSE
        m_trainingSetAccuracy = 100.0 - ( incorrectEntries / trainingSet.size() * 100.0 );
        m_trainingSetMSE = MSE / ( m_pNetwork->m_numOutputs * trainingSet.size() );
    }

    void NetworkTrainer::Backpropagate( std::vector<int32_t> const& expectedOutputs )
    {
        // Modify deltas between hidden and output layers
        //--------------------------------------------------------------------------------------------------------
        for ( auto OutputIdx = 0; OutputIdx < m_pNetwork->m_numOutputs; OutputIdx++ )
        {
            // Get error gradient for every output node
            m_errorGradientsOutput[OutputIdx] = GetOutputErrorGradient( (double) expectedOutputs[OutputIdx], m_pNetwork->m_outputNeurons[OutputIdx] );

            // For all nodes in hidden layer and bias neuron
            for ( auto hiddenIdx = 0; hiddenIdx <= m_pNetwork->m_numHidden; hiddenIdx++ )
            {
                int32_t const weightIdx = m_pNetwork->GetHiddenOutputWeightIndex( hiddenIdx, OutputIdx );

                // Calculate change in weight
                if ( m_useBatchLearning )
                {
                    m_deltaHiddenOutput[weightIdx] += m_learningRate * m_pNetwork->m_hiddenNeurons[hiddenIdx] * m_errorGradientsOutput[OutputIdx];
                }
                else
                {
                    m_deltaHiddenOutput[weightIdx] = m_learningRate * m_pNetwork->m_hiddenNeurons[hiddenIdx] * m_errorGradientsOutput[OutputIdx] + m_momentum * m_deltaHiddenOutput[weightIdx];
                }
            }
        }

        // Modify deltas between input and hidden layers
        //--------------------------------------------------------------------------------------------------------

        for ( auto hiddenIdx = 0; hiddenIdx <= m_pNetwork->m_numHidden; hiddenIdx++ )
        {
            // Get error gradient for every hidden node
            m_errorGradientsHidden[hiddenIdx] = GetHiddenErrorGradient( hiddenIdx );

            // For all nodes in input layer and bias neuron
            for ( auto inputIdx = 0; inputIdx <= m_pNetwork->m_numInputs; inputIdx++ )
            {
                int32_t const weightIdx = m_pNetwork->GetInputHiddenWeightIndex( inputIdx, hiddenIdx );

                // Calculate change in weight 
                if ( m_useBatchLearning )
                {
                    m_deltaInputHidden[weightIdx] += m_learningRate * m_pNetwork->m_inputNeurons[inputIdx] * m_errorGradientsHidden[hiddenIdx];
                }
                else
                {
                    m_deltaInputHidden[weightIdx] = m_learningRate * m_pNetwork->m_inputNeurons[inputIdx] * m_errorGradientsHidden[hiddenIdx] + m_momentum * m_deltaInputHidden[weightIdx];
                }
            }
        }

        // If using stochastic learning update the weights immediately
        if ( !m_useBatchLearning )
        {
            UpdateWeights();
        }
    }

    void NetworkTrainer::UpdateWeights()
    {
        // Input -> hidden weights
        //--------------------------------------------------------------------------------------------------------

        for ( auto InputIdx = 0; InputIdx <= m_pNetwork->m_numInputs; InputIdx++ )
        {
            for ( auto hiddenIdx = 0; hiddenIdx <= m_pNetwork->m_numHidden; hiddenIdx++ )
            {
                int32_t const weightIdx = m_pNetwork->GetInputHiddenWeightIndex( InputIdx, hiddenIdx );
                m_pNetwork->m_weightsInputHidden[weightIdx] += m_deltaInputHidden[weightIdx];

                // Clear delta only if using batch (previous delta is needed for momentum
                if ( m_useBatchLearning )
                {
                    m_deltaInputHidden[weightIdx] = 0;
                }
            }
        }

        // Hidden -> output weights
        //--------------------------------------------------------------------------------------------------------

        for ( auto hiddenIdx = 0; hiddenIdx <= m_pNetwork->m_numHidden; hiddenIdx++ )
        {
            for ( auto outputIdx = 0; outputIdx < m_pNetwork->m_numOutputs; outputIdx++ )
            {
                int32_t const weightIdx = m_pNetwork->GetHiddenOutputWeightIndex( hiddenIdx, outputIdx );
                m_pNetwork->m_weightsHiddenOutput[weightIdx] += m_deltaHiddenOutput[weightIdx];

                // Clear delta only if using batch (previous delta is needed for momentum)
                if ( m_useBatchLearning )
                {
                    m_deltaInputHidden[weightIdx] = 0;
                }
            }
        }
    }

    void NetworkTrainer::GetSetAccuracyAndMSE( TrainingSet const& trainingSet, double& accuracy, double& MSE ) const
    {
        accuracy = 0;
        MSE = 0;

        double numIncorrectResults = 0;
        for ( auto const& trainingEntry : trainingSet )
        {
            m_pNetwork->Evaluate( trainingEntry.m_inputs );

            // Check if the network outputs match the expected outputs
            bool correctResult = true;
            for ( int32_t outputIdx = 0; outputIdx < m_pNetwork->m_numOutputs; outputIdx++ )
            {
                if ( (double) m_pNetwork->m_clampedOutputs[outputIdx] != trainingEntry.m_expectedOutputs[outputIdx] )
                {
                    correctResult = false;
                }

                MSE += pow( ( m_pNetwork->m_outputNeurons[outputIdx] - trainingEntry.m_expectedOutputs[outputIdx] ), 2 );
            }

            if ( !correctResult )
            {
                numIncorrectResults++;
            }
        }

        accuracy = 100.0f - ( numIncorrectResults / trainingSet.size() * 100.0 );
        MSE = MSE / ( m_pNetwork->m_numOutputs * trainingSet.size() );
    }

}