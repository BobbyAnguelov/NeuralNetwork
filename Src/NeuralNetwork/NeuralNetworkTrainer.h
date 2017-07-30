//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------
// Basic Gradient Descent NN Trainer with Momentum and Batch Learning

#pragma once

#include "NeuralNetwork.h"
#include <fstream>

namespace BPN
{
    struct TrainingEntry
    {
        std::vector<double>         m_inputs;
        std::vector<int32_t>        m_expectedOutputs;
    };

    typedef std::vector<TrainingEntry> TrainingSet;

    struct TrainingData
    {
        TrainingSet m_trainingSet;
        TrainingSet m_generalizationSet;
        TrainingSet m_validationSet;
    };

    //-------------------------------------------------------------------------

    class NetworkTrainer
    {
    public:

        struct Settings
        {
            // Learning params
            double      m_learningRate = 0.001;
            double      m_momentum = 0.9;
            bool        m_useBatchLearning = false;

            // Stopping conditions
            uint32_t    m_maxEpochs = 150;
            double      m_desiredAccuracy = 90;
        };

    public:

        NetworkTrainer( Settings const& settings, Network* pNetwork );

        void Train( TrainingData const& trainingData );

    private:

        inline double GetOutputErrorGradient( double desiredValue, double outputValue ) const { return outputValue * ( 1.0 - outputValue ) * ( desiredValue - outputValue ); }
        double GetHiddenErrorGradient( int32_t hiddenIdx ) const;

        void RunEpoch( TrainingSet const& trainingSet );
        void Backpropagate( std::vector<int32_t> const& expectedOutputs );
        void UpdateWeights();

        void GetSetAccuracyAndMSE( TrainingSet const& trainingSet, double& accuracy, double& mse ) const;

    private:
        
        Network*                    m_pNetwork;                 // Network to train

        // Training settings
        double                      m_learningRate;             // Adjusts the step size of the weight update
        double                      m_momentum;                 // Improves performance of stochastic learning (don't use for batch)
        double                      m_desiredAccuracy;          // Target accuracy for training
        uint32_t                    m_maxEpochs;                // Max number of training epochs
        bool                        m_useBatchLearning;         // Should we use batch learning

        // Training data
        std::vector<double>         m_deltaInputHidden;         // Delta for input hidden layer
        std::vector<double>         m_deltaHiddenOutput;        // Delta for hidden output layer
        std::vector<double>         m_errorGradientsHidden;     // Error gradients for the hidden layer
        std::vector<double>         m_errorGradientsOutput;     // Error gradients for the outputs

        uint32_t                    m_currentEpoch;             // Epoch counter
        double                      m_trainingSetAccuracy;
        double                      m_validationSetAccuracy;
        double                      m_generalizationSetAccuracy;
        double                      m_trainingSetMSE;
        double                      m_validationSetMSE;
        double                      m_generalizationSetMSE;
    };
}