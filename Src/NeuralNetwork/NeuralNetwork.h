//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------
// A simple neural network supporting only a single hidden layer

#pragma once
#include <stdint.h>
#include <vector>

//-------------------------------------------------------------------------

namespace BPN
{
    enum class ActivationFunctionType
    {
        Sigmoid
    };

    //-------------------------------------------------------------------------

    class Network
    {
        friend class NetworkTrainer;

        //-------------------------------------------------------------------------

        inline static double SigmoidActivationFunction( double x )
        {
            return 1.0 / ( 1.0 + std::exp( -x ) );
        }

        inline static int32_t ClampOutputValue( double x )
        {
            if ( x < 0.1 ) return 0;
            else if ( x > 0.9 ) return 1;
            else return -1;
        }

    public:

        struct Settings
        {
            uint32_t                        m_numInputs;
            uint32_t                        m_numHidden;
            uint32_t                        m_numOutputs;
        };

    public:

        Network( Settings const& settings );
        Network( Settings const& settings, std::vector<double> const& weights );

        std::vector<int32_t> const& Evaluate( std::vector<double> const& input );

        std::vector<double> const& GetInputHiddenWeights() const { return m_weightsInputHidden; }
        std::vector<double> const& GetHiddenOutputWeights() const { return m_weightsHiddenOutput; }

    private:

        void InitializeNetwork();
        void InitializeWeights();
        void LoadWeights( std::vector<double> const& weights );

        int32_t GetInputHiddenWeightIndex( int32_t inputIdx, int32_t hiddenIdx ) const { return inputIdx * m_numHidden + hiddenIdx; }
        int32_t GetHiddenOutputWeightIndex( int32_t hiddenIdx, int32_t outputIdx ) const { return hiddenIdx * m_numOutputs + outputIdx; }

    private:

        int32_t                 m_numInputs;
        int32_t                 m_numHidden;
        int32_t                 m_numOutputs;

        std::vector<double>     m_inputNeurons;
        std::vector<double>     m_hiddenNeurons;
        std::vector<double>     m_outputNeurons;

        std::vector<int32_t>    m_clampedOutputs;

        std::vector<double>     m_weightsInputHidden;
        std::vector<double>     m_weightsHiddenOutput;
    };
}