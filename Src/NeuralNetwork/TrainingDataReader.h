//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#pragma once

#include "NeuralNetworkTrainer.h"
#include <string>

//-------------------------------------------------------------------------

namespace BPN
{
    class TrainingDataReader
    {
    public:

        TrainingDataReader( std::string const& filename, int32_t numInputs, int32_t numOutputs );

        bool ReadData();

        inline int32_t GetNumInputs() const { return m_numInputs; }
        inline int32_t GetNumOutputs() const { return m_numOutputs; }

        inline int32_t GetNumTrainingSets() const { return 0; }
        TrainingData const& GetTrainingData() const { return m_data; }

    private:

        void CreateTrainingData();

    private:

        std::string                     m_filename;
        int32_t                         m_numInputs;
        int32_t                         m_numOutputs;

        std::vector<TrainingEntry>      m_entries;
        TrainingData                    m_data;
    };
}