//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include "TrainingDataReader.h"
#include <assert.h>
#include <iosfwd>
#include <algorithm>
#include <iostream>

//-------------------------------------------------------------------------


namespace BPN
{
    TrainingDataReader::TrainingDataReader( std::string const& filename, int32_t numInputs, int32_t numOutputs )
        : m_filename( filename )
        , m_numInputs( numInputs )
        , m_numOutputs( numOutputs )
    {
        assert( !filename.empty() && m_numInputs > 0 && m_numOutputs > 0 );
    }

    bool TrainingDataReader::ReadData()
    {
        assert( !m_filename.empty() );

        std::fstream inputFile;
        inputFile.open( m_filename, std::ios::in );

        if ( inputFile.is_open() )
        {
            std::string line;

            // Read data
            //-------------------------------------------------------------------------

            int32_t const totalValuesToRead = m_numInputs + m_numOutputs;

            while ( !inputFile.eof() )
            {
                std::getline( inputFile, line );
                if ( line.length() > 2 )
                {
                    m_entries.push_back( TrainingEntry() );
                    TrainingEntry& entry = m_entries.back();

                    char* cstr = new char[line.size() + 1];
                    strcpy_s( cstr, line.size() + 1, line.c_str() );

                    // Read values
                    int i = 0;
                    char* nextToken = nullptr;
                    char* pToken = strtok_s( cstr, ",", &nextToken );

                    while ( pToken != nullptr && i < totalValuesToRead )
                    {
                        if ( i < m_numInputs )
                        {
                            entry.m_inputs.push_back( atof( pToken ) );
                        }
                        else
                        {
                            double const outputValue = atof( pToken );
                            entry.m_expectedOutputs.push_back( (int32_t) outputValue );
                        }

                        pToken = strtok_s( nullptr, ",", &nextToken );
                        i++;
                    }
                }
            }

            inputFile.close();

            if ( !m_entries.empty() )
            {
                CreateTrainingData();
            }

            std::cout << "Input file: " << m_filename << "\nRead complete: " << m_entries.size() << " inputs loaded" << std::endl;
            return true;
        }
        else
        {
            std::cout << "Error Opening Input File: " << m_filename << std::endl;
            return false;
        }
    }

    void TrainingDataReader::CreateTrainingData()
    {
        assert( !m_entries.empty() );

        std::random_shuffle( m_entries.begin(), m_entries.end() );

        // Training set
        int32_t const numEntries = (int32_t) m_entries.size();
        int32_t const numTrainingEntries  = (int32_t) ( 0.6 * numEntries );
        int32_t const numGeneralizationEntries = (int32_t) ( ceil( 0.2 * numEntries ) );

        int32_t entryIdx = 0;
        for ( ; entryIdx < numTrainingEntries; entryIdx++ )
        {
            m_data.m_trainingSet.push_back( m_entries[entryIdx] );
        }

        // Generalization set
        for ( ; entryIdx < numTrainingEntries + numGeneralizationEntries; entryIdx++ )
        {
            m_data.m_generalizationSet.push_back( m_entries[entryIdx] );
        }

        // Validation set
        for ( ; entryIdx < numEntries; entryIdx++ )
        {
            m_data.m_validationSet.push_back( m_entries[entryIdx] );
        }
    }
}