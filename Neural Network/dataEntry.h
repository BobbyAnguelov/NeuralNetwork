#ifndef _DATAENTRY
#define _DATAENTRY

//standard libraries
#include <iostream>
#include <vector>

using namespace std;

class dataEntry
{
public:
	
	//public members
	//----------------------------------------------------------------------------------------------------------------
	double* pattern;	//all the patterns
	double* target;		//all the targets

public:

	//public methods
	//----------------------------------------------------------------------------------------------------------------

	//constructor
	dataEntry(double* p, double* t): pattern(p), target(t) {}
		
	~dataEntry()
	{				
		delete[] pattern;
		delete[] target;
	}

};

#endif
