#pragma once

#include "BPRMF.hpp"

class MMMF : public BPRMF
{
public:
	MMMF(corpus* corp, int K, double lambda, double biasReg) 
		: BPRMF(corp, K, lambda, biasReg) {}

	~MMMF(){}
	
	void updateFactors(int user_id, int pos_item_id, int neg_item_id, double learn_rate);
	string toString();
};
