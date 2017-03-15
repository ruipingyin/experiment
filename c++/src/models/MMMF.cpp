#include "MMMF.hpp"


void MMMF::updateFactors(int user_id, int pos_item_id, int neg_item_id, double learn_rate)
{
	double x_uij = beta_item[pos_item_id] - beta_item[neg_item_id];
	x_uij += inner(gamma_user[user_id], gamma_item[pos_item_id], K) - inner(gamma_user[user_id], gamma_item[neg_item_id], K);

	double deri = x_uij < 0 ? 1 : 0;

	beta_item[pos_item_id] += learn_rate * (deri - biasReg * beta_item[pos_item_id]);
	beta_item[neg_item_id] += learn_rate * (-deri - biasReg * beta_item[neg_item_id]);

	// adjust latent factors
	for (int f = 0; f < K; f ++) {
		double w_uf = gamma_user[user_id][f];
		double h_if = gamma_item[pos_item_id][f];
		double h_jf = gamma_item[neg_item_id][f];

		gamma_user[user_id][f]     += learn_rate * ( deri * (h_if - h_jf) - lambda * w_uf);
		gamma_item[pos_item_id][f] += learn_rate * ( deri * w_uf - lambda * h_if);
		gamma_item[neg_item_id][f] += learn_rate * (-deri * w_uf - lambda / 10.0 * h_jf);
	}
}

string MMMF::toString()
{
	char str[10000];
	sprintf(str, "MMMF__K_%d_lambda_%.2f_biasReg_%.2f", K, lambda, biasReg);
	return str;
}

