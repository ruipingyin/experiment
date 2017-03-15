#include "corpus.hpp"
#include "POP.hpp"
#include "MMMF.hpp"
#include "BPRMF.hpp"
#include "VBPR.hpp"


void go_POP(corpus* corp)
{
	POP md(corp);
	fprintf(stderr, "\n\n <<< Popularity >>> Precision = %f, Recall = %f\n", md.precision(100), md.recall(100));
	fprintf(stderr, "\n\n <<< Popularity >>> Cold Start: #Item = d, Test AUC = %f, Test Std = %f\n", md.precision_coldItem(100), md.recall_coldItem(100));
}

void go_MMMF(corpus* corp, int K, double lambda, double biasReg, int iterations, const char* corp_name)
{
	MMMF md(corp, K, lambda, biasReg);
	md.init();
	md.train(iterations, 0.005);
	// md.saveModel((string(corp_name) + "__" + md.toString()).c_str());
	md.cleanUp();
}

void go_BPRMF(corpus* corp, int K, double lambda, double biasReg, int iterations, const char* corp_name)
{
	BPRMF md(corp, K, lambda, biasReg);
	md.init();
	md.train(iterations, 0.005);
	// md.saveModel((string(corp_name) + "__" + md.toString()).c_str());
	md.cleanUp();
}

void go_VBPR(corpus* corp, int K, int K2, double lambda, double lambda2, double biasReg, int iterations, const char* corp_name)
{
	VBPR md(corp, K, K2, lambda, lambda2, biasReg);
	md.init();
	md.train(iterations, 0.005);
	// md.saveModel((string(corp_name) + "__" + md.toString()).c_str());
	md.cleanUp();
}

int main(int argc, char** argv)
{
	srand(0);
	
	// if (argc != 10) {
		// printf(" Parameters as following: \n");
		// printf(" 1. Review file path\n");
		// printf(" 2. Img feature path\n");
		// printf(" 3. Latent Feature Dim. (K)\n");
		// printf(" 4. Visual Feature Dim. (K')\n");
		// printf(" 5. biasReg (regularize bias terms)\n");
		// printf(" 6. lambda  (regularize general terms)\n");
		// printf(" 7. lambda2 (regularize embedding matrix (for VBPR)\n");
		// printf(" 8. Max #iter \n");
		// printf(" 9. corpus name \n");
		// exit(1);
	// }

	// char* reviewPath = argv[1];
	// char* imgFeatPath = argv[2];
	// int K  = atoi(argv[3]);
	// int K2 = atoi(argv[4]);
	// double biasReg = atof(argv[5]);
	// double lambda = atof(argv[6]);
	// double lambda2 = atof(argv[7]);
	// int iter = atoi(argv[8]);
	// char* corpName = argv[9];
    
    char* reviewPath = "./dataset/reviews_Women.txt";
	int K  = 20;
	int K2 = 20;
	double biasReg = 1;
	double lambda = 10;
	double lambda2 = 0;
	int iter = 20;
	char* corpName = "Amazon_Women";

	fprintf(stderr, "{\n");
	fprintf(stderr, "  \"corpus\": \"%s\",\n", reviewPath);

	corpus corp;
	corp.loadData(reviewPath, 5, 0);

	go_POP(&corp);
	//go_MMMF(&corp, K, lambda, biasReg, iter, corpName);
	//go_BPRMF(&corp, K, lambda, biasReg, iter, corpName);
	//go_VBPR(&corp, K, K2, lambda, lambda2, biasReg, iter, corpName);
    corp.save2Txt();
	corp.cleanUp();
	fprintf(stderr, "}\n");
	return 0;
}
