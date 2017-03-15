#include "common.hpp"

/// Safely open a file
FILE* fopen_(const char* p, const char* m)
{
	FILE* f = fopen(p, m);
	if (!f) {
		printf("Failed to open %s\n", p);
		exit(1);
	}
	return f;
}
