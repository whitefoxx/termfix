#ifndef COMMEN_H_
#define COMMEN_H_

#define NGRAM (4)
#define THRESHOLD (0.625)
#define MAX_READ_LENGTH (3000)
#define ACTIVE_THRESHOLD (0.118)

#include <tr1/unordered_map>
#include <string>

using namespace std;

struct Node
{
    float logist;
    float bwinnow_upper;
    float bwinnow_lower;
    int nsnb_spam;
    int nsnb_ham;
    double nsnb_confidence;
    float pam;
    float pa;
    float winnow;
    int hit_spam;
    int hit_ham;
    float hit;
    int nb_spam;
    int nb_ham;
    float cw;
    float cw_sigma;
};

typedef struct Node node;
typedef struct Node * ptr_node;

extern void vectorization(string email_content, tr1::unordered_map<string,
                          ptr_node>& tmp_weights, tr1::unordered_map<string, node>& weights);

inline double logist(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

inline double invlogist(double x)
{
    return log(x/(1-x));
}

#endif /* COMMEN_H_ */
