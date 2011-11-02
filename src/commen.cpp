#include <cmath>
#include "commen.h"
using namespace std;

void vectorization(string email_content,
                   tr1::unordered_map<string, ptr_node>& tmp_weights, tr1::unordered_map<string, node>& weights)
{
    int len = email_content.length();
    for (int i = 0; i <= len - NGRAM; i++)
    {
        string feature = email_content.substr(i, NGRAM);
        if (weights.count(feature) == 0)
        {
            weights[feature]
            = (node)
            {
                0.0, 2.0, 1.0, 0, 0, 1.0, 0.0, 0.0, 1.0, 0, 0, 0.0, 0, 0, 0.0, 1.0
            };
        }
        tmp_weights[feature] = &weights[feature];
    }
}
