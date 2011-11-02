//============================================================================
// Name        : terminator_in_c.cpp
// Author      : freiz
// Version     :
// Copyright   : GPL V2
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <tr1/unordered_map>
#include <vector>

#include "commen.h"
#include "classifier.h"
using namespace std;

int main(int argc, char** argv)
{
    clock_t begin = clock();
    bool active = false;
    int allowrence = 0;
    if (argc == 5)
    {
        active = true;
        allowrence = atoi(argv[4]);
    }

    string
    help_text =
        "Terminator is a utility to filtering spam with powerful combined model.\n"
        "\n[Usage]:\n"
        "\tFor Task -- \"full\" || \"delayed\" || \"partial\"\n"
        "\t\tterminator [model_type] index_dir result_file\n"
        "\tFor Task -- \"active learning\"\n"
        "\t\tterminator [model_type] index_dir result_file alllowrence \n"
        "\n[model_type]: {0, 1, 2, 3}\n"
        "\t0 -- for all filters combined\n"
        "\t1 -- for best six filters combined\n"
        "\t2 -- for best five filters combined\n"
        "\t3 -- for only discriminative filters combined\n"
        "\t4 -- not so naive bayes -- <<Not So Naive Online Bayesian Spam Filter>>\n"
        "\t5 -- hit classifier -- <<Joint NLP lab between HIT2 at CEAS Spam-filter Challenge 2008>>\n"
        "\t6 -- online logistic regression -- <<Online Discriminative Spam Filter training>>\n"
        "\t7 -- passive-regressive -- <<Large Scale Learning to Rank>>\n"
        "\t8 -- perceptron with margins -- <<Spam Filtering using Inexact String Matching in Explicit Feature Space with On-Line Linear Classifiers>>\n"
        "\t9 -- confidence-weighted -- <<Confidence-Weighted Linear Classification>>\n"
        "\n[Examples]:\n"
        "\t ./terminator 0 ~/Corpus/trec05p/full/ trec05p-full.result\n"
        "\t ./terminator 1 ~/Corpus/trec05p/delayed/ terc05p-delayed.result\n"
        "\t ./terminator 2 ~/Corpus/trec05p/full/ trec05p-active.result 1000\n";

    string
    version_text =
        "Version -- 0.9.5\tAuthor -- freiz || E-mail -- freizsu@gmail.com\n"
        "Developed at Room 313 CaoGuangBiao building, Yuquan Campus of Zhejiang University.\n";
    if (argc == 2)
    {
        string param = string(argv[1]);
        if (param == "--help" || param == "-help" || param == "-h")
        {
            cout << help_text;
        }
        else if (param == "--version" || param == "-version" || param == "-v")
        {
            cout << version_text;
        }
        else
        {
            cout << "Bad Parameters...exit" << endl;
        }
        exit(1);
    }
    if (argc == 1)
    {
        cout << help_text << endl << version_text;
        exit(0);
    }

    int mark = atoi(argv[1]);

    tr1::unordered_map<string, node> weights;
    tr1::unordered_map<string, ptr_node> tmp_weights;

    string corpus_path = string(argv[2]);
    char *result_path = argv[3];

    string index_path = corpus_path + "index";
    ifstream index;
    index.open(index_path.c_str());
    string email_type, email_path;
    char buff_output[1 * 1024 * 1024];
    FILE* output = fopen(result_path, "w");
    setvbuf(output, buff_output, _IOFBF, 1 * 1024 * 1024);
    char buff[MAX_READ_LENGTH];
    char buff_result[200];
    vector<string> email_type_list;
    vector<string> email_path_list;
    while (index >> email_type >> email_path)
    {
        email_type_list.push_back(email_type);
        email_path_list.push_back(email_path);
    }
    int total_emails = email_path_list.size();
    int email_count = 0;

    double* second_layer_weights = (double*)malloc(filter_set[mark].num_of_classifiers * sizeof(double));
    for (int i = 0; i < filter_set[mark].num_of_classifiers; i++)
    {
        second_layer_weights[i] = 1.0;
    }

    while (email_count < total_emails)
    {
        email_type = email_type_list[email_count];
        email_path = email_path_list[email_count];
        ++email_count;
        // if (email_count % 5000 == 0) {
        //     clock_t now = clock();
        //     printf("Processing -- %5.2f%%\t\tElapsed Time -- %6.1f (s)",
        //             (email_count + 0.0) / total_emails * 100, (now - begin
        //                     + 0.0) / CLOCKS_PER_SEC);
        //     if (active == true)
        //         printf("\t\t Allowrence remaining -- %4d\n", allowrence);
        //     else
        //         printf("%s", "\n");
        // }
        memset(buff, 0, MAX_READ_LENGTH);
        email_path = corpus_path + email_path;
        ifstream email;
        email.open(email_path.c_str());
        email.read(buff, MAX_READ_LENGTH);
        email.close();

        vectorization(string(buff), tmp_weights, weights);
        double score = 0.5;
        if (email_type == "ham" || email_type == "spam" || email_type == "Ham"
                || email_type == "Spam")
        {
            score = predict(tmp_weights, filter_set[mark], second_layer_weights);
            string prediction, judge;
            if (score > THRESHOLD)
                prediction = "spam";
            else
                prediction = "ham";
            if (email_type == "ham" || email_type == "Ham")
                judge = "ham";
            if (email_type == "spam" || email_type == "Spam")
                judge = "spam";
            int result_length = sprintf(buff_result,
                                        "%s judge=%s class=%s score=%.8f remain=%d\n",
                                        email_path.c_str(), judge.c_str(), prediction.c_str(),
                                        score, allowrence);
            fwrite(buff_result, 1, result_length, output);
        }

        if (active == false || (active == true && allowrence >= 0 && score
                                > 0.5 - ACTIVE_THRESHOLD && score < 0.5 + ACTIVE_THRESHOLD))
        {
            if (email_type == "ham" || email_type == "HAM")
                train(tmp_weights, "ham", filter_set[mark], second_layer_weights);
            if (email_type == "spam" || email_type == "SPAM")
                train(tmp_weights, "spam", filter_set[mark], second_layer_weights);
            allowrence--;
        }

        tmp_weights.clear();
    }

    index.close();

    clock_t end = clock();
    // fprintf(stderr, "Time: %.2f\n", (end - begin + 0.0) / CLOCKS_PER_SEC);
    // fprintf(stderr, "Average: %.1f (e-mails per second)\n", email_count / ((end
    //         - begin + 0.0) / CLOCKS_PER_SEC));

    for (int i = 0; i < filter_set[mark].num_of_classifiers; i++)
    {
        cout << second_layer_weights[i] << " ";
    }
    cout << endl;

    fclose(output);
    return EXIT_SUCCESS;
}
