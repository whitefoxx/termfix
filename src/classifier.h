#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include <string>
#include <cmath>
#include <tr1/unordered_map>

#include "commen.h"

static const double thickness = 0.25;

static const double logist_learning_rate = 0.01;
static const double logist_shift = 10;
static const double logist_thickness = 0.20;
static const double logist_max_iters = 200;

static const double bwinnow_alpha = 1.11;
static const double bwinnow_beta = 0.89;
static const double bwinnow_shift = 1;
static const double bwinnow_threshold = 1.0;
static const double bwinnow_thickness = 0.10;
static const double bwinnow_max_iters = 200;

static const double nsnb_shift = 3200;
static const double nsnb_smooth = 1e-5;
static const double nsnb_thickness = 0.25;
static const double nsnb_learning_rate = 0.65;
static const double nsnb_max_iters = 250;

static int total_ham = 0;
static int total_spam = 0;

static const double winnow_threshold = 1.0;
static const double winnow_shift = 1;
static const double winnow_thickness = 0.1;
static const double winnow_alpha = 1.23;
static const double winnow_beta = 0.83;
static const int winnow_max_iters = 20;

static const double hit_rate = 0.01;
static const double hit_shift = 60;
static const double hit_thickness = 0.27;
static const double hit_smooth = 1e-5;
static const int hit_max_iters = 250; // best 250

static const double nb_shift = 3200;
static const double nb_smooth = 1e-5;
static const double nb_thickness = 0.25;
static const int nb_increasing = 15;
static const int nb_max_iters = 20;

static const double pa_shift = 1.0;

static const double pam_shift = 1.25;
static const double pam_lambda = 0.1;
static const int pam_max_iters = 200; // best 200

static const double cw_confidence = 0.05;
static const double cw_shift = 40;
static const double cw_thickness = 0.25;
static const int cw_max_iters = 100;

extern double cw_predict(tr1::unordered_map<string, ptr_node>&);
extern void cw_train(tr1::unordered_map<string, ptr_node>&, string);

extern double
logist_predict(tr1::unordered_map<string, ptr_node>&);
extern void logist_train(tr1::unordered_map<string, ptr_node>&, string);

extern double
bwinnow_predict(tr1::unordered_map<string, ptr_node>&);
extern void bwinnow_train(tr1::unordered_map<string, ptr_node>&, string);

extern double nsnb_predict(tr1::unordered_map<string, ptr_node>&);
extern void nsnb_train_cell(tr1::unordered_map<string, ptr_node>&, string);
extern void nsnb_train(tr1::unordered_map<string, ptr_node>&, string);

extern double pam_predict(tr1::unordered_map<string, ptr_node>&);
extern void pam_train(tr1::unordered_map<string, ptr_node>&, string);

extern double pa_predict(tr1::unordered_map<string, ptr_node>&);
extern void pa_train(tr1::unordered_map<string, ptr_node>&, string);

extern double winnow_predict(tr1::unordered_map<string, ptr_node>&);
extern void winnow_train(tr1::unordered_map<string, ptr_node>&, string);

extern double hit_predict(tr1::unordered_map<string, ptr_node>&);
extern void hit_train(tr1::unordered_map<string, ptr_node>&, string);

extern double nb_predict(tr1::unordered_map<string, ptr_node>&);
extern void nb_train(tr1::unordered_map<string, ptr_node>&, string);
extern void nb_train_cell(tr1::unordered_map<string, ptr_node>&, string);

static struct combined_setting
{
    double
    (*predictors[10])(tr1::unordered_map<string, ptr_node>&);
    void
    (*trainers[10])(tr1::unordered_map<string, ptr_node>&, string);
    unsigned num_of_classifiers;
} filter_set[] =
{
    {   {nb_predict, hit_predict, winnow_predict, bwinnow_predict, nsnb_predict, pa_predict, pam_predict, logist_predict},
        {nb_train, hit_train, winnow_train, bwinnow_train, nsnb_train, pa_train, pam_train, logist_train},
        8
    }, {{
            nsnb_predict
        }, {
            nsnb_train
        }, 1
    }, { {
            nb_predict, hit_predict
        }, { nb_train, hit_train
           }, 2
    }, { {
            nb_predict, hit_predict, winnow_predict
        }, {
            nb_train, hit_train, winnow_train
        }, 3
    }, { {
            nb_predict, hit_predict, winnow_predict, bwinnow_predict
        }, {
            nb_train, hit_train, winnow_train, bwinnow_train
        }, 4
    }, { {
            nb_predict, hit_predict, winnow_predict, bwinnow_predict, nsnb_predict
        }, { nb_train, hit_train, winnow_train, bwinnow_train, nsnb_train }, 5
    }, { { nb_predict, hit_predict, winnow_predict, bwinnow_predict, nsnb_predict, pa_predict },
        { nb_train, hit_train, winnow_train, bwinnow_train, nsnb_train, pa_train },
        6
    }, { { nb_predict, hit_predict, winnow_predict, bwinnow_predict, nsnb_predict, pa_predict, pam_predict },
        { nb_train, hit_train, winnow_train, bwinnow_train, nsnb_train, pa_train, pam_train },
        7
    }, { { nb_predict }, {
            nb_train
        }, 1
    }, { { nsnb_predict }, { nsnb_train }, 1 
    }, { { winnow_predict }, { winnow_train }, 1
    }, { {bwinnow_predict}, {bwinnow_train}, 1
}, { {logist_predict}, {logist_train}, 1
    }, {{hit_predict}, {hit_train}, 1
    }, {{pa_predict}, {pa_train}, 1
    }, {{pam_predict}, {pam_train}, 1
    }
};

extern double
predict(tr1::unordered_map<string, ptr_node>&, const combined_setting&, double*);
extern void train(tr1::unordered_map<string, ptr_node>& tmp_weights,
                  string email_type, const combined_setting&, double*);

#endif
