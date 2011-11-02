#include "classifier.h"

double cw_predict(tr1::unordered_map<string, ptr_node>& tmp_weights)
{
    double score = 0.0;
    tr1::unordered_map<string, ptr_node>::const_iterator iter;
    for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
    {
        score += (iter->second)->cw;
    }
    score = logist(score / cw_shift);
    return score;
}

void cw_train(tr1::unordered_map<string, ptr_node>& tmp_weights,
              string email_type)
{
    int label;
    if (email_type == "spam")
        label = 1;
    else
        label = -1;
    double score = cw_predict(tmp_weights);
    int count = 0;
    while (label == 1 && score < 0.5 + cw_thickness && count < cw_max_iters
            || label == -1 && score > 0.5 - cw_thickness && count
            < cw_max_iters)
    {
        double m = 0.0;
        double v = 0.0;
        tr1::unordered_map<string, ptr_node>::const_iterator iter;
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            m += (iter->second)->cw;
            v += (iter->second)->cw_sigma;
        }
        m *= label;
        double tmp = (1 + 2 * m * cw_confidence);
        double lambda = (-tmp + sqrt(tmp * tmp - 8 * cw_confidence * (m
                                     - cw_confidence * v))) / (4 * v * cw_confidence);
        if (lambda < 0)
            lambda = 0.0;
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->cw += lambda * label * (iter->second)->cw_sigma;
            (iter->second)->cw_sigma = 1.0 / (1.0 / ((iter->second)->cw_sigma)
                                              + 2 * lambda * cw_confidence);
        }
        score = cw_predict(tmp_weights);
        count++;
    }
}

double nb_predict(tr1::unordered_map<string, ptr_node>& tmp_weights)
{
    double score = 0.0;
    tr1::unordered_map<string, ptr_node>::const_iterator iter;
    int s, h;
    for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
    {
        s = (iter->second)->nb_spam;
        h = (iter->second)->nb_ham;
        if (s == 0 && h == 0)
            continue;
        score += log((s + nb_smooth) / (h + nb_smooth) * (total_ham + 2
                     * nb_smooth) / (total_spam + 2 * nb_smooth));
    }
    score += log((total_spam + nb_smooth) / (total_ham + nb_smooth));
    score = logist(score / nb_shift);
    return score;
}

void nb_train_cell(tr1::unordered_map<string, ptr_node>& tmp_weights,
                   string email_type)
{
    tr1::unordered_map<string, ptr_node>::const_iterator iter;
    if (email_type == "spam")
    {
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->nb_spam += nb_increasing;
        }
    }
    else
    {
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->nb_ham += nb_increasing;
        }
    }
}

void nb_train(tr1::unordered_map<string, ptr_node>& tmp_weights,
              string email_type)
{
    tr1::unordered_map<string, ptr_node>::const_iterator iter;
    double score = nb_predict(tmp_weights);
    int count = 0;
    while (email_type == "spam" && score < 0.5 + nb_thickness && count
            < nb_max_iters)
    {
        nb_train_cell(tmp_weights, email_type);
        score = nb_predict(tmp_weights);
        count++;
    }
    count = 0;
    while (email_type == "ham" && score > 0.5 - nb_thickness && count
            < nb_max_iters)
    {
        nb_train_cell(tmp_weights, email_type);
        score = nb_predict(tmp_weights);
        count++;
    }
}

double hit_predict(tr1::unordered_map<string, ptr_node>& tmp_weights)
{
    double score = 0.0;
    tr1::unordered_map<string, ptr_node>::const_iterator iter;
    for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
    {
        score += (iter->second)->hit;
    }
    score = logist(score / hit_shift);
    return score;
}

void hit_train(tr1::unordered_map<string, ptr_node>& tmp_weights,
               string email_type)
{
    tr1::unordered_map<string, ptr_node>::const_iterator iter;
    if (email_type == "spam")
    {
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->hit_spam += 1;
        }
    }
    else
    {
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->hit_ham += 1;
        }
    }
    double score = 0.0;
    double ratio, p;
    score = hit_predict(tmp_weights);
    int count = 0;
    while (email_type == "spam" && score < 0.5 + hit_thickness && count
            < hit_max_iters)
    {
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            p = ((iter->second)->hit_spam + hit_smooth)
                / ((iter->second)->hit_ham + (iter->second)->hit_spam + 2.0
                   * hit_smooth);
            ratio = fabs(2 * p - 1.0);
            (iter->second)->hit += (1.0 - score) * hit_rate;
            (iter->second)->hit *= ratio;
        }
        score = hit_predict(tmp_weights);
        count += 1;
    }
    count = 0;
    while (email_type == "ham" && score > 0.5 - hit_thickness && count
            < hit_max_iters)
    {
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            p = ((iter->second)->hit_spam + hit_smooth)
                / ((iter->second)->hit_ham + (iter->second)->hit_spam + 2.0
                   * hit_smooth);
            ratio = fabs(2 * p - 1.0);
            (iter->second)->hit -= score * hit_rate;
            (iter->second)->hit *= ratio;
        }
        score = hit_predict(tmp_weights);
        count += 1;
    }
}

double winnow_predict(tr1::unordered_map<string, ptr_node>& tmp_weights)
{
    double score = 0.0;
    tr1::unordered_map<string, ptr_node>::const_iterator iter;
    for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
    {
        score += (iter->second)->winnow;
    }
    score /= tmp_weights.size();
    score -= winnow_threshold;
    score = logist(score / winnow_shift);
    return score;
}

void winnow_train(tr1::unordered_map<string, ptr_node>& tmp_weights,
                  string email_type)
{
    tr1::unordered_map<string, ptr_node>::const_iterator iter;
    double score = winnow_predict(tmp_weights);

    int count = 0;
    if (email_type == "spam" && score < 0.5 + winnow_thickness && count
            < winnow_max_iters)
    {
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->winnow *= winnow_alpha;
        }
        count++;
    }

    count = 0;
    if (email_type == "ham" && score > 0.5 - winnow_thickness && count
            < winnow_max_iters)
    {
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->winnow *= winnow_beta;
        }
        count++;
    }
}

double pa_predict(tr1::unordered_map<string, ptr_node>& tmp_weights)
{
    double score = 0.0;
    tr1::unordered_map<string, ptr_node>::const_iterator iter;
    for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
    {
        score += (iter->second)->pa;
    }
    score = logist(score / pa_shift);
    return score;
}

void pa_train(tr1::unordered_map<string, ptr_node>& tmp_weights,
              string email_type)
{
    int label;
    if (email_type == "spam")
        label = 1;
    else
        label = -1;
    double score = 0.0;
    tr1::unordered_map<string, ptr_node>::iterator iter;
    for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
    {
        score += (iter->second)->pa;
    }
    // hinge loss
    double loss = 0 > (1.0 - label * score) ? 0 : (1.0 - label * score);
    double tol = loss / tmp_weights.size();
    for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
    {
        (iter->second)->pa += label * tol;
    }
}

double pam_predict(tr1::unordered_map<string, ptr_node>& tmp_weights)
{
    double score = 0.0;
    tr1::unordered_map<string, ptr_node>::const_iterator iter;
    for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
    {
        score += (iter->second)->pam;
    }
    score = logist(score / pam_shift);
    return score;
}

void pam_train(tr1::unordered_map<string, ptr_node>& tmp_weights,
               string email_type)
{
    int label;
    if (email_type == "spam")
        label = 1;
    else
        label = -1;
    double score = 0.0;
    tr1::unordered_map<string, ptr_node>::iterator iter;
    for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
    {
        score += (iter->second)->pam;
    }
    // hinge loss
    int count = 0;
    while (label * score < 1.0 && count < pam_max_iters)
    {
        double tol = pam_lambda / tmp_weights.size();
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->pam += label * tol;
        }
        score = 0.0;
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            score += (iter->second)->pam;
        }
        count++;
    }
}

double nsnb_predict(tr1::unordered_map<string, ptr_node>& tmp_weights)
{
    double score = 0.0;
    int s, h;
    tr1::unordered_map<string, ptr_node>::const_iterator iter;
    for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
    {
        s = (iter->second)->nsnb_spam;
        h = (iter->second)->nsnb_ham;
        if (s == 0 && h == 0)
            continue;
        score += log((s + nsnb_smooth) / (h + nsnb_smooth) * (total_ham + 2
                     * nsnb_smooth) / (total_spam + 2 * nsnb_smooth)
                     * (iter->second)->nsnb_confidence);
    }
    score += log((total_spam + nsnb_smooth) / (total_ham + nsnb_smooth));
    score = logist(score / nsnb_shift);
    return score;
}

void nsnb_train_cell(tr1::unordered_map<string, ptr_node>& tmp_weights,
                     string email_type)
{
    tr1::unordered_map<string, ptr_node>::iterator iter;
    if (email_type == "spam")
    {
        total_spam += 1;
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->nsnb_spam += 1;
        }
    }
    else
    {
        total_ham += 1;
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->nsnb_ham += 1;
        }
    }
}
void nsnb_train(tr1::unordered_map<string, ptr_node>& tmp_weights,
                string email_type)
{
    tr1::unordered_map<string, ptr_node>::iterator iter;
    double score = nsnb_predict(tmp_weights);
    if (email_type == "spam")
    {
        total_spam += 1;
    }
    else
    {
        total_ham += 1;
    }
    int count = 0;
    while (email_type == "spam" && score < 0.5 + nsnb_thickness && count < nsnb_max_iters)
    {
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->nsnb_confidence /= nsnb_learning_rate;
        }
        nsnb_train_cell(tmp_weights, email_type);
        total_spam -= 1;
        score = nsnb_predict(tmp_weights);
        count++;
    }
    count = 0;
    while (email_type == "ham" && score > 0.5 - nsnb_thickness && count < nsnb_max_iters)
    {
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->nsnb_confidence *= nsnb_learning_rate;
        }
        nsnb_train_cell(tmp_weights, email_type);
        total_ham -= 1;
        score = nsnb_predict(tmp_weights);
        count++;
    }
}

double logist_predict(tr1::unordered_map<string, ptr_node>& tmp_weights)
{
    double logist_score = 0.0;
    tr1::unordered_map<string, ptr_node>::const_iterator iter =
        tmp_weights.begin();
    while (iter != tmp_weights.end())
    {
        logist_score += (iter->second)->logist;
        ++iter;
    }
    logist_score = logist(logist_score / logist_shift);
    return logist_score;
}

double bwinnow_predict(tr1::unordered_map<string, ptr_node>& tmp_weights)
{
    double bwinnow_score = 0.0;
    tr1::unordered_map<string, ptr_node>::const_iterator iter =
        tmp_weights.begin();
    while (iter != tmp_weights.end())
    {
        bwinnow_score += (iter->second)->bwinnow_upper
                         - (iter->second)->bwinnow_lower;
        ++iter;
    }
    bwinnow_score /= tmp_weights.size();
    bwinnow_score -= bwinnow_threshold;
    bwinnow_score = logist(bwinnow_score / bwinnow_shift);
    return bwinnow_score;
}

void logist_train(tr1::unordered_map<string, ptr_node>& tmp_weights,
                  string email_type)
{
    double logist_score = logist_predict(tmp_weights);
    tr1::unordered_map<string, ptr_node>::iterator iter;
    int count = 0;

    while (email_type == "spam" && logist_score <= 0.5 + logist_thickness && count < logist_max_iters)
    {
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->logist += (1.0 - logist_score)
                                      * logist_learning_rate;
        }
        logist_score = logist_predict(tmp_weights);
        count++;
    }
    count = 0;
    while (email_type == "ham" && logist_score >= 0.5 - logist_thickness && count < logist_max_iters)
    {
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->logist -= logist_score * logist_learning_rate;
        }
        logist_score = logist_predict(tmp_weights);
        count++;
    }
}

void bwinnow_train(tr1::unordered_map<string, ptr_node>& tmp_weights,
                   string email_type)
{
    double bwinnow_score = bwinnow_predict(tmp_weights);
    tr1::unordered_map<string, ptr_node>::iterator iter;
    int count = 0;

    while (email_type == "spam" && bwinnow_score <= 0.5 + bwinnow_thickness && count < bwinnow_max_iters)
    {
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->bwinnow_upper *= bwinnow_alpha;
            (iter->second)->bwinnow_lower *= bwinnow_beta;
        }
        bwinnow_score = bwinnow_predict(tmp_weights);
        count++;
    }
    count = 0;
    while (email_type == "ham" && bwinnow_score >= 0.5 - bwinnow_thickness && count < bwinnow_max_iters)
    {
        for (iter = tmp_weights.begin(); iter != tmp_weights.end(); ++iter)
        {
            (iter->second)->bwinnow_upper *= bwinnow_beta;
            (iter->second)->bwinnow_lower *= bwinnow_alpha;
        }
        bwinnow_score = bwinnow_predict(tmp_weights);
        count++;
    }
}

/**
 Voting
*/

// double predict(tr1::unordered_map<string, ptr_node>& tmp_weights,
//                const combined_setting& setting)
// {
//     double final_score = 0.0;
//     double ham_score = 0.0;
//     double spam_score = 0.0;
//     int ham_number = 0;
//     int spam_number = 0;
//     for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//     {
//         double tmp_score = (setting.predictors[i])(tmp_weights);
//         if (tmp_score <= 0.5)
//         {
//             ham_number += 1;
//             ham_score += tmp_score;
//         }
//         else
//         {
//             spam_number += 1;
//             spam_score += tmp_score;
//         }
//     }
//     if (ham_number >= spam_number)
//         final_score = ham_score / ham_number;
//     else
//         final_score = spam_score / spam_number;
//     return final_score;
// }
//
// void train(tr1::unordered_map<string, ptr_node>& tmp_weights,
//            string email_type, const combined_setting& setting)
// {
//     for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//     {
//         (setting.trainers[i])(tmp_weights, email_type);
//     }
// }

// double predict(tr1::unordered_map<string, ptr_node>& tmp_weights,
//                const combined_setting& setting, double* weights)
// {
//     double final_score = 0.0;
//     for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//     {
//         final_score += (setting.predictors[i])(tmp_weights) * weights[i];
//     }
//     return final_score;
// }
//
// void train(tr1::unordered_map<string, ptr_node>& tmp_weights,
//            string email_type, const combined_setting& setting, double* weights)
// {
//
//     int label;
//     if (email_type == "spam")
//         label = 1;
//     else
//         label = -1;
//
//     double* scores = (double*)malloc(setting.num_of_classifiers * sizeof(double));
//     for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//     {
//         scores[i] = (setting.predictors[i])(tmp_weights);
//     }
//
//     double score = 0.0;
//     for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//     {
//         score += scores[i] * weights[i];
//     }
//
//     double loss = 0 > (1.0 - label * score) ? 0 : (1.0 - label * score);
//     double all = 0;
//      for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//          all += scores[i];
//     double tol = loss / all;
//     for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//     {
//         weights[i] += scores[i] * tol * label;
//     }
//     double total_weights = 0.0;
//     for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//     {
//         total_weights += weights[i];
//     }
//     for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//     {
//         weights[i] /= total_weights;
//     }
//
//     for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//     {
//         (setting.trainers[i])(tmp_weights, email_type);
//     }
// }

// double predict(tr1::unordered_map<string, ptr_node>& tmp_weights,
//             const combined_setting& setting, double* weights)
// {
//     double final_score = 0.0;
//     double total_weights = 0.0;
//     for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//     {
//         final_score += (setting.predictors[i])(tmp_weights) * weights[i];
//         total_weights += weights[i];
//     }
//     return final_score / total_weights;
// }
//
// void train(tr1::unordered_map<string, ptr_node>& tmp_weights,
//             string email_type, const combined_setting& setting, double* weights)
// {
//      for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//      {
//          (setting.trainers[i])(tmp_weights, email_type);
//      }
// }

/**
Best ever, weighted!
*/
double predict(tr1::unordered_map<string, ptr_node>& tmp_weights,
               const combined_setting& setting, double* weights)
{
    double final_score = 0.0;
    double total_weights = 0.0;
    for (unsigned i = 0; i < setting.num_of_classifiers; i++)
    {
        final_score += (setting.predictors[i])(tmp_weights) * weights[i];
        total_weights += weights[i];
    }
    return final_score / total_weights;
}

void train(tr1::unordered_map<string, ptr_node>& tmp_weights,
           string email_type, const combined_setting& setting, double* weights)
{
    double lambda = 0.02;
    double final_score = 0.0;
    double total_weights = 0.0;
    double* scores = (double*)malloc(setting.num_of_classifiers * sizeof(double));
    for (unsigned i = 0; i < setting.num_of_classifiers; i++)
    {
        scores[i] = (setting.predictors[i])(tmp_weights);
        final_score +=  scores[i] * weights[i];
        total_weights += weights[i];
    }
    final_score /= total_weights;
    if (email_type == "spam")
    {
        if (final_score > 0.5)
        {
            for (unsigned i = 0; i < setting.num_of_classifiers; i++)
            {
                if (scores[i] <= 0.5)
                    weights[i] -= lambda;
            }
        }
        else
        {
            for (unsigned i = 0; i < setting.num_of_classifiers; i++)
            {
                if (scores[i] > 0.5)
                    weights[i] += 20 * lambda;
            }
        }
    }
    else
    {
        if (final_score > 0.5)
        {
            for (unsigned i = 0; i < setting.num_of_classifiers; i++)
            {
                if (scores[i] <= 0.5)
                    weights[i] += 20 * lambda;
            }
        }
        else
        {
            for (unsigned i = 0; i < setting.num_of_classifiers; i++)
            {
                if (scores[i] > 0.5)
                    weights[i] -= lambda;
            }
        }
    }

    for (unsigned i = 0; i < setting.num_of_classifiers; i++)
    {
        (setting.trainers[i])(tmp_weights, email_type);
    }
}


/**
logist regression
*/

// double predict(tr1::unordered_map<string, ptr_node>& tmp_weights,
//                const combined_setting& setting, double* weights)
// {
//     double final_score = 0.0;
//     for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//     {
//         final_score += (setting.predictors[i])(tmp_weights) * weights[i];
//     }
//     return logist(final_score);
// }
//
// void train(tr1::unordered_map<string, ptr_node>& tmp_weights,
//            string email_type, const combined_setting& setting, double* weights)
// {
//     double lambda = 1;
//     double final_score = 0.0;
//     double* scores = (double*)malloc(setting.num_of_classifiers * sizeof(double));
//     for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//     {
//         scores[i] = (setting.predictors[i])(tmp_weights);
//         final_score +=  scores[i] * weights[i];
//     }
//     final_score = logist(final_score);
//     if (email_type == "spam")
//     {
//         for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//         {
//             weights[i] += (1-final_score) * lambda * scores[i];
//         }
//     }
//     else
//     {
//         for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//         {
//             weights[i] -= final_score * lambda * scores[i];
//         }
//     }
//
//     for (unsigned i = 0; i < setting.num_of_classifiers; i++)
//     {
//         (setting.trainers[i])(tmp_weights, email_type);
//     }
// }