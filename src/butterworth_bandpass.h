#ifndef BUTTERWORTH_BANDPASS_H
#define BUTTERWORTH_BANDPASS_H

#include "common.h"

#include <memory>
#include <cstdlib>

namespace phase_picker {

    struct butter_bandpass_coefficients {
        double *dcof = nullptr;
        double *ccof = nullptr;
        int n = 0;
        
        ~butter_bandpass_coefficients() {
            free(dcof);
            free(ccof);
        }
    };

    double *binomial_mult(int n, double *p);
    double *trinomial_mult(int n, double *b, double *c);

    double *dcof_bwlp(int n, double fcf);
    double *dcof_bwhp(int n, double fcf);
    double *dcof_bwbp(int n, double f1f, double f2f);
    double *dcof_bwbs(int n, double f1f, double f2f);

    int *ccof_bwlp( int n );
    int *ccof_bwhp( int n );
    double *ccof_bwbp( int n );
    double *ccof_bwbs( int n, double f1f, double f2f );

    double sf_bwlp( int n, double fcf );
    double sf_bwhp( int n, double fcf );
    double sf_bwbp( int n, double f1f, double f2f );
    double sf_bwbs( int n, double f1f, double f2f );

    std::unique_ptr<butter_bandpass_coefficients> bwbp_args(int n, double f1f, double f2f);
}

#endif //BUTTERWORTH_BANDPASS_H
