// Pi0 leptoproduction in Goloskokov-Kroll (GK) model. The code is currently being tested and implemented in PARTONS framework with additional features. If you plan to use this work in a publication, please use and reference the most recent version of PARTONS in http://partons.cea.fr 

// The code evaluates Pi0 amplitudes and cross sections in GK model. 3-dimensional integrals are evaluated by using the VEGAS Monte-Carlo integration routines implemented in gsl library. 

// Make sure the gsl library is downloaded in your computer. To run the code, in the terminal type: 
// 1) g++ -o Pi0_GK_Vegas.out Pi0_GK_Vegas.cpp -lgsl -lgslcblas -lm 
// 2) ./Pi0_GK_Vegas.out 

// For questions, please write to me at kemaltezgin@gmail.com

#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>  // Needed for atoi and atof functions
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf.h>
#include <cmath>
#include <complex>
#include <utility>

using namespace std;

double PROTON_MASS = 0.938272081;
double leptonEnergy = 10.604; //FIX THIS TO BE THE EXACT CORRECT VALUE
//double leptonEnergy = 10.6; //FIX THIS TO BE THE EXACT CORRECT VALUE



double m_Q2;
// = 2; // Q^2 value
double m_t;
 //= -0.2; // invariant momentum transfer t
double m_xbj;
// =0.12;// Bjorken x

// // W value
// double W = 6.0;
// // Bjorken x
// double m_xbj = m_Q2 / (pow(W, 2.0) + m_Q2 - pow(PROTON_MASS, 2.0));

// Specify an xbjorken value, then calculate W from it (not clear we even need to calculate W, but oh well)


//double W = sqrt(m_Q2 / m_xbj + pow(PROTON_MASS, 2.0) - m_Q2);


double EulerGamma = 0.577216;
double PositronCharge = 0.3028;
double U_ELEC_CHARGE = 2. / 3.;
double D_ELEC_CHARGE = -1. / 3.;
double Nc = 3.;
double m_cNf = 3.;
double m_cLambdaQCD = 0.22;
double decayConstant = 0.132;
double Cf = 4. / 3.;
double transverseSize3 = 1.8;
double muPi = 2.0;




// END OF INPUTS




// Skewness parameter
double m_xi; 
//= m_xbj / (2.0 - m_xbj);

double gammaa; 
//= 2.0 * m_xbj * PROTON_MASS / sqrt(m_Q2);
double y; 
//= ( pow(W, 2.0) + m_Q2 - pow(PROTON_MASS, 2.0) ) / ( 2.0 * PROTON_MASS * leptonEnergy );
double epsilon; 
// = (1.0 - y - 1.0 / 4.0 * pow(y, 2.0) * pow(gammaa, 2.0)) / (1.0 - y + 1.0 / 2.0 * pow(y, 2.0) + 1.0 / 4.0 * pow(y, 2.0) * pow(gammaa, 2.0));




// double m_xbj = 2. * m_xi / (1. + m_xi);

//double accuracy = 0.000001;


double W;
double m_tmin;


// Conversion of units to nb to be used in cross section calculations 
double Conversion;


// Valence quark expansions to be used in the computation of GPDs.
double ValenceExpansion(double x, double i, double k) {
    double dum;

    if ((x + m_xi) < 0.0) { 
        dum = 0.0;
    } else { 
        if ((x - m_xi) < 0.0) {
            dum =
                    3. / 2. / pow(m_xi, 3.)
                            * (pow((x + m_xi) / (1. + m_xi), (2. + i - k))
                                    * (m_xi * m_xi - x
                                            + (2. + i - k) * m_xi * (1. - x)))
                            / (1. + i - k) / (2. + i - k) / (3. + i - k);
        } else {
            dum = 3. / 2. / pow(m_xi, 3.) / (1. + i - k) / (2. + i - k)
                    / (3. + i - k)
                    * ((m_xi * m_xi - x)
                            * (pow((x + m_xi) / (1. + m_xi), (2. + i - k))
                                    - pow((x - m_xi) / (1. - m_xi),
                                            (2. + i - k)))
                            + m_xi * (1. - x) * (2. + i - k)
                                    * (pow((x + m_xi) / (1. + m_xi),
                                            (2. + i - k))
                                            + pow((x - m_xi) / (1. - m_xi),
                                                    (2. + i - k))));
        }
    }
    return dum;
}

// calculation of the GPD \tilde{H} for up quarks
double computeHtUp(double x) {
    
    double Nu, c1, c2, c3, c4, b0;

    double alpha, delta;

// u valence

    alpha = 0.45;
    delta = 0.32;
    double kHtuval = delta + alpha * m_t;

    Nu = 1.0;

    c1 = 0.213 * Nu;
    c2 = 0.929 * Nu;
    c3 = 12.59 * Nu;
    c4 = -12.57 * Nu;

    b0 = 0.59;

    double uVal = exp(b0 * m_t)
            * (c1 * ValenceExpansion(x, 0., kHtuval) + c2 * ValenceExpansion(x, 0.5, kHtuval)
                    + c3 * ValenceExpansion(x, 1.0, kHtuval) + c4 * ValenceExpansion(x, 1.5, kHtuval));

    return uVal;

}

// calculation of the GPD \tilde{H} for down quarks
double computeHtDown(double x) {

    double Nd, c1, c2, c3, c4, b0;

    double alpha, delta;

// d valence

    alpha = 0.45;
    delta = 0.32;
    double kHtdval = delta + alpha * m_t;

    Nd = 1.0;
    c1 = -0.204 * Nd;
    c2 = -0.940 * Nd;
    c3 = -0.314 * Nd;
    c4 = 1.524 * Nd;
    b0 = 0.59;

    double dVal = exp(b0 * m_t)
            * (c1 * ValenceExpansion(x, 0., kHtdval) + c2 * ValenceExpansion(x, 0.5, kHtdval)
                    + c3 * ValenceExpansion(x, 1., kHtdval) + c4 * ValenceExpansion(x, 1.5, kHtdval));

    return dVal;
}

// calculation of the GPD \tilde{E} for up quarks
double computeEtUp(double x) {
    
    double Nu, c1, c2, c3, c4, b0;

    double alpha, delta;

// u valence

    alpha = 0.45;
    delta = 0.48;
    double kEtuval = delta + alpha * m_t;

    Nu = 14.0;
    c1 = Nu;
    c2 = -2. * Nu;
    c3 = Nu;
    c4 = 0.0;
    b0 = 0.9;

    double uVal = exp(b0 * m_t)
            * (c1 * ValenceExpansion(x, 0., kEtuval) + c2 * ValenceExpansion(x, 1.0, kEtuval)
                    + c3 * ValenceExpansion(x, 2.0, kEtuval) + c4 * ValenceExpansion(x, 3.0, kEtuval));

    return uVal;

}

// calculation of the GPD \tilde{E} for down quarks
double computeEtDown(double x) {

    double Nd, c1, c2, c3, c4, b0;

    double alpha, delta;

// d valence

    alpha = 0.45;
    delta = 0.48;
    double kEtdval = delta + alpha * m_t;


    Nd = 4.0;
    c1 = Nd;
    c2 = -2. * Nd;
    c3 = Nd;
    c4 = 0.0;
    b0 = 0.9;

    double dVal = exp(b0 * m_t)
            * (c1 * ValenceExpansion(x, 0., kEtdval) + c2 * ValenceExpansion(x, 1.0, kEtdval)
                    + c3 * ValenceExpansion(x, 2., kEtdval) + c4 * ValenceExpansion(x, 3.0, kEtdval));

    return dVal;
}

// calculation of the GPD H_T for up quarks
double computeHTUp(double x) {
    
    double Nu, c1, c2, c3, c4, c5, c6, b0;

    double alpha, delta;

// u valence

    alpha = 0.45;
    delta = -0.17;
    double kHTransuval = delta + alpha * m_t;

    Nu = 1.1;

    c1 = 3.653;
    c2 = -0.583;
    c3 = 19.807;
    c4 = -23.487;
    c5 = -23.46;
    c6 = 24.07;

    b0 = 0.3;

    double uVal = exp(b0 * m_t) * Nu
            * (c1 * ValenceExpansion(x, 0., kHTransuval) + c2 * ValenceExpansion(x, 0.5, kHTransuval)
                    + c3 * ValenceExpansion(x, 1., kHTransuval) + c4 * ValenceExpansion(x, 1.5, kHTransuval)
                            + c5 * ValenceExpansion(x, 2., kHTransuval) + c6 * ValenceExpansion(x, 2.5, kHTransuval));

    return uVal;

}

// calculation of the GPD H_T for down quarks
double computeHTDown(double x) {

    double Nd, c1, c2, c3, c4, c5, c6, b0;

    double alpha, delta;

// d valence

    alpha = 0.45;
    delta = -0.17;
    double kHTransdval = delta + alpha * m_t;

    Nd = -0.3;

    c1 = 1.924;
    c2 = 0.179;
    c3 = -7.775;
    c4 = 3.504;
    c5 = 5.851;
    c6 = -3.683;

    b0 = 0.3;

    double dVal = exp(b0 * m_t) * Nd
            * (c1 * ValenceExpansion(x, 0., kHTransdval) + c2 * ValenceExpansion(x, 0.5, kHTransdval)
                    + c3 * ValenceExpansion(x, 1., kHTransdval) + c4 * ValenceExpansion(x, 1.5, kHTransdval)
                            + c5 * ValenceExpansion(x, 2., kHTransdval) + c6 * ValenceExpansion(x, 2.5, kHTransdval));

    return dVal;
}

// calculation of the GPD \bar{E}_T for up quarks
double computeEbarUp(double x) {
    
    double Nu, c1, c2, c3, b0;

    double alpha, delta;

// u valence

    alpha = 0.45;
    delta = 0.30;
    double kEbaruval = delta + alpha * m_t;

    Nu = 4.83;
    c1 = Nu;
    c2 = -1. * Nu;
    c3 = 0.0;
    b0 = 0.5;

    double uVal = exp(b0 * m_t)
            * (c1 * ValenceExpansion(x, 0., kEbaruval) + c2 * ValenceExpansion(x, 1., kEbaruval)
                    + c3 * ValenceExpansion(x, 2., kEbaruval));

    return uVal;

}

// calculation of the GPD \bar{E}_T for down quarks
double computeEbarDown(double x) {

    double Nd, c1, c2, c3, c4, b0;

    double alpha, delta;

// d valence

    alpha = 0.45;
    delta = 0.30;
    double kEbardval = delta + alpha * m_t;

    Nd = 3.57;
    c1 = Nd;
    c2 = -2. * Nd;
    c3 = Nd;
    b0 = 0.5;

    double dVal = exp(b0 * m_t)
            * (c1 * ValenceExpansion(x, 0., kEbardval) + c2 * ValenceExpansion(x, 1., kEbardval)
                    + c3 * ValenceExpansion(x, 2., kEbardval));

    return dVal;
}

double computeMuR(double tau, double b) {

    double Q = sqrt(m_Q2);

    double maximum = tau * Q;

    if (1. - tau > tau)
        maximum = (1. - tau) * Q;

    if (1. / b > maximum)
        maximum = 1. / b;

    return maximum;
}


double alphaS(double MuR) {

    double Q = sqrt(m_Q2);

    double coupling = (12.0 * M_PI) / ((33. - 2. * m_cNf) * log(pow(MuR,2.) / pow(m_cLambdaQCD,2.)));

    return coupling;
}

double sudakovFactorFunctionS(double tau, double b) {

    // sudakov function s is described, for example, in the appendix of https://arxiv.org/pdf/hep-ph/9503418.pdf

    double Q = sqrt(m_Q2);

    double sudakov;

    //beta0 factor
    double beta0 = 11. - 2. * m_cNf / 3.;

    //beta1 factor
    double beta1 = 102. - 38. * m_cNf / 3.;

    //b^
    double bHat = -1. * log(b * m_cLambdaQCD);

    //q^
    double qHat = log(tau * Q / (sqrt(2.) * m_cLambdaQCD));

    //A^(2) factor
    double A2 = 67. / 9. - pow(M_PI, 2.) / 3. - 10. / 27. * m_cNf + 2. * beta0 / 3. * log(exp(EulerGamma) / 2.);

    if (b - sqrt(2.) / (tau * Q) <= 0.)
        sudakov = 0.;
    else
        sudakov = 8. / (3. * beta0) * (qHat * log(qHat / bHat) - qHat + bHat)
                    + (4. * beta1) / (3. * pow(beta0,3.)) * (qHat * ((log(2.*qHat)+ 1.)/qHat - (log(2.*bHat)+1.0)/bHat)
                            + 1. / 2. * (pow(log(2. * qHat),2.) - pow(log(2. * bHat),2.)))
                    + 4. / (3. * beta0) * log(exp(2. * EulerGamma - 1.) / 2.) * log(qHat / bHat)
                            + A2 * 4. / pow(beta0, 2.) * ((qHat - bHat) / bHat - log(qHat / bHat));

    return sudakov;

}

double expSudakovFactor(double tau, double b) {

    //sqrt of Q2
    double Q = sqrt(m_Q2);

    double expSudakov;

    //beta0 factor
    double beta0 = 11. - 2. * m_cNf / 3.;

    //b^
    double bHat = -1. * log(b * m_cLambdaQCD);

    //Eq. (12) from https://arxiv.org/pdf/hep-ph/0611290.pdf
    double sudakovFactor =  sudakovFactorFunctionS(tau, b) + sudakovFactorFunctionS(1. - tau, b)
            - (4. / beta0) * log(log(computeMuR(tau, b) / m_cLambdaQCD) / bHat);

    if (exp(-1. * sudakovFactor) >= 1.)
        expSudakov = 1.;
    else
        expSudakov = exp(-1. * sudakovFactor);

    return expSudakov;
}


double mesonWFGaussianTwist2(double tau, double b) {


    double transverseSize2 = 1. / (8. * pow(M_PI, 2.0) * pow(decayConstant, 2.));

    double WFtwist2 = 2. * M_PI * decayConstant / sqrt(2.*Nc) * 6. * tau * (1. - tau) *
            exp(-1. * tau * (1. - tau) * pow(b, 2.0) / (4. * transverseSize2));

    return WFtwist2;

}

double mesonWFGaussianTwist3(double b) {

    double WFtwist3 = 4. * M_PI * decayConstant / sqrt(2. * Nc) * muPi * pow(transverseSize3, 2.) *
            exp(-1.0 * pow(b, 2.) / (8. * pow(transverseSize3, 2.0))) * gsl_sf_bessel_In(0, pow(b, 2.) / (8. * pow(transverseSize3, 2.0)));

    return WFtwist3;

}

std::complex<double> HankelFunctionFirstKind(double z) {

    //This function defines the Hankel Function of the first kind H_0^{(1)}(z) = J_0(z) + i * Y_0(z)

    std::complex<double> Hankel0 = gsl_sf_bessel_J0(z) + 1i * gsl_sf_bessel_Y0(z);

    return Hankel0;
}


std::complex<double> subprocessPi0Twist2(double x, double tau, double b) {

    std::complex<double> Ts, Tu, subprocessPi0Tw2;

    if (x >= m_xi) 
        Ts = -1. * 1i / 4. * (gsl_sf_bessel_J0(sqrt((1. - tau) * (x - m_xi) / (2. * m_xi)) * b * sqrt(m_Q2)) + 
             1i * gsl_sf_bessel_Y0(sqrt((1. - tau) * (x - m_xi) / (2. * m_xi)) * b * sqrt(m_Q2)));
    else 
        Ts = -1. / (2. * M_PI) * gsl_sf_bessel_K0(sqrt((1. - tau) * (m_xi - x) / (2. * m_xi)) * b * sqrt(m_Q2));

    Tu = -1. / (2. * M_PI) * gsl_sf_bessel_K0(sqrt(tau * (x + m_xi) / (2. * m_xi)) * b * sqrt(m_Q2));

    subprocessPi0Tw2 = Cf * sqrt(2. / Nc) * m_Q2 / m_xi * 2. * M_PI *
            b * mesonWFGaussianTwist2(tau, b) * alphaS(computeMuR(tau,b)) * expSudakovFactor(tau, b) * (Ts - Tu);

    return subprocessPi0Tw2;

}

std::complex<double> subprocessPi0Twist3(double x, double tau, double b) {

    std::complex<double> Ts, Tu, subprocessPi0Tw3;

    if (x >= m_xi) 
        Ts = -1. * 1i / 4. * (gsl_sf_bessel_J0(sqrt((1. - tau) * (x - m_xi) / (2. * m_xi)) * b * sqrt(m_Q2)) + 
             1i * gsl_sf_bessel_Y0(sqrt((1. - tau) * (x - m_xi) / (2. * m_xi)) * b * sqrt(m_Q2)));
    else 
        Ts = -1. / (2. * M_PI) * gsl_sf_bessel_K0(sqrt((1. - tau) * (m_xi - x) / (2. * m_xi)) * b * sqrt(m_Q2));

    Tu = -1. / (2. * M_PI) * gsl_sf_bessel_K0(sqrt(tau * (x + m_xi) / (2. * m_xi)) * b * sqrt(m_Q2));

    subprocessPi0Tw3 = 4.0 * Cf / sqrt(2. * Nc) * m_Q2 / m_xi * 2. * M_PI *
            b * mesonWFGaussianTwist3(b) * alphaS(computeMuR(tau,b)) * expSudakovFactor(tau, b) * ((1. - tau) * Ts + tau * Tu);

    return subprocessPi0Tw3;

}


// Calculation of the real part of the convolution < \tilde{H} >. To be inserted below inside the function HtConvolutionPi0. 
double HtConvolutionPi0Re(double *xtaub, size_t dim, void *params) {

    // For pi^0, GPDs appear in the combination of 1/sqrt(2) * (e^u * F^u  - e^d * F^d)

    std::complex<double> convolutionPi0Tw2 = 1. / sqrt(2.) * (U_ELEC_CHARGE * computeHtUp(xtaub[0]) - D_ELEC_CHARGE * computeHtDown(xtaub[0]))
            * subprocessPi0Twist2(xtaub[0], xtaub[1], xtaub[2]);

    return real(convolutionPi0Tw2);
}

// Calculation of the imaginary part of the convolution < \tilde{H} >. To be inserted below inside the function HtConvolutionPi0.
double HtConvolutionPi0Im(double *xtaub, size_t dim, void *params) {

    // For pi^0, GPDs appear in the combination of 1/sqrt(2) * (e^u * F^u  - e^d * F^d)

    std::complex<double> convolutionPi0Tw2 = 1. / sqrt(2.) * (U_ELEC_CHARGE * computeHtUp(xtaub[0]) - D_ELEC_CHARGE * computeHtDown(xtaub[0]))
            * subprocessPi0Twist2(xtaub[0], xtaub[1], xtaub[2]);

    return imag(convolutionPi0Tw2);
}

// Calculation of the convolution < \tilde{H} >. VEGAS Monte-Carlo integration routines are called. To increase the accuracy of the integral, you may need to increase the number of calls in the Monte-Carlo integration.
std::complex<double> HtConvolutionPi0(void) {

    // For pi^0, GPDs appear in the combination of 1/sqrt(2) * (e^u * F^u  - e^d * F^d)

    double rangeMin[3] = { -m_xi, 0.0, 0.0 }; 
    double rangeMax[3] = { 1.0, 1.0, 1.0/m_cLambdaQCD };
    double resultHtRe, errorHtRe, resultHtIm, errorHtIm;

    const size_t nWarmUp = 10000; // Warming up the Monte-Carlo integral
    const size_t nCalls = 100000; // Number of calls in the Monte-Carlo integration 

    gsl_rng* gslRndHtRe;
    const gsl_rng_type* gslRndTypeHtRe;

    gsl_rng* gslRndHtIm;
    const gsl_rng_type* gslRndTypeHtIm;

    gsl_rng_env_setup();

    gslRndTypeHtRe = gsl_rng_default;
    gslRndTypeHtIm = gsl_rng_default;

    gslRndHtRe = gsl_rng_alloc(gslRndTypeHtRe);
    gslRndHtIm = gsl_rng_alloc(gslRndTypeHtIm);

    gsl_monte_function gslFunctionHtRe = {&HtConvolutionPi0Re, 3, 0};
    gsl_monte_function gslFunctionHtIm = {&HtConvolutionPi0Im, 3, 0};

    gsl_monte_vegas_state* gslStateHtRe = gsl_monte_vegas_alloc(3);
    gsl_monte_vegas_state* gslStateHtIm = gsl_monte_vegas_alloc(3);

    //Warm-up
    gsl_monte_vegas_integrate(&gslFunctionHtRe, rangeMin, rangeMax, 3, nWarmUp, gslRndHtRe, gslStateHtRe, &resultHtRe, &errorHtRe);
    gsl_monte_vegas_integrate(&gslFunctionHtIm, rangeMin, rangeMax, 3, nWarmUp, gslRndHtIm, gslStateHtIm, &resultHtIm, &errorHtIm);

    //integrate
    gsl_monte_vegas_integrate(&gslFunctionHtRe, rangeMin, rangeMax, 3, nCalls, gslRndHtRe, gslStateHtRe, &resultHtRe, &errorHtRe);
    gsl_monte_vegas_integrate(&gslFunctionHtIm, rangeMin, rangeMax, 3, nCalls, gslRndHtIm, gslStateHtIm, &resultHtIm, &errorHtIm);

    //free
    gsl_monte_vegas_free(gslStateHtRe);
    gsl_monte_vegas_free(gslStateHtIm);
    gsl_rng_free(gslRndHtRe);
    gsl_rng_free(gslRndHtIm);

    std::complex<double> resultHt = resultHtRe + 1i * resultHtIm;

    return resultHt;

}


double EtConvolutionPi0Re(double *xtaub, size_t dim, void *params) {

    // For pi^0, GPDs appear in the combination of 1/sqrt(2) * (e^u * F^u  - e^d * F^d)

    std::complex<double> convolutionPi0Tw2 = 1. / sqrt(2.) * (U_ELEC_CHARGE * computeEtUp(xtaub[0]) - D_ELEC_CHARGE * computeEtDown(xtaub[0]))
            * subprocessPi0Twist2(xtaub[0], xtaub[1], xtaub[2]);

    return real(convolutionPi0Tw2);
}

double EtConvolutionPi0Im(double *xtaub, size_t dim, void *params) {

    // For pi^0, GPDs appear in the combination of 1/sqrt(2) * (e^u * F^u  - e^d * F^d)

    std::complex<double> convolutionPi0Tw2 = 1. / sqrt(2.) * (U_ELEC_CHARGE * computeEtUp(xtaub[0]) - D_ELEC_CHARGE * computeEtDown(xtaub[0]))
            * subprocessPi0Twist2(xtaub[0], xtaub[1], xtaub[2]);

    return imag(convolutionPi0Tw2);
}

std::complex<double> EtConvolutionPi0(void) {

    // For pi^0, GPDs appear in the combination of 1/sqrt(2) * (e^u * F^u  - e^d * F^d)

    double rangeMin[3] = { -m_xi, 0.0, 0.0 };
    double rangeMax[3] = { 1.0, 1.0, 1.0/m_cLambdaQCD };
    double resultEtRe, errorEtRe, resultEtIm, errorEtIm;

    const size_t nWarmUp = 10000;
    const size_t nCalls = 100000;

    gsl_rng* gslRndEtRe;
    const gsl_rng_type* gslRndTypeEtRe;

    gsl_rng* gslRndEtIm;
    const gsl_rng_type* gslRndTypeEtIm;

    gsl_rng_env_setup();

    gslRndTypeEtRe = gsl_rng_default;
    gslRndTypeEtIm = gsl_rng_default;

    gslRndEtRe = gsl_rng_alloc(gslRndTypeEtRe);
    gslRndEtIm = gsl_rng_alloc(gslRndTypeEtIm);

    gsl_monte_function gslFunctionEtRe = {&EtConvolutionPi0Re, 3, 0};
    gsl_monte_function gslFunctionEtIm = {&EtConvolutionPi0Im, 3, 0};

    gsl_monte_vegas_state* gslStateEtRe = gsl_monte_vegas_alloc(3);
    gsl_monte_vegas_state* gslStateEtIm = gsl_monte_vegas_alloc(3);

    //Warm-up
    gsl_monte_vegas_integrate(&gslFunctionEtRe, rangeMin, rangeMax, 3, nWarmUp, gslRndEtRe, gslStateEtRe, &resultEtRe, &errorEtRe);
    gsl_monte_vegas_integrate(&gslFunctionEtIm, rangeMin, rangeMax, 3, nWarmUp, gslRndEtIm, gslStateEtIm, &resultEtIm, &errorEtIm);

    //integrate
    gsl_monte_vegas_integrate(&gslFunctionEtRe, rangeMin, rangeMax, 3, nCalls, gslRndEtRe, gslStateEtRe, &resultEtRe, &errorEtRe);
    gsl_monte_vegas_integrate(&gslFunctionEtIm, rangeMin, rangeMax, 3, nCalls, gslRndEtIm, gslStateEtIm, &resultEtIm, &errorEtIm);

    //free
    gsl_monte_vegas_free(gslStateEtRe);
    gsl_monte_vegas_free(gslStateEtIm);
    gsl_rng_free(gslRndEtRe);
    gsl_rng_free(gslRndEtIm);

    std::complex<double> resultEt = resultEtRe + 1i * resultEtIm;

    return resultEt;

}

double HTConvolutionPi0Re(double *xtaub, size_t dim, void *params) {

    // For pi^0, GPDs appear in the combination of 1/sqrt(2) * (e^u * F^u  - e^d * F^d)

    std::complex<double> convolutionPi0Tw3 = 1. / sqrt(2.) * (U_ELEC_CHARGE * computeHTUp(xtaub[0]) - D_ELEC_CHARGE * computeHTDown(xtaub[0]))
            * subprocessPi0Twist3(xtaub[0], xtaub[1], xtaub[2]);

    return real(convolutionPi0Tw3);
}

double HTConvolutionPi0Im(double *xtaub, size_t dim, void *params) {

    // For pi^0, GPDs appear in the combination of 1/sqrt(2) * (e^u * F^u  - e^d * F^d)

    std::complex<double> convolutionPi0Tw3 = 1. / sqrt(2.) * (U_ELEC_CHARGE * computeHTUp(xtaub[0]) - D_ELEC_CHARGE * computeHTDown(xtaub[0]))
            * subprocessPi0Twist3(xtaub[0], xtaub[1], xtaub[2]);

    return imag(convolutionPi0Tw3);
}

// This function is to be used below in HTConvolutionPi0. At twist-3, there are 1-dimensional integral terms appearing in the convolution 
double HTConvolutionPi0Analytic (double x, void * params) {

  double alpha = *(double *) params;
  double convolution = 1. / (x + m_xi) * (1. / sqrt(2.) * (U_ELEC_CHARGE * computeHTUp(x) - D_ELEC_CHARGE * computeHTDown(x))) 
			- 1. / (x - m_xi) * (1. / sqrt(2.) * (U_ELEC_CHARGE * computeHTUp(x) - D_ELEC_CHARGE * computeHTDown(x)) 				- 1. / sqrt(2.) * (U_ELEC_CHARGE * computeHTUp(m_xi) - D_ELEC_CHARGE * computeHTDown(m_xi)));

  return 16. * M_PI * Cf / Nc * alphaS(sqrt(m_Q2 / 2.)) * decayConstant * muPi * pow(transverseSize3, 2.) * convolution;
}

std::complex<double> HTConvolutionPi0(void) {

    // For pi^0, GPDs appear in the combination of 1/sqrt(2) * (e^u * F^u  - e^d * F^d)

    std::complex<double> convolutionPi0Tw3;

    std::complex<double> convolutionPi0Tw3Analytic = 16. * M_PI * Cf / Nc * alphaS(sqrt(m_Q2 / 2.)) * decayConstant * muPi
	* pow(transverseSize3, 2.) * (1. / sqrt(2.) * (U_ELEC_CHARGE * computeHTUp(m_xi) - D_ELEC_CHARGE * computeHTDown(m_xi)) 
	* (1i * M_PI - log((1.-m_xi)/(2.*m_xi)))); // Full analytic terms appearing in the convolution 

    double rangeMin[3] = { -m_xi, 0.0, 0.0 };
    double rangeMax[3] = { 1.0, 1.0, 1.0/m_cLambdaQCD };
    double resultHTRe, errorHTRe, resultHTIm, errorHTIm;

    const size_t nWarmUp = 40000;
    const size_t nCalls = 400000;

    gsl_rng* gslRndHTRe;
    const gsl_rng_type* gslRndTypeHTRe;

    gsl_rng* gslRndHTIm;
    const gsl_rng_type* gslRndTypeHTIm;

    gsl_rng_env_setup();

    gslRndTypeHTRe = gsl_rng_default;
    gslRndTypeHTIm = gsl_rng_default;

    gslRndHTRe = gsl_rng_alloc(gslRndTypeHTRe);
    gslRndHTIm = gsl_rng_alloc(gslRndTypeHTIm);

    gsl_monte_function gslFunctionHTRe = {&HTConvolutionPi0Re, 3, 0};
    gsl_monte_function gslFunctionHTIm = {&HTConvolutionPi0Im, 3, 0};

    gsl_monte_vegas_state* gslStateHTRe = gsl_monte_vegas_alloc(3);
    gsl_monte_vegas_state* gslStateHTIm = gsl_monte_vegas_alloc(3);

    //Warm-up
    gsl_monte_vegas_integrate(&gslFunctionHTRe, rangeMin, rangeMax, 3, nWarmUp, gslRndHTRe, gslStateHTRe, &resultHTRe, &errorHTRe);
    gsl_monte_vegas_integrate(&gslFunctionHTIm, rangeMin, rangeMax, 3, nWarmUp, gslRndHTIm, gslStateHTIm, &resultHTIm, &errorHTIm);

    //integrate
    gsl_monte_vegas_integrate(&gslFunctionHTRe, rangeMin, rangeMax, 3, nCalls, gslRndHTRe, gslStateHTRe, &resultHTRe, &errorHTRe);
    gsl_monte_vegas_integrate(&gslFunctionHTIm, rangeMin, rangeMax, 3, nCalls, gslRndHTIm, gslStateHTIm, &resultHTIm, &errorHTIm);

    //free
    gsl_monte_vegas_free(gslStateHTRe);
    gsl_monte_vegas_free(gslStateHTIm);
    gsl_rng_free(gslRndHTRe);
    gsl_rng_free(gslRndHTIm);

    std::complex<double> resultHT = resultHTRe + 1i * resultHTIm;

    // 1D integration begins
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double integration1D, error1D;
    double alpha = 1.0;

    gsl_function gslFunctionHT1D;
    gslFunctionHT1D.function = &HTConvolutionPi0Analytic;
    gslFunctionHT1D.params = &alpha;

    gsl_integration_qags (&gslFunctionHT1D, -m_xi, 1.0, 0, 1e-5, 10000, w, &integration1D, &error1D);

    gsl_integration_workspace_free (w);
    // 1D integration ends
    
    convolutionPi0Tw3 = convolutionPi0Tw3Analytic + integration1D + resultHT;

    return convolutionPi0Tw3;

}


double EbarConvolutionPi0Re(double *xtaub, size_t dim, void *params) {

    // For pi^0, GPDs appear in the combination of 1/sqrt(2) * (e^u * F^u  - e^d * F^d)

    std::complex<double> convolutionPi0Tw3 = 1. / sqrt(2.) * (U_ELEC_CHARGE * computeEbarUp(xtaub[0]) - D_ELEC_CHARGE * computeEbarDown(xtaub[0])) * subprocessPi0Twist3(xtaub[0], xtaub[1], xtaub[2]);

    return real(convolutionPi0Tw3);
}

double EbarConvolutionPi0Im(double *xtaub, size_t dim, void *params) {

    // For pi^0, GPDs appear in the combination of 1/sqrt(2) * (e^u * F^u  - e^d * F^d)

    std::complex<double> convolutionPi0Tw3 = 1. / sqrt(2.) * (U_ELEC_CHARGE * computeEbarUp(xtaub[0]) - D_ELEC_CHARGE * computeEbarDown(xtaub[0])) * subprocessPi0Twist3(xtaub[0], xtaub[1], xtaub[2]);

    return imag(convolutionPi0Tw3);
}

// This function is to be used below in EbarConvolutionPi0. At twist-3, there are 1-dimensional integral terms appearing in the convolution 
double EbarConvolutionPi0Analytic (double x, void * params) {

  double alpha = *(double *) params;
  double convolution = 1. / (x + m_xi) * (1. / sqrt(2.) * (U_ELEC_CHARGE * computeEbarUp(x) - D_ELEC_CHARGE * computeEbarDown(x))) 
			- 1. / (x - m_xi) * (1. / sqrt(2.) * (U_ELEC_CHARGE * computeEbarUp(x) - D_ELEC_CHARGE * computeEbarDown(x)) 				- 1. / sqrt(2.) * (U_ELEC_CHARGE * computeEbarUp(m_xi) - D_ELEC_CHARGE * computeEbarDown(m_xi)));

  return 16. * M_PI * Cf / Nc * alphaS(sqrt(m_Q2 / 2.)) * decayConstant * muPi * pow(transverseSize3, 2.) * convolution;
}


std::complex<double> EbarConvolutionPi0(void) {

    // For pi^0, GPDs appear in the combination of 1/sqrt(2) * (e^u * F^u  - e^d * F^d)

    std::complex<double> convolutionPi0Tw3;

    std::complex<double> convolutionPi0Tw3Analytic = 16. * M_PI * Cf / Nc * alphaS(sqrt(m_Q2 / 2.)) * decayConstant * muPi
	* pow(transverseSize3, 2.) * (1. / sqrt(2.) * (U_ELEC_CHARGE * computeEbarUp(m_xi) - D_ELEC_CHARGE * computeEbarDown(m_xi)) 
	* (1i * M_PI - log((1.-m_xi)/(2.*m_xi)))); // Full analytic terms appearing in the convolution

    double rangeMin[3] = { -m_xi, 0.0, 0.0 };
    double rangeMax[3] = { 1.0, 1.0, 1.0/m_cLambdaQCD };
    double resultEbarRe, errorEbarRe, resultEbarIm, errorEbarIm;

    const size_t nWarmUp = 40000;
    const size_t nCalls = 400000;

    gsl_rng* gslRndEbarRe;
    const gsl_rng_type* gslRndTypeEbarRe;

    gsl_rng* gslRndEbarIm;
    const gsl_rng_type* gslRndTypeEbarIm;

    gsl_rng_env_setup();

    gslRndTypeEbarRe = gsl_rng_default;
    gslRndTypeEbarIm = gsl_rng_default;

    gslRndEbarRe = gsl_rng_alloc(gslRndTypeEbarRe);
    gslRndEbarIm = gsl_rng_alloc(gslRndTypeEbarIm);

    gsl_monte_function gslFunctionEbarRe = {&EbarConvolutionPi0Re, 3, 0};
    gsl_monte_function gslFunctionEbarIm = {&EbarConvolutionPi0Im, 3, 0};

    gsl_monte_vegas_state* gslStateEbarRe = gsl_monte_vegas_alloc(3);
    gsl_monte_vegas_state* gslStateEbarIm = gsl_monte_vegas_alloc(3);

    //Warm-up
    gsl_monte_vegas_integrate(&gslFunctionEbarRe, rangeMin, rangeMax, 3, nWarmUp, gslRndEbarRe, gslStateEbarRe, &resultEbarRe, &errorEbarRe);
    gsl_monte_vegas_integrate(&gslFunctionEbarIm, rangeMin, rangeMax, 3, nWarmUp, gslRndEbarIm, gslStateEbarIm, &resultEbarIm, &errorEbarIm);

    //integrate
    gsl_monte_vegas_integrate(&gslFunctionEbarRe, rangeMin, rangeMax, 3, nCalls, gslRndEbarRe, gslStateEbarRe, &resultEbarRe, &errorEbarRe);
    gsl_monte_vegas_integrate(&gslFunctionEbarIm, rangeMin, rangeMax, 3, nCalls, gslRndEbarIm, gslStateEbarIm, &resultEbarIm, &errorEbarIm);

    //free
    gsl_monte_vegas_free(gslStateEbarRe);
    gsl_monte_vegas_free(gslStateEbarIm);
    gsl_rng_free(gslRndEbarRe);
    gsl_rng_free(gslRndEbarIm);

    std::complex<double> resultEbar = resultEbarRe + 1i * resultEbarIm;

    // 1D integration begins
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double integration1D, error1D;
    double alpha = 1.0;

    gsl_function gslFunctionEbar1D;
    gslFunctionEbar1D.function = &EbarConvolutionPi0Analytic;
    gslFunctionEbar1D.params = &alpha;

    gsl_integration_qags (&gslFunctionEbar1D, -m_xi, 1.0, 0, 1e-5, 10000,
                        w, &integration1D, &error1D);

    gsl_integration_workspace_free (w);
    // 1D integration ends
    
    convolutionPi0Tw3 = convolutionPi0Tw3Analytic + integration1D + resultEbar;

    return convolutionPi0Tw3;

}


// Computation of the amplitude M_{0+0+}
std::complex<double> Amplitude0p0pPi0() {

    std::complex<double> amplitude0p0p = sqrt(1. - pow(m_xi, 2.)) * PositronCharge / sqrt(m_Q2) *
            (HtConvolutionPi0() - pow(m_xi, 2.) / (1. - pow(m_xi, 2.)) * EtConvolutionPi0());

    return amplitude0p0p;

}

// Computation of the amplitude M_{0-0+}
std::complex<double> Amplitude0m0pPi0() {

    std::complex<double> amplitude0m0p = PositronCharge / sqrt(m_Q2) * sqrt(-(m_t - m_tmin)) * m_xi / (2. * PROTON_MASS) *
            EtConvolutionPi0();

    return amplitude0m0p;

}

// Computation of the amplitude M_{0-++}
std::complex<double> Amplitude0mppPi0() {

    std::complex<double> amplitude0mpp = PositronCharge * sqrt(1. - pow(m_xi, 2.)) * HTConvolutionPi0();

    return amplitude0mpp;

}

// Computation of the amplitude M_{0+++}
std::complex<double> Amplitude0pppPi0() {

    std::complex<double> amplitude0ppp = -1.0 * PositronCharge * sqrt(-(m_t - m_tmin)) / (4. * PROTON_MASS) * EbarConvolutionPi0();

    return amplitude0ppp;

}

// Computation of the amplitude M_{0+-+}
std::complex<double> Amplitude0pmpPi0() {

    std::complex<double> amplitude0pmp = -1.0 * PositronCharge * sqrt(-(m_t - m_tmin)) / (4. * PROTON_MASS) * EbarConvolutionPi0();

    return amplitude0pmp;

}

// Computation of the amplitude M_{0--+}
std::complex<double> Amplitude0mmpPi0() {

    std::complex<double> amplitude0mmp = 0.0;

    return amplitude0mmp;

}

// Computation of the \sigma_L
double CrossSectionPi0L(void) {

    double CSL = (pow(abs(Amplitude0p0pPi0()), 2.) + pow(abs(Amplitude0m0pPi0()), 2.)) * Conversion;

    return CSL;
}

// Computation of the \sigma_T
double CrossSectionPi0T(void) {

    double CST = (pow(abs(Amplitude0mppPi0()), 2.) + pow(abs(Amplitude0mmpPi0()), 2.) + pow(abs(Amplitude0pppPi0()), 2.) + pow(abs(Amplitude0pmpPi0()), 2.)) * Conversion / 2.;

    return CST;
}

// Computation of the \sigma_{LT}
double CrossSectionPi0LT(void) {

    double CSLT = -1.0 * sqrt(2.) * real(conj(Amplitude0m0pPi0()) * (Amplitude0mppPi0() - Amplitude0mmpPi0()) + conj(Amplitude0p0pPi0()) * (Amplitude0pppPi0() - Amplitude0pmpPi0())) * Conversion / 2.;

    return CSLT;
}

// Computation of the \sigma_{LT}^\prime
double CrossSectionPi0LTprime(void) {

    double CSLTprime = -1.0 * sqrt(2.) * imag(conj(Amplitude0m0pPi0()) * (Amplitude0mppPi0() - Amplitude0mmpPi0()) + conj(Amplitude0p0pPi0()) * (Amplitude0pppPi0() - Amplitude0pmpPi0())) * Conversion / 2.;

    return CSLTprime;
}

// Computation of the \sigma_{TT}
double CrossSectionPi0TT(void) {
    


    double CSTT = -1.0 * real(conj(Amplitude0mppPi0()) * Amplitude0mmpPi0() + conj(Amplitude0pppPi0()) * Amplitude0pmpPi0()) * Conversion / 2.;

    return CSTT;
}

double Sigma_0(void) {

    double normalization = 1.0 / 2.0 * ( pow(abs(Amplitude0pppPi0()), 2.) + pow(abs(Amplitude0mmpPi0()), 2.) + pow(abs(Amplitude0mppPi0()), 2.) + pow(abs(Amplitude0pmpPi0()), 2.)) + epsilon * ( pow(abs(Amplitude0p0pPi0()), 2.) + pow(abs(Amplitude0m0pPi0()), 2.) );

    return normalization;
}

// Computation of the A_{LU}
double A_LU(void) {

    double ALU = 1.0 / Sigma_0() * sqrt(epsilon * (1.0 - epsilon)) * imag( (conj(Amplitude0pppPi0()) - conj(Amplitude0pmpPi0())) * Amplitude0p0pPi0() + (conj(Amplitude0mppPi0()) - conj(Amplitude0mmpPi0())) * Amplitude0m0pPi0() );

    return ALU;
}


int main (int argc, char** argv){

    //fprintf(stdout, "Q2\txB\tmt\tsigma_T\tsigma_L\tsigma_LT\tsigma_TT\tW\ty\tepsilon\tgammaa\ttmin\n");


       if (argc != 8) {  
        printf("Usage: %s m_xbj_start m_xbj_end m_xbj_space m_Q2_start m_Q2_end m_Q2_space output_file_base_name\n", argv[0]);
        return 1;
    }

    double m_xbj_start = atof(argv[1]);  // Convert string argument to double
    double m_xbj_end = atof(argv[2]);
    double m_xbj_space = atof(argv[3]);

    double m_Q2_start = atof(argv[4]);
    double m_Q2_end = atof(argv[5]);
    double m_Q2_space = atof(argv[6]);

    //create spacing for t
    //double m_t_start = -0.1; in principle should correctly calculate tmin and start there
    double m_t_start = -0.7;
    double m_t_end = -2.0;
    double m_t_space = 0.05;

    const char* output_file_base_name = argv[7];  // Base name of the output file

    char output_file_name[256];
    sprintf(output_file_name, "%s_xbj_%0.2lf_%0.2lf_%0.2lf_Q2_%0.2lf_%0.2lf_%0.2lf_mt_%0.2lf_%0.2lf_%0.2lf.txt",
            output_file_base_name,
            m_xbj_start, m_xbj_end, m_xbj_space,
            m_Q2_start, m_Q2_end, m_Q2_space,
            m_t_start, m_t_end, m_t_space);


    fprintf(stdout, "Q2\txB\tmt\tsigma_T\tsigma_L\tsigma_LT\tsigma_TT\tW\ty\tepsilon\tgammaa\ttmin\n");

    for (double m_xbj = m_xbj_start; m_xbj <= m_xbj_end; m_xbj += m_xbj_space) {

    //for (m_xbj = 0.35; m_xbj < 0.635; m_xbj += 0.2) {
        for (m_Q2 = m_Q2_start; m_Q2 <= m_Q2_end; m_Q2 += m_Q2_space) {
            for (m_t = m_t_start; m_t >= m_t_end; m_t -= m_t_space) {
                
                m_xi = m_xbj / (2.0 - m_xbj);

                gammaa = 2.0 * m_xbj * PROTON_MASS / sqrt(m_Q2);
                y = ( pow(W, 2.0) + m_Q2 - pow(PROTON_MASS, 2.0) ) / ( 2.0 * PROTON_MASS * leptonEnergy );
                epsilon = (1.0 - y - 1.0 / 4.0 * pow(y, 2.0) * pow(gammaa, 2.0)) / (1.0 - y + 1.0 / 2.0 * pow(y, 2.0) + 1.0 / 4.0 * pow(y, 2.0) * pow(gammaa, 2.0));
                W = sqrt(m_Q2 / m_xbj + pow(PROTON_MASS, 2.0) - m_Q2);

                // minimum t value; asymptotic formula
                m_tmin = -4. * pow(PROTON_MASS, 2.) * pow(m_xi, 2.) / (1. - pow(m_xi, 2.));

                Conversion = 0.3894 * pow(10.0, 6.0) / (16.0 * M_PI * (pow(W, 2.0) - pow(PROTON_MASS, 2.0)) * 
                                    sqrt(pow(W, 4.0) + pow(m_Q2, 2.0) + pow(PROTON_MASS, 4.0) + 2.0 * pow(W, 2.0) * m_Q2 
                                        - 2.0 * pow(W, 2.0) * pow(PROTON_MASS, 2.0) + 2.0 * m_Q2 * pow(PROTON_MASS, 2.0)));

                printf("t = %.7lf Q2=%.7lf xB=%.7lf m_xi=%.7lf W=%.7lf leptonEnergy=%.7lf \n", m_t,m_Q2, m_xbj,m_xi,W,leptonEnergy);

                // Open file in append mode
                FILE *f = fopen(output_file_name, "a");

                // If file is opened successfully, write to it.
                if (f != NULL) {
                    fprintf(f, " %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \n", m_Q2, m_xbj, m_t, CrossSectionPi0T(), CrossSectionPi0L(), CrossSectionPi0LT(), CrossSectionPi0TT(),W,y,epsilon,gammaa,m_tmin);
                    fclose(f); // Close the file after writing to it
                }
                else {
                    // Handle error when file is not opened
                    printf("Error opening the file.\n");
                    return 1;
                }
            }
        }
    }

    return 0;
}

// Comment out the rest of the code:

/* The following is old code that doesn't work well for file saving
int main (int argc, char** argv){

    // print the minimum t value
    //printf("t_min = %.7lf\n", m_tmin);
    //printf("x bjorken = %.7lf \n", m_xbj);
    //double W = sqrt(m_Q2 / m_xbj + pow(PROTON_MASS, 2.0) - m_Q2);

    //printf("W = %.7lf \n", W);



    // below, let us calculate the partial cross section of transversely polarized photons at a particular t value
    //m_t= -0.02;
    //printf("t is = %.7lf \n", m_t);
    // printf("Cross section T = %.7lf at t = %.5lf\n", CrossSectionPi0T(), m_t);
    // printf("Cross section L = %.7lf at t = %.5lf\n", CrossSectionPi0L(), m_t);
    // printf("Cross section LT = %.7lf at t = %.5lf\n", CrossSectionPi0LT(), m_t);
    // printf("Cross section TT = %.7lf at t = %.5lf\n", CrossSectionPi0TT(), m_t);

    // write cross section T L LT and TT to a file
    // FILE *f = fopen("cross_section_pi0_10600.txt", "w");
    FILE *f = fopen("cross_section_pi0_10604_july.txt", "w");

    fprintf(f, "Q2\txB\tmt\tsigma_T\tsigma_L\tsigma_LT\tsigma_TT\tW\ty\tepsilon\tgammaa\ttmin\n");

    for (m_xbj = 0.35; m_xbj < 0.635; m_xbj += 0.2) {
        for (m_Q2 = 2; m_Q2 < 10; m_Q2 += 2) {
            for (m_t = -0.2; m_t > -1; m_t -= 0.2) {
                
            //m_Q2 = 2.25; // Q^2 value
            //m_xbj =0.225;// Bjorken x

            // printf("Q2=%.7lf",M_PI);

            m_xi = m_xbj / (2.0 - m_xbj);

            gammaa = 2.0 * m_xbj * PROTON_MASS / sqrt(m_Q2);
            y = ( pow(W, 2.0) + m_Q2 - pow(PROTON_MASS, 2.0) ) / ( 2.0 * PROTON_MASS * leptonEnergy );
            epsilon = (1.0 - y - 1.0 / 4.0 * pow(y, 2.0) * pow(gammaa, 2.0)) / (1.0 - y + 1.0 / 2.0 * pow(y, 2.0) + 1.0 / 4.0 * pow(y, 2.0) * pow(gammaa, 2.0));
            W = sqrt(m_Q2 / m_xbj + pow(PROTON_MASS, 2.0) - m_Q2);

            // minimum t value; asymptotic formula
            m_tmin = -4. * pow(PROTON_MASS, 2.) * pow(m_xi, 2.) / (1. - pow(m_xi, 2.));

            Conversion = 0.3894 * pow(10.0, 6.0) / (16.0 * M_PI * (pow(W, 2.0) - pow(PROTON_MASS, 2.0)) * 
                                sqrt(pow(W, 4.0) + pow(m_Q2, 2.0) + pow(PROTON_MASS, 4.0) + 2.0 * pow(W, 2.0) * m_Q2 
                                    - 2.0 * pow(W, 2.0) * pow(PROTON_MASS, 2.0) + 2.0 * m_Q2 * pow(PROTON_MASS, 2.0)));

            //FILE *f = fopen("cross_section_pi0.txt", "a");
            printf("t = %.7lf Q2=%.7lf xB=%.7lf m_xi=%.7lf W=%.7lf leptonEnergy=%.7lf \n", m_t,m_Q2, m_xbj,m_xi,W,leptonEnergy);

            fprintf(f, " %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \t %.7lf \n", m_Q2, m_xbj, m_t, CrossSectionPi0T(), CrossSectionPi0L(), CrossSectionPi0LT(), CrossSectionPi0TT(),W,y,epsilon,gammaa,m_tmin);
            

            }
        }
    }
    


    // m_t= -0.05;
    // printf("Cross section T = %.7lf at t = %.5lf\n", CrossSectionPi0LT(), m_t);

    // m_t= -0.1;
    // printf("Cross section T = %.7lf at t = %.5lf\n", CrossSectionPi0T(), m_t);


    return 1;
}

*/
