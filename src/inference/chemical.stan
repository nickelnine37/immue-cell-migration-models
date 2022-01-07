// define a set of custom functions

functions{

    // exponential integral function. Based on continued fractional expansion  approximation
    // see https://doi.org/10.1016/S0022-1694(97)00134-0
    real expi(real x){
        return - exp(x) / (1 - x - (1 / (3 - x - (2 ^ 2 / (5 - x - (3 ^ 2 / (7 - x - (4 ^ 2) / (5 - x))))))));
    }

    real expi2(real x){

        real xn = -x;
        real Sn = -x;
        real Sm1 = 0.0;
        real hsum = 1.0;
        real g = 0.5772156649015328606065121;
        real y = 1.0;
        real factorial = 1.0;
        real epsilon = 0.001;

        if (x == 0.0) {return -1e10;}

        while (fabs(Sn - Sm1) > epsilon * fabs(Sm1)){

            Sm1 = Sn;
            y += 1.0;
            xn *= (-x);
            factorial *= y;
            hsum += (1.0 / y);
            Sn += hsum * xn / factorial;

        }

        return g + log(fabs(x)) - exp(x) * Sn;

    }

    // concentration function, at vector positions (r, t) given
    // given parameters q, D and tau
    vector concentration(real q, real D, real tau, vector r, vector t){

        int N = num_elements(r);
        vector[N] out;

        for (n in 1:30){
            if (t[n] < tau)
                out[n] = -expi2(- r[n] ^ 2 / (4 * D * t[n]));
            else
                out[n] = expi2(- r[n] ^ 2 / (4 * D * (t[n] - tau))) - expi2(- r[n] ^ 2 / (4 * D * t[n]));
        }
        return (q / (4 * 3.14159 * D)) * out;
    }

    // the number of bound complexes at vector positions (r, t) given
    // q, D, tau, R_0, kappa
    vector complexes(real q, real D, real tau, real R_0, real kappa,  vector r, vector t){

        int N = num_elements(r);
        vector[N] A = concentration(q, D, tau, r, t);
        vector[N] B = (kappa + R_0 + A);
        vector[N] k = 0.25 *  B .* B - R_0 * A;
        return 0.5 * B - sqrt(k);
    }

    // the observed bias that should be observed at a point (r, t) given a vector of
    // parameters, q, D, tau, R_0, kappa, m, b
    vector bias(vector params,  vector r, vector t){
        return params[6] * (complexes(params[1], params[2], params[3], params[4], params[5], r - 15, t) - complexes(params[1], params[2], params[3], params[4], params[5], r + 15, t)) + params[7];
    }

}

// the input data for inference
data {
    int n_readings;                      // the number of OB readings taken
    vector[n_readings] r;                // a vector speciying the radii of the readings
    vector[n_readings] t;                // a vector specifying the time of the readings
    vector[n_readings] obs_bias_mean;    // a vector specifying the observed bias mean at these points
    vector[n_readings] obs_bias_sd;      // a vector specifying the observed bias σ at these points
    vector[7] params_mean;               // a vector specifying the mean of the normal prior over the parameters
    vector[7] params_sd;                 // a vector specifying σ for the normal prior over the parameters
}

// the parameters for which we want to find the posterior over
parameters {
    vector<lower=0>[7] params;
}

// model setup
model {
    // parameters prior is independant MVN
    params ~ normal(params_mean, params_sd);

    // observed bias is independant MVN with mean bias(params, r, t)
    obs_bias_mean ~ normal(bias(params, r, t), obs_bias_sd);
}