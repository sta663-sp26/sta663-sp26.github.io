functions {
  // From https://mc-stan.org/docs/stan-users-guide/gaussian-processes.html#predictive-inference-with-a-gaussian-process
  vector gp_pred_rng(
    array[] real x2,
    vector y1,
    array[] real x1,
    real alpha,
    real rho,
    real sigma,
    real delta
  ) {
    int N1 = rows(y1);
    int N2 = size(x2);
    vector[N2] f2;
    {
      matrix[N1, N1] L_K;
      vector[N1] K_div_y1;
      matrix[N1, N2] k_x1_x2;
      matrix[N1, N2] v_pred;
      vector[N2] f2_mu;
      matrix[N2, N2] cov_f2;
      matrix[N2, N2] diag_delta;
      matrix[N1, N1] K;
      K = gp_exp_quad_cov(x1, alpha, rho);
      for (n in 1:N1) {
        K[n, n] = K[n, n] + square(sigma);
      }
      L_K = cholesky_decompose(K);
      K_div_y1 = mdivide_left_tri_low(L_K, y1);
      K_div_y1 = mdivide_right_tri_low(K_div_y1', L_K)';
      k_x1_x2 = gp_exp_quad_cov(x1, x2, alpha, rho);
      f2_mu = (k_x1_x2' * K_div_y1);
      v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
      cov_f2 = gp_exp_quad_cov(x2, alpha, rho) - v_pred' * v_pred;
      diag_delta = diag_matrix(rep_vector(delta, N2));

      f2 = multi_normal_rng(f2_mu, cov_f2 + diag_delta);
    }
    return f2;
  }
}
data {
  int<lower=1> N;      // number of observations
  array[N] real x;         // univariate covariate
  vector[N] y;         // target variable
  int<lower=1> Np;     // number of test points
  array[Np] real xp;       // univariate test points
}
transformed data {
  real delta = 1e-9;
}
parameters {
  real<lower=0> l;
  real<lower=0> s;
  real<lower=0> nug;
}
model {
  // Covariance
  matrix[N, N] K = gp_exp_quad_cov(x, s, l);
  K = add_diag(K, nug^2);
  matrix[N, N] L = cholesky_decompose(K);
  
  // priors
  l ~ gamma(2, 1);
  s ~ cauchy(0, 5);
  nug ~ cauchy(0, 1);
  
  // model
  y ~ multi_normal_cholesky(rep_vector(0, N), L);
}
generated quantities {
  vector[Np] f = gp_pred_rng(xp, y, x, s, l, nug, delta);
}
