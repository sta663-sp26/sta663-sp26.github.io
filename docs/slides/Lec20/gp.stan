data {
  int<lower=1> N;
  array[N] real x;
  vector[N] y;
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
