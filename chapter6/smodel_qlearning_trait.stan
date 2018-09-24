// Standard Q-learning model with fixed effect of trait on learning rate

data {
      int<lower=1> N ; // number of subjects (or sessions)
      int<lower=1> T ; // number of trials (per subject)
      int<lower=1,upper=2> c[N,T]; // choice
      real r[N,T]; // reward
      real trait[N]; // traint
      int<lower=0> flg_trait_alpha; 
      int WBICmode; // 0:bayes, 1:sampling for WBIC 
    }
    
parameters {
   real mu_p_alpha;
   real<lower=0> sigma_p_alpha;
   real mu_p_beta;
   real<lower=0> sigma_p_beta;
   real b1; # regression coefficient (trait to learng rate)
   
   real eta_alpha[N];
   real eta_beta[N];
}

transformed parameters {
  real<lower=0.0,upper=1.0> alpha[N]; 
  real<lower=0.0> beta[N];
  real<lower=0.0,upper=1.0> alpha_p; 
  real<lower=0.0> beta_p;
  
  for (n in 1:N) {
    alpha[n] = inv_logit(mu_p_alpha + flg_trait_alpha * b1 * trait[n] + sigma_p_alpha * eta_alpha[n]);
    beta[n] = 20 * inv_logit(mu_p_beta + sigma_p_beta * eta_beta[n]); 
  }
  
  alpha_p = inv_logit(mu_p_alpha);
  beta_p = 20*inv_logit(mu_p_beta);
  
}

model {
  matrix[2,T] Q; // Q values (option x trial)
  vector[2] tmp;
  
  // population distribution
  mu_p_alpha ~ normal(0,1.5);
  sigma_p_alpha ~ uniform(0.0, 1.5);
  mu_p_beta ~ normal(0,1.5);
  sigma_p_beta ~ uniform(0.0, 1.5); 
  b1 ~ normal(0,1);

  eta_alpha ~ normal(0,1);
  eta_beta ~ normal(0,1); 
 
  for ( i in 1:N ) {
    // initial value set
    Q[1, 1] = 0;
    Q[2, 1] = 0;
    for ( t in 1:T ) {
      
      if (WBICmode) {
        target +=
        1/log(N*T) * // inverse temperature for WBIC 1/log(number of samples)
         log( 
           1.0/(1.0 + exp(-beta[i] * (Q[c[i,t],t] - Q[3-c[i,t],t]))) 
         );
      } else {
       target +=
         log( 
           1.0/(1.0 + exp(-beta[i] * (Q[c[i,t],t] - Q[3-c[i,t],t]))) 
         );
      }
      
      // update action value
      if (t < T) {
        // chosen action
        Q[c[i,t], t+1] = Q[c[i,t], t] + alpha[i] * (r[i,t] - Q[c[i,t], t]);
        // unchosen action
        Q[3-c[i,t], t+1] = Q[3-c[i,t], t];
      }
    }
  }
}


generated quantities {

  vector[N*T] log_lik;

  {
    matrix[2,T] Q; // Q values (option x trial)
    vector[2] tmp;
    int trial_count;

    trial_count = 0;

    for ( i in 1:N ) {
      // initial value set
      Q[1, 1] = 0;
      Q[2, 1] = 0;
      for ( t in 1:T ) {
        trial_count = trial_count + 1;

         log_lik[trial_count] =
           log( 1.0/(1.0 + exp(-beta[i] * (Q[c[i,t],t] - Q[3-c[i,t],t]))) );

        // update action value
        if (t < T) {
          // chosen action
          Q[c[i,t], t+1] = Q[c[i,t], t] + alpha[i] * (r[i,t] - Q[c[i,t], t]);
          // unchosen action
          Q[3-c[i,t], t+1] = Q[3-c[i,t], t];
        }
      }
    }
  }
}
