// Standard Q-learning model for multiple group

data {
      int<lower=1> N ; // number of subjects (or sessions)
      int<lower=1> G[N] ; // group id 
      int<lower=1> nGroup ; // number of groups 
      int<lower=0> flgCommon_alpha ; 
      // 1: common 0:different (for population dist. of alpha)
      int<lower=0> flgCommon_beta ; 
      // 1: common 0:different (for population dist. of beta)
      int<lower=1> T ; //  total number of trials (over subjects)
      int<lower=1,upper=2> c[N,T]; // choice
      real r[N,T]; // reward
      int WBICmode; // 0:normal Bayes, 1:sampling for WBIC 
    }
    
parameters {
   real mu_p_alpha[nGroup];
   real<lower=0> sigma_p_alpha[nGroup];
   real mu_p_beta[nGroup];
   real<lower=0> sigma_p_beta[nGroup];
   
   real eta_alpha[N];
   real eta_beta[N];
}

transformed parameters {
  real<lower=0.0,upper=1.0> alpha[N]; 
  real<lower=0.0> beta[N];
  real<lower=0.0,upper=1.0> alpha_p[nGroup]; 
  real<lower=0.0> beta_p[nGroup];
  
  for (n in 1:N) {
    if (flgCommon_alpha) {
      alpha[n] = inv_logit(mu_p_alpha[1] + sigma_p_alpha[1] * eta_alpha[n]);
    } else {
      alpha[n] = inv_logit(mu_p_alpha[G[n]] + sigma_p_alpha[G[n]] * eta_alpha[n]);
    }
    if (flgCommon_beta) {
      beta[n] = 20 * inv_logit(mu_p_beta[1] + sigma_p_beta[1] * eta_beta[n]); 
    } else {
      beta[n] = 20 * inv_logit(mu_p_beta[G[n]] + sigma_p_beta[G[n]] * eta_beta[n]); 
    }
  }
  for (idxG in 1:nGroup) {
    alpha_p[idxG] = inv_logit(mu_p_alpha[idxG]);
    beta_p[idxG] = 20*inv_logit(mu_p_beta[idxG]);
  }
}

model {
  matrix[2,T] Q; // Q values (option x trial)
  vector[2] tmp;
  
  // population distribution
  for (idxG in 1:nGroup) {
      mu_p_alpha[idxG] ~ normal(0,1.5);
      sigma_p_alpha[idxG] ~ uniform(0.0, 1.5);
      mu_p_beta[idxG] ~ normal(0,1.5);
      sigma_p_beta[idxG] ~ uniform(0.0, 1.5); 
  }
  
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
  
  real alpha_diff;
  real beta_diff;

  alpha_diff = alpha_p[2] - alpha_p[1];
  beta_diff = beta_p[2] - beta_p[1];

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
