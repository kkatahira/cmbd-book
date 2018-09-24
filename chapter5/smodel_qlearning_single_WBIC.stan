data {
      int<lower=1> T ;
      int<lower=1,upper=2> c[T]; // choice
      real r[T]; // reward
    }
    
parameters {
   real<lower=0.0,upper=1.0> alpha;
   real<lower=0.0> beta;
}

model {
  matrix[2,T] Q; // Q values (option x trial)
  vector[2] tmp;
  
  alpha ~ beta(2, 2); // vectorized, truncated [0,1]
  beta ~ gamma(2, 0.333); // vectorized, truncated (> 0)

  // initial value set
  Q[1, 1] = 0;
  Q[2, 1] = 0;
  for ( t in 1:T ) {
    
     target += 
       1/log(T) * // inverse temperature for WBIC 1/log(number of samples)
       log( 1.0/(1.0 + exp(-beta * (Q[c[t],t] - Q[3-c[t],t]))) );
    
    // update action value
    if (t < T) {
      // chosen action
      Q[c[t], t+1] = Q[c[t], t] + alpha * (r[t] - Q[c[t], t]);
      // unchosen action
      Q[3-c[t], t+1] = Q[3-c[t], t];
    }
  }
}



generated quantities {
  
  vector[T] log_lik;
  
  {
    matrix[2,T] Q; // Q values (option x trial)
    vector[2] tmp;
    int idxsample;
    
    idxsample = 0;
  
    // initial value set
    Q[1, 1] = 0;
    Q[2, 1] = 0;
    for ( t in 1:T ) {
      idxsample = idxsample + 1;
       log_lik[idxsample] =  
         log( 1.0/(1.0 + exp(-beta * (Q[c[t],t] - Q[3-c[t],t]))) );
         
      // update action value
      if (t < T) {
        // chosen action
        Q[c[t], t+1] = Q[c[t], t] + alpha * (r[t] - Q[c[t], t]);
        // unchosen action
        Q[3-c[t], t+1] = Q[3-c[t], t];
      }
    }
  }
}

