data {
      int < lower =1> N ;               // 参加者数
      int < lower =1> T ;               // 一人の参加者あたりの試行数
      int < lower =1, upper =2> c[N,T]; // 選択データ
      real r[N,T];                      // 報酬
    }
    
parameters {
  // 集団レベル分布のパラメータ
  real mu_p_alpha;
  real<lower=0> sigma_p_alpha;
  real mu_p_beta;
  real<lower=0> sigma_p_beta;
   
  // 参加者ごとの集団レベル分布の平均からの偏差
  real eta_alpha[N];
  real eta_beta[N];
}

transformed parameters {
  real<lower=0.0,upper=1.0> alpha[N]; 
  real<lower=0.0> beta[N];
  
  for (n in 1:N) {
    alpha[n] = inv_logit(mu_p_alpha + sigma_p_alpha * eta_alpha[n]);
    beta[n] = 20 * inv_logit(mu_p_beta + sigma_p_beta * eta_beta[n]); 
  }
}

model {
  matrix [2,T] Q; // 行動価値( 選択肢x 試行)
  
  // 集団レベルのパラメータを事前分布から生成
  mu_p_alpha ~ normal(0 ,1.5);
  sigma_p_alpha ~ uniform(0.0 , 1.5);
  mu_p_beta ~ normal(0 ,1.5);
  sigma_p_beta ~ uniform(0.0 , 1.5); 
  
  // 参加者ごとの偏差を生成
  eta_alpha ~ normal(0 ,1);
  eta_beta ~ normal(0 ,1);
  
  for ( i in 1:N ) {
    // 行動価値の初期値の設定
    Q[1, 1] = 0;
    Q[2, 1] = 0;
    
    for ( t in 1:T ) {
      target += log (1.0/(1.0 + exp(- beta [i] * (Q[c[i,t],t] - Q[3-c[i,t],t ]))));
      
      // 行動価値の更新
      if (t < T) {
        // 選択された選択肢
        Q[c[i,t], t+1] = Q[c[i,t], t] + alpha [i] * (r[i,t] - Q[c[i,t], t]);
        // 選択されなかった選択肢
        Q[3-c[i,t], t +1] = Q[3-c[i,t], t];
      }
    }
  }
}
