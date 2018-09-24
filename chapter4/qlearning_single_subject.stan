data {
      int<lower=1> T ;               // 試行数
      int<lower=1,upper=2> c[T];     // 選択
      real r[T];                     // 報酬
    }
    
parameters {
   real<lower=0.0,upper=1.0> alpha;  // 学習率
   real<lower=0.0> beta;             // 逆温度
}

model {
  matrix[2,T] Q; // 行動価値 (選択肢の数x 試行数)
  
  // 事前分布からのパラメータのサンプリング
  alpha ~ beta(2, 2);                // 学習率はベータ分布から生成
  beta ~ gamma(2, 0.333);            // 逆温度はガンマ分布から生成

  // 初期値の設定 (実際は定義した際にゼロになるのでなくてもよい)
  Q[1, 1] = 0;
  Q[2, 1] = 0;
  
  for ( t in 1:T ) {
    
    // 試行t の対数尤度を足し合わせる
    target += log( 1.0 / (1.0 + exp(-beta * (Q[c[t],t] - Q[3-c[t],t]))) ); 
    
    // 行動価値の更新
    if (t < T) {
      // 選択された選択肢
      Q[c[t], t+1] = Q[c[t], t] + alpha * (r[t] - Q[c[t], t]);
      // 選択されなかった選択肢
      Q[3-c[t], t+1] = Q[3-c[t], t];
    }
  }
}

