param_bounds = {'x1': (-1, 5),
                'x2': (0,4)}
def y_function(x1, x2):
    return -x1 **2 -(x2 -2) **2 +10
# 첫번째 항은 최대 , 두번째 항은 최소가 될때 (y)가장 큰값을 반환함. 
# 절대값으로 가장 작은값, 큰값이 된다 (x1,x2) 

print(y_function(0,2))

# pip install bayesian-optimization
from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(f=y_function,pbounds=param_bounds,random_state=1234) 

optimizer.maximize(init_points=2, n_iter=20)  #초기 2번  ,n-iter : 20번 돌거다! 총 22번 돈다 

print(optimizer.max) #{'target': 9.999835918969607, 'params': {'x1': 0.00783279093916099, 'x2': 1.9898644972252864}}

# bayesian 은 알고리즘을 통해 파라미터를 찾음. grid서치는 모델을 돌려서 찾음