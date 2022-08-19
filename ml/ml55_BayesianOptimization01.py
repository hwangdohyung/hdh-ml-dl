param_bounds = {'x1': (-1, 5),
                'x2': (0,4)}
def eval_function(x1, x2):
    return -x1 **2 -(x2 -2) **2 +10
# 첫번째 항은 최대 , 두번째 항은 최소가 될때 (y)가장 큰값을 반환함. 

print(eval_function(0,2))


