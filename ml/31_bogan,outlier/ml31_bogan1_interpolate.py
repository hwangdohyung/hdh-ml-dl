# 결측치 처리 
#1. 행 또는 열 삭제 
#2. 임의의 값(평균,중위,0,앞값,뒷값) 

# 앞값: ffill
# 뒷값: bfill
# 평균: mean
# 중위: median
# 0  : fillna
# 특정값: ...

#3. 보간 - interpolate 
# 빈자리를 linear 방식으로 찾아낸다.
#4. 모델 - predict 에 넣어서 결과값을 빈자리에 채운다.
#5. 부스팅계열 =통상 결측치,이상치에 대해 자유롭다. 

import pandas as pd
import numpy as np 
from datetime import datetime 

dates = ['8/10/2022', '8/11/2022', '8/12/2022', '8/13/2022', '8/14/2022',]

dates = pd.to_datetime(dates)
print(dates)

print('===========================')
ts = pd.Series([2, np.nan, np.nan, 8, 10], index=dates)
print(ts)

print('===========================')
ts = ts.interpolate()
print(ts)