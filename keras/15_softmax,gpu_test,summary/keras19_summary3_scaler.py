#전처리 과정 * x는 바뀔수 있으나 y는 바뀌면 안된다.

#  	종류	설명
# 1	StandardScaler	기본 스케일. 평균과 표준편차 사용
# 2	MinMaxScaler	최대/최소값이 각각 1, 0이 되도록 스케일링
# 3	MaxAbsScaler	최대절대값과 0이 각각 1, 0이 되도록 스케일링
# 4	RobustScaler	중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화

# 1. StandardScaler
# 평균을 제거하고 데이터를 단위 분산으로 조정한다. 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 된다.
# 따라서 이상치가 있는 경우 균형 잡힌 척도를 보장할 수 없다. x를 평균으로 빼고 표준편차로 나눈다. - 평균이 0이되게끔 함.
# from sklearn.preprocessing import StandardScaler
# standardScaler = StandardScaler()
# print(standardScaler.fit(train_data))
# train_data_standardScaled = standardScaler.transform(train_data)

# 2. MinMaxScaler
# 모든 feature 값이 0~1사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축될 수 있다.
# 즉, MinMaxScaler 역시 아웃라이어의 존재에 매우 민감하다. 
# from sklearn.preprocessing import MinMaxScaler
# minMaxScaler = MinMaxScaler()
# print(minMaxScaler.fit(train_data))
# train_data_minMaxScaled = minMaxScaler.transform(train_data)

# 3. MaxAbsScaler
# 절대값이 0~1사이에 매핑되도록 한다. 즉 -1~1 사이로 재조정한다. 양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.
# from sklearn.preprocessing import MaxAbsScaler
# maxAbsScaler = MaxAbsScaler()
# print(maxAbsScaler.fit(train_data))
# train_data_maxAbsScaled = maxAbsScaler.transform(train_data)

# 4. RobustScaler
# 아웃라이어의 영향을 최소화한 기법이다. 중앙값(median)과 IQR(interquartile range)을 사용하기 때문에 StandardScaler와 비교해보면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있다.
# from sklearn.preprocessing import RobustScaler
# robustScaler = RobustScaler()
# print(robustScaler.fit(train_data))
# train_data_robustScaled = robustScaler.transform(train_data)
# 결론적으로 모든 스케일러 처리 전에는 아웃라이어 제거가 선행되어야 한다. 또한 데이터의 분포 특징에 따라 적절한 스케일러를 적용해주는 것이 좋다.
# # IQR = Q3 - Q1 : 즉, 25퍼센타일과 75퍼센타일의 값들을 다룬다.
