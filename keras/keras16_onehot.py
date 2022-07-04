#[과제]
#3가지 onehotencoding 방식을 비교할것

#1.pandas의 get_dummies - argmax 부분은 numpy말고 tensorflow로 하면 돌아간다. 라벨인코딩은 필요가없다.



#2.tensorflow의 to_categorical - 컬럼 구성을 무조건 0부터 시작하게됨. 0이없으면 생성하게됨



#3.sklearn의 OneHotEncoder - reshape 와 sparse= False 필수 (넘파이 배열로 반환)
#  true로 하면 희소행렬( 대부분 0으로 구성된 행렬과 계산이나 메모리 효율을 이용해 0이 아닌 값의 index만 관리)로 반환.
