# Part-based Deep Hashing for Large-scale Person Re-identification

## 요약

대규모 사람 재인식에 효율적이고 정확한 평가를 위해 딥러닝과 해싱방법을 결합함.
Deep + Hash function
추가로, Part based Deep Hashing을 제안함.

이전 연구들은 무작위 대입 방식이나 hand-crafted hashing 방식을 채용함.

H: 네트워크 출력을 0과 1로 부호화 시키는 함수
fH: H함수가 미분 불가하므로, sigmoid로 대체함.
H^: 테스팅 단계에선 fH에서 -.5를 하고 부호화하여 사용함.

네트워크의 출력이 0과 1로 부호화된 값이므로, 해밍 거리 함수를 손실함수로 사용해야하지만, 미분 불가하므로 평범하게 fH + L2 Norm사용

결과는 꽤 좋아보임.
특히 Single Query보다 Multiple Query가 생각보다 큰 효과를 냄.
논문에서 제안한 PDH도 생각보다 큰 성능 향상을 줌.
나눠질수록 성능이 향상되는듯.
