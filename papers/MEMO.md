# Questions

## ANN 검색이 가능한 데이터베이스 솔루션이 존재하는가?
존재하지만 거의 사용되지 않는거 같음.
주로 Faiss같은 라이브러리를 사용해 직접 만드는 듯
https://anndb.com/

## Hashing 기법이 쓸모 있는가?
주로 속도 및 메모리 최적화용으로 쓰이는듯

## Access Control System에서 사용하는 매칭 알고리즘 혹은 검색 기법은?

연구 논문에서는 주로 코사인 유사도등 직접 계산하여 사용함.
이스트소프트의 안경 검색 서비스 개발기: https://blog.est.ai/2019/11/%EC%95%88%EA%B2%BD-%EA%B2%80%EC%83%89-%EC%84%9C%EB%B9%84%EC%8A%A4-glasses-finder/

## classification 모델을 그대로 사용할 경우 Visual Search Task에서 어떠한 문제가 있는가?

분류 기반 학습은 inter-class 변동에는 강인하지만, intra-class 변동에는 강인하지 않을 수 있기 때문에 적절하지 않을 수 있음.
즉 key point는 intra-class robustness인데, gan based disentangled representation framework는 intra-class에 robust하기에 사용해도 될듯.

Deep Image Retrieval A Survey:
A classification-based fine-tuning method improves the model-level adaptability for new datasets, which, to some extent, has mitigated the issue of model transfer for image retrieval.
However, there still exists room to improve in terms of classification-based supervised learning. On the one hand, the fine-tuned networks are quite robust to inter-class variability, but may have some difficulties in learning discriminative intra-class variability to distinguish particular objects. On the other hand, class label annotation is time-consuming and labor-intensive for some practical applications. To this end, verification-based fine-tuning methods are combined with classification methods to further improve network capacity.

## ann에서 어떤 단점이 있는지 찾기, 이걸 찾아야 disentangled representation을 사용해도 괜찮은지 알게됨

## annoy와 이전 방식의 차이점 및 장단점

### ann 기법이 기존 방법과 다르게 정렬 되지 않아도 상관 없다는 근거

## 추천시스템의 핵심 알고리즘은 최근접 이웃 검색

## approximate nearest neighbor를 사용하면 완벽한 최단점을 찾지 못할 수 있다. 그럼에도 불구하고 사용하는 이유는 무엇일까?

지문 인식 관련 특허 및 기사에서 현존하는 생체 인식 기술은 정확도가 존재함을 알 수 있다. 특히 특허에서 임계치 T + a, T, T - a를 통해 상중하로 보안등급을 설정한다. 생체인식 시 완벽하게 동일한것을 찾는건 불가능한것으로 생각된다. 따라서 속도 우선의 ANN을 통해 여러차례 질의를 보내고 임계치 이상의 정확도를 보일 때 작동하는 방식이 옳아 보인다.
임계치는 데이터베이스의 양 및 인식 속도를 통해 결정되는 하이퍼파라미터 같다.
이러한 결론의 근거는 특허와 기사이므로, 논문을 좀더 찾아보는게 좋을듯.

특허: https://patentimages.storage.googleapis.com/0d/84/d4/3711e28981b4ec/KR100747446B1.pdf
[표]는 지문인식과 관련한 보안등급이 3개의 레벨(상급, 중급, 하급)로 이루어진 것으로 가정했을 때, 각 보안등급별
기준값을 나타낸 것이다.

기사: https://www.cctvnews.co.kr/news/articleView.html?idxno=209417

# libraries

## spotify annoy

Annoy (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings to search for points in space that are close to a given query point. It also creates large read-only file-based data structures that are mmapped into memory so that many processes may share the same data.
https://github.com/spotify/annoy

## kakao n2

lightweight approximate Nearest Neighbor library which runs fast even with large datasets
https://github.com/kakao/n2

# 읽어볼것

## https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6

정확한 최근접 이웃을 보장하는 방법은 exhaustive search가 유일하다.
the only available method for guaranteed retrieval of the exact nearest neighbor is exhaustive search

This makes exact nearest neighbors impractical even and allows “Approximate Nearest Neighbors “

정확한 최근접 이웃을 찾기위한 방법:

1. exhaustive search: 모두 비교
2. grid trick: 분할정복 we are speaking on high dimension datasets this is impractical.

annoy간단 설명:
In Annoy, in order to construct the index we create a forrest (aka many trees) Each tree is constructed in the following way, we pick two points at random and split the space into two by their hyperplane, we keep splitting in the subspaces recursively until the points associated with a node is small enough.

## Deepfood - Image Retrieval System in Production

시스템 구조도 있음. 참고하면 좋을거같
https://yonigottesman.github.io/2020/05/15/deep-search-aws.html/

# 혼자서 도저히 이해 되자 않는것

## CRF (Conditional Random Field)
