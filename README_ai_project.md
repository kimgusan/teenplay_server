# <img src="https://github.com/team-teenplay/teenplay_server/assets/156397974/10dfb0a8-62c1-412f-bc78-fd415191e84c" width="25"> 틴플레이 영상 재생 알고리즘 AI 프로젝트 보고서 🎉

틴플레이: http://teenplay.store

## ♠️ 목차

1. **AI 서비스 설명**
2. **AI 사전 훈련 모델**

    - **필요 데이터 선택 (데이터 수집)**
    - **데이터 전처리**
    - **훈련 모델 선택**
    - **훈련 모델 검증**
      
4. **AI 사용 모델**
5. **AI 훈련 모델 화면**
6. **평가**
7. **트러블슈팅**
8. **느낀점**

<hr>

### 1. AI 서비스 설명 ❤️

-   사용자가 관심있는 분야, 영상 '좋아요'를 가장 많이 누른 관심 케이스, 위시리스트에 대한 회원 관심사에 대하여 영상을 보여주는 AI 추천 서비스
  
<hr>

### 2. AI 사전 훈련 모델 💻

#### 가. 데이터 수집 📊
    
**-Features-**
    
    1. u_info_category
    2. u_wishlist_category
    3. u_favorite_teenplay_like_club_category
    
**-Target-**

    1. club_category
    
##### 1) 클럽 리스트에 대하여 전체 count 확인.

<details>
  <summary>Click all Code</summary>

    import pandas as pd
    import numpy as np

    c_df = pd.read_csv('./club_list_2.csv')
    c_df

    <!-- # value_counts 결과를 DataFrame으로 변환 -->
    counts = c_df[['club_main_category_id', 'category_name']].value_counts().reset_index(name='counts')

    <!-- # club_main_category_id를 기준으로 정렬 -->
    sorted_counts = counts.sort_values(by='club_main_category_id')

    <!-- # 결과 출력 -->
    sorted_counts

</details>
<img width="294" alt="스크린샷 2024-05-22 오후 8 55 14" src="https://github.com/kimgusan/teenplay_server/assets/156397911/31a8f2bd-9a54-4f83-bd41-14f0b4f7ec87">

##### 2) 크롤링을 통해 가져온 각 카테고리별 title 데이터 프레임 생성.

<details>
  <summary>Click all Code</summary>
    
    <!-- # 랜덤 titile 확인 -->

    import pandas as pd
    import numpy as np

    <!-- # CSV 파일 목록 -->
    file_list = ['./youtube_another.csv', './youtube_culture.csv', './youtube_festival.csv',
                './youtube_food.csv', './youtube_hobby.csv', './youtube_love.csv',
                './youtube_sport.csv', './youtube_stock.csv', './youtube_travel.csv']

    <!-- # 각 CSV 파일을 읽어서 데이터프레임 리스트로 저장 -->
    df_list = [pd.read_csv(file) for file in file_list]

    <!-- # 모든 데이터프레임을 세로로 결합 -->
    all_df = pd.concat(df_list, axis=0, ignore_index=True)

    <!-- # 결과 출력 -->
    all_df

</details>

<img width="841" alt="스크린샷 2024-05-22 오후 8 56 14" src="https://github.com/kimgusan/teenplay_server/assets/156397911/25c72b34-f56e-47e4-b81f-accd0ad97da8">

<br>
<br>

#### 나. 데이터 전처리 🧹

##### 1) 가져온 데이터에 대하여 카테고리별로 데이터 분류

<details>
  <summary>Click all Code</summary>

    def assign_value_based_on_column(value):
        if value =='취미':
            return 1
        elif value == "문화·예술":
            return 2
        elif value == "운동·액티비티":
            return 3
        elif value == "푸드·드링크":
            return 4
        elif value == "여행·동행":
            return 5
        elif value == "성장·자기개발":
            return 6
        elif value == "동네·또래":
            return 7
        elif value == "연애·소개팅":
            return 8
        elif value == "재테크":
            return 9
        elif value == "외국어":
            return 10
        elif value == "스터디":
            return 11
        elif value == "지역축제":
            return 12
        else:
            return 13

    pre_t_df['club_main_category_id'] = pre_t_df['club_main_category_id'].apply(assign_value_based_on_column)
    pre_t_df

</details>

<img width="177" alt="스크린샷 2024-05-22 오후 8 56 46" src="https://github.com/kimgusan/teenplay_server/assets/156397911/4f42a089-8e53-4702-bc1f-cfab31eacb81">


##### 2) 영상 데이터에 대해 사용자가 좋아하는 카테고리, 위시리스트 카테고리 일정 비율로 값 부여

<details>
  <summary>Click all Code</summary>

    <!-- # 3:7 비율로 숫자를 맞추기 위한 함수 -->
    <!-- user category의 경우 좋아요를 누른 카테고리 이기 때문에 3:7 비율로 카테고리 값을 넣어준다.-->
    def match_categories(df, like_col, target_col, ratio=(3, 7)):
        like_values = df[like_col].value_counts().index
        for value in like_values:
            like_indices = df[df[like_col] == value].index
            num_target = int(len(like_indices) * ratio[1] / sum(ratio))
            target_indices = np.random.choice(like_indices, size=num_target, replace=False)
            df.loc[target_indices, target_col] = value
        return df

    <!-- # club_category(target) 열 생성 -->
    pre_t_df['user_category'] = np.nan

    <!-- # user_like_category와 club_category(target) 값을 3:7 비율로 맞춤 -->
    pre_t_df = match_categories(pre_t_df, 'club_main_category_id', 'user_category')

    <!-- # 결측치 채우기: 남아 있는 결측치에 대해 임의의 값으로 채움 (여기서는 같은 비율로 채우기) -->
    remaining_indices = pre_t_df['user_category'].isna()
    num_remaining = remaining_indices.sum()
    if num_remaining > 0:
        remaining_values = np.random.choice(pre_t_df['club_main_category_id'].unique(), size=num_remaining, replace=True)
        pre_t_df.loc[remaining_indices, 'user_category'] = remaining_values


    <!-- # 결과 출력 -->
    pre_t_df['user_category'] = pre_t_df['user_category'].astype(np.int64)
    pre_t_df

</details>

<img width="523" alt="스크린샷 2024-05-22 오후 8 58 18" src="https://github.com/kimgusan/teenplay_server/assets/156397911/2341417a-4b6e-4932-a1e4-0276f2e9f143">

#### 다. 훈련 모델 선택 🤖

##### 1) 결정트리 모델 (교차검증) 훈련

<details>
  <summary>Click all Code</summary>
  
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.tree import DecisionTreeClassifier

    features, targets = pre_t_df.iloc[:, :-1], pre_t_df.iloc[:, -1]

    X_train, X_test, y_train, y_test = \
    train_test_split(features.values, targets.values, stratify=targets, test_size= 0.2, random_state=321)

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)

    <!-- # 교차 검증 진행 GridSearchCV -->
    from sklearn.model_selection import GridSearchCV

    parameters = {
        'max_depth': [5, 7, 9 ,10],
        'min_samples_split': [10, 20 ,30]
    }

    g_dtc = GridSearchCV(dtc, param_grid=parameters, cv=3, refit=True, return_train_score=True)
    g_dtc.fit(X_train, y_train)

</details>

<img width="221" alt="스크린샷 2024-05-22 오후 8 59 54" src="https://github.com/kimgusan/teenplay_server/assets/156397911/d1b20549-ddcb-4846-b5da-0d965b24855c">
<img width="272" alt="스크린샷 2024-05-22 오후 8 59 59" src="https://github.com/kimgusan/teenplay_server/assets/156397911/39b3b415-128a-41eb-8c52-20682585de59">


##### 평가지표 확인

<details>
    <summary>Click all Code</summary>

    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

    def get_evaluation(y_test, prediction, classifier=None, X_test=None):
        confusion = confusion_matrix(y_test, prediction)
        accuracy = accuracy_score(y_test , prediction)
        precision = precision_score(y_test , prediction, average='micro')
        recall = recall_score(y_test , prediction, average='micro')
        f1 = f1_score(y_test, prediction, average='micro')
        # auc = roc_auc_score(y_test, prediction)

        print('오차 행렬')
        print(confusion)
        print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}'.format(accuracy, precision, recall, f1))
        print("#" * 80)

        if classifier is not None and  X_test is not None:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
            titles_options = [("Confusion matrix", None), ("Normalized confusion matrix", "true")]

            for (title, normalize), ax in zip(titles_options, axes.flatten()):
                disp = ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, ax=ax, cmap=plt.cm.Blues, normalize=normalize)
                disp.ax_.set_title(title)
            plt.show()

</details>

<img width="668" alt="스크린샷 2024-05-22 오후 8 59 17" src="https://github.com/kimgusan/teenplay_server/assets/156397911/df61e7ef-08c1-4833-8033-911f91b20539">


##### 2) KNN최근접 이웃 모델 (교차검증) 훈련

<details>
  <summary>Click all Code</summary>
    from sklearn.neighbors import KNeighborsClassifier

    knn_c = KNeighborsClassifier()

    parameters = {
        'n_neighbors' : [3, 5, 7, 9, 11]
    }

    g_knn_c = GridSearchCV(knn_c, param_grid=parameters, cv=3, refit=True, return_train_score=True)
    g_knn_c.fit(X_train, y_train)

</details>

<img width="205" alt="스크린샷 2024-05-22 오후 9 00 36" src="https://github.com/kimgusan/teenplay_server/assets/156397911/bba74580-5609-4f2f-bd03-f8d6f387d515">

##### 평가지표 확인

<details>
  <summary>Click all Code</summary>

    knn_c = g_knn_c.best_estimator_
    prediction = knn_c.predict(X_test)
    get_evaluation(y_test, prediction, knn_c, X_test)

</details>

<img width="680" alt="스크린샷 2024-05-22 오후 9 00 52" src="https://github.com/kimgusan/teenplay_server/assets/156397911/fbc8dda0-03e7-40bf-b122-98950f0e92a9">


##### 3) AdaboostClassifier 모델 (교차검증) 사용

<details>
  <summary>Click all Code</summary>

    <!-- # 데이터의 훈련 시 시간이 얼마나 걸릴 지 모르기 때문에 우선 1부터 조금씩 낮아지도록 훈련 진행 -->
    from sklearn.ensemble import AdaBoostClassifier

    parameters = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.7, 0.8, 0.9, 1]
    }

    g_ada = GridSearchCV(AdaBoostClassifier(), param_grid= parameters, cv=3, refit=True, return_train_score=True, n_jobs = -1)
    g_ada.fit(X_train, y_train)

</details>

<img width="194" alt="스크린샷 2024-05-22 오후 9 03 27" src="https://github.com/kimgusan/teenplay_server/assets/156397911/bbd09c51-2827-4ab4-9fa9-6fce0a6f49b0">

##### 평가지표 확인

<details>
  <summary>Click all Code</summary>

    ada = g_ada.best_estimator_
    prediction = ada.predict(X_test)
    get_evaluation(y_test, prediction, ada, X_test)

</details>

<img width="673" alt="스크린샷 2024-05-22 오후 9 03 37" src="https://github.com/kimgusan/teenplay_server/assets/156397911/7405403e-1798-4189-aabc-685f5ae80350">

##### 4) Randomforest 모델 (교차검증) 사용

<details>
  <summary>Click all Code</summary>

    from sklearn.ensemble import RandomForestClassifier
    
    parameters = {
        'max_depth' : [8, 9, 10],
        'min_samples_split' : [50, 100, 150]
    }
    
    rfc = RandomForestClassifier(n_estimators=70)
    
    g_rfc = GridSearchCV(rfc, param_grid=parameters, cv=3, n_jobs= -1)
    g_rfc.fit(X_trian, y_train)

</details>

<img width="212" alt="스크린샷 2024-05-22 오후 9 09 12" src="https://github.com/kimgusan/teenplay_server/assets/156397911/af93b6f3-767e-4e06-a241-6c2970773128">

##### 평가지표 확인

<details>
  <summary>Click all Code</summary>

    rfc = g_rfc.best_estimator_
    prediction = rfc.predict(X_test)
    get_evaluation(y_test, prediction, rfc, X_test)

</details>

<img width="675" alt="스크린샷 2024-05-22 오후 9 31 07" src="https://github.com/kimgusan/teenplay_server/assets/156397911/b507ace1-d68d-4dd6-be2b-da3c1422b992">

#### 라. 모델 검증

##### 1) 검증 데이터 분류 후 Randomforest 모델 훈련

<img width="224" alt="스크린샷 2024-05-22 오후 9 10 48" src="https://github.com/kimgusan/teenplay_server/assets/156397911/ba3b5fc0-e5b5-492f-8794-f7b505b780e7">

##### 2) 훈련 데이터와 평가 지표 비교하여 분석

<img width="672" alt="스크린샷 2024-05-22 오후 9 11 15" src="https://github.com/kimgusan/teenplay_server/assets/156397911/158bdc65-c394-44c5-a493-e4b11783d688">
<img width="671" alt="스크린샷 2024-05-22 오후 9 11 23" src="https://github.com/kimgusan/teenplay_server/assets/156397911/e8e27231-e3e4-4de0-8d0f-08ea382abd9e">


###### 3) 검증된 모델에 대하여 predict_proba 를 이용한 target의 확률 확인

<details>
  <summary>Click all Code</summary>

    proba = rfc.predict_proba(X_test)
    prediction = rfc.predict(X_test)


    class_labels = rfc.classes_

    <!-- # proba 및 predict 출력 -->
    for proba, predict in zip(proba.tolist(), prediction.tolist()):
        print("Predicted class:", predict)
        print("Class probabilities:", {label: p for label, p in zip(class_labels, proba)})
        print()  # 빈 줄 추가

</details>

<img width="739" alt="스크린샷 2024-05-22 오후 9 12 02" src="https://github.com/kimgusan/teenplay_server/assets/156397911/d287197c-7bc8-4b03-bdf0-a0a6143e3a08">

##### 4) 사용자에게 입력된 값에 대하여 결과 모델 확인

<details>
  <summary>Click all Code</summary>

    import pandas as pd
    import numpy as np

    <!-- # RFC 모델로부터 최적의 추정기 가져오기 -->
    rfc = g_rfc.best_estimator_

    <!-- # 사용자가 입력한 피처 값 -->
    user_input_feature = [[3, 2, 1]]

    <!-- # 예측 확률 얻기 -->
    prediction_proba = rfc.predict_proba(user_input_feature)

    <!-- # 상위 1개의 확률에 해당하는 클래스 추출 -->
    top_3_indices = np.argsort(prediction_proba[0])[-1:][::-1]
    top_3_classes = rfc.classes_[top_3_indices]

    print("Top 3 predicted classes:", top_3_classes)

    <!-- # 필터링된 결과 출력 -->
    filtered_df = pre_t_df[pre_t_df['club_main_category_id(target)'].isin(top_3_classes)]

    print(f"Rows where target matches the top 3 predicted values {top_3_classes}:")

</details>

<div>
    <img width="144" alt="스크린샷 2024-05-22 오후 9 12 48" src="https://github.com/kimgusan/teenplay_server/assets/156397911/309f19ee-8900-4eb4-bcdb-54f37264fcf2">
</div>
<div>
    <img width="314" alt="스크린샷 2024-05-22 오후 9 12 30" src="https://github.com/kimgusan/teenplay_server/assets/156397911/b30730bf-1fac-4868-9201-00b90d7ed362">
</div>

<hr>

### 3. AI 사용 모델 (Ensemble 기법의 RandomForest Classifier 모델 사용) 🌟
- 사용 이유
  - 하이퍼 파라미터 튜닝을 통해 관리자가 원하는 값을 조절하기 쉬우며 데이터 scaling 조절이 필요하지 않습니다.
  - Overfitting이 잘 되지 않으며 분산을 감소시켜 정확도가 높습니다.
  - 실시간 fit이 아닌 버전업을 이용한 모델로 패치 작업 전에 학습 시 충분한 시간을 가질 수 있습니다.

<hr>
    
### 4. AI 훈련 모델 화면 🎥

- 개인 관심사
<img width="497" alt="스크린샷 2024-05-23 오후 1 04 33" src="https://github.com/kimgusan/teenplay_server/assets/156397911/273c0475-db84-4ec4-afe4-ae3d13965856">

- 위시리스트(작성) 관심사
<img width="790" alt="스크린샷 2024-05-23 오후 1 08 07" src="https://github.com/kimgusan/teenplay_server/assets/156397911/ced34def-5f1f-42a4-b0f9-28a51b0eb7f4">


- 좋아요 관심사
<img width="1115" alt="스크린샷 2024-05-23 오전 11 12 27" src="https://github.com/kimgusan/teenplay_server/assets/156397911/6de03e6b-2cf0-451f-853c-94fc864ca404">



### 5. 평가 📈

- **사용자 관심사 학습**: 해당 학습 모델을 통해 사용자의 여러 관심사를 학습시킬 수 있습니다.
- **변화 가능성**: 사용자가 선택한 관심사와 실제 영상 클릭 시 발생되는 좋아요 관심사는 초기 예상과 다를 수 있습니다.
- **로그인 전 후 모델 적용**: 사용자가 로그인하기 전에는 기본 학습 모델을 통한 다양한 관심사를 기반으로 영상을 보여줍니다. 로그인 후에는 좋아요 한 카테고리, 위시리스트 카테고리, 개인 설정 관심사에 따라 관심 있는 항목 3개의 카테고리에 대한 영상을 제공합니다.
- **비율 기반 학습**: 영상의 특성상 여러 카테고리가 혼합되어야 하기 때문에 초기 훈련 모델에서 1:1 비율 대신 3:7 비율로 학습시켜 65%의 정확도를 기준으로 유사한 값의 평가 지표를 설정했습니다.
- **패치 및 업데이트**: 향후 패치 시 새로운 학습 모델에 용이하게 대응하기 위해 각 회원별 카테고리 종류에 대한 뷰를 3개 구성해 해당 값을 조회할 수 있도록 설정했습니다.
- **사용자 경험에 따른 영상 표기 개선**: 영상을 스크롤하면서 확인할 때 사용자의 관심사에 대한 영상이 비율적으로 다수 나타나며, 사용자의 관심사에 따라 나타나는 영상이 달라진다는 것을 확인할 수 있습니다.


### 6. 트러블슈팅 🚨
#### 가. 문제 발생
- 영상 추천의 경우 사용자가 관심있는 분야가 나타나야 하지만 다양한 정보를 선택 했을 때 내가 선택한 정보에 대해서만 관련있는 영상이 나타나는 문제가 발생했습니다.
   
#### 나. 원인 추론
- Feature의 경우 사용자가 관심있는 분야만 선택하는 경우 1:1 로 분류 훈련이 진행 되기 때문에 다양한 카테고리에 대한 영상을 확인할 수 없다는 판단을 하였습니다.
   
#### 다. 해결 방안
- 단일 중요도 feature 만 고려하는 것이 아닌, 다양한 카테고리를 선택했을 때 상위 3개의 카테고리에 대한 영상을 불러오는 것으로 변경하였습니다.
- Rest API를 통해 영상을 추가적으로 불러올 때 고정된 카테고리가 아닌 다중으로 선택한 카테고리 중 랜덤으로 값을 가져와서 해당 항목으로 초기 5개, 스크롤 진행시 30개 씩 가져옴으로서 내가 선택한 카테고리에 대한 영상만 보이지 않고,
  관심있는 영상에 대하여 어느정도 비중을 가져오는 것으로 하였습니다.
- 모델 훈련 시 좋아요를 클릭한 카테고리와 실제 target의 영상 카테고리에 대하여 일정 비율을 유지하여 실제 사용 시 해당 값에 대하여 어느정도 연관성 있게 훈련을 진행하였습니다.
```
# 특정 feature 2개에 대하여 비율을 맞춰서 가져오는 함수
def match_categories(df, like_col, target_col, ratio=(3, 7)):
    like_values = df[like_col].value_counts().index
    for value in like_values:
        like_indices = df[df[like_col] == value].index
        num_target = int(len(like_indices) * ratio[1] / sum(ratio))
        target_indices = np.random.choice(like_indices, size=num_target, replace=False)
        df.loc[target_indices, target_col] = value
    return df
```
   
#### 라. 결과 확인
- 영상을 표기 할 시 (여행·동행, 운동·액티비티, 스터디) 에 대한 영상이 출력 되는 것을 확인할 수 있습니다.

<table>
  <tr>
    <th>운동</th>
    <th>여행</th>
    <th>스터디</th>
  </tr>
  <tr>
    <td><img width="357" alt="스크린샷 2024-05-23 오전 11 32 54" src="https://github.com/kimgusan/teenplay_server/assets/156397911/627b58cf-bf48-485c-b0fa-f6b14b073a3c"></td>
    <td><img width="369" alt="스크린샷 2024-05-23 오전 11 35 37" src="https://github.com/kimgusan/teenplay_server/assets/156397911/b5209a81-366f-4a16-afdf-0a13435758e6"></td>
    <td><img width="362" alt="스크린샷 2024-05-23 오후 1 02 22" src="https://github.com/kimgusan/teenplay_server/assets/156397911/8dce2262-9341-4b39-81c4-aa43a1c371af"></td>
  </tr>
</table>

### 7. 😎 느낀점 
- 머신러닝의 다양한 모델에 대해 실습하고, 분류와 회귀를 다루는 프로젝트를 진행하면서 가상 데이터로 훈련한 결과물이 실제 실무에서 어떻게 적용되는지 정확히 알지 못했습니다. 하지만 이번 프로젝트를 통해 제가 배운 모델이 실제 화면에 어떻게 출력되는지 확인할 수 있었고, 차후 버전 업데이트 시 특정 데이터를 훈련시키기 위해 view를 사용하여 유지보수할 수 있는 방법을 구현해보았습니다.

- 또한, 사용자 로그인을 통해 로그인된 유저의 데이터를 기반으로 알고리즘이 학습되어 화면에 어떻게 적용되는지를 확인할 수 있었습니다. 이 과정을 통해 모델이 실제로 어떻게 동작하는지, 그리고 사용자 맞춤형 추천 시스템이 어떻게 구축되는지를 경험할 수 있었습니다. 매우 신기하고 흥미로운 경험이었습니다.

- 다중분류 모델을 학습시키고 예측된 값을 실제 화면에 나타내는 작업을 하면서, 100% 확률로 정확한 결과를 보여주는 것이 아니라, 약 60%의 확률로 사용자에게 학습된 알고리즘을 기반으로 한 결과를 제시하면서도 다른 항목들을 함께 보여줘야 하는 적절한 합의점을 찾는 것이 어려웠습니다. 그러나 이러한 문제를 해결하면서 사용자에게 다양한 선택지를 제공할 수 있는 방법을 찾아냈고, 이를 통해 사용자 경험을 향상시킬 수 있었습니다.

  
