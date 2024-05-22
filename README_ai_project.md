## 틴플레이 영상 재생 알고리즘 AI 프로젝트 보고서

틴플레이: http://teenplay.store

### ♠️ 목차

1. AI 서비스 설명
2. AI 사전 훈련 모델
    - 필요 데이터 선택 (데이터 수집)
    - 데이터 전처리
    - 훈련 모델 선택
    - 훈련 모델 검증
3. AI 사용 모델 
4. AI 훈련 모델 화면
5. 평가

#### 1. AI 서비스 설명

-   사용자가 관심있는 분야, 영상 좋아요를 가장 많이 누른 케이스, 위시리스트에 대한 회원 관심사에 대하여 영상을 보여주는 ai 추천 서비스

#### 2. AI 사전 훈련 모델

-   데이터 수집

    features

    1. u_info_category
    2. u_wishlist_category
    3. u_favorite_teenplay_like_club_category

    target

    1. club_category

<details>
  <summary>Click all Code</summary>

##### 가) 클럽 리스트에 대하여 전체 count 확인.

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
- 이미지

##### 나) 크롤링을 통해 가져온 각 category별 title 데이터 프레임 생성.

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
- 이미지

-   데이터 전처리

#### 가) 가져온 데이터에 대하여 cateogory 별로 데이터 분류

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

-   이미지

#### 나) 영상 데이터에 대하여 user가 좋아하는 category, wishlist category 랜덤값으로 부여

user category의 경우 좋아요를 누른 카테고리 이기 때문에 3:7 비율로 카테고리 값을 넣어준다.

<details>
  <summary>Click all Code</summary>

    <!-- # 3:7 비율로 숫자를 맞추기 위한 함수 -->
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

-   훈련 모델 선택

#### 가) 결정트리 모델 (교차검증) 훈련

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

-   이미지 추가

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

-   이미지 추가

#### 나) KNN 최근접 이웃 모델 (교차검증) 훈련

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
- 이미지 추가

##### 평가지표 확인

<details>
  <summary>Click all Code</summary>

    knn_c = g_knn_c.best_estimator_
    prediction = knn_c.predict(X_test)
    get_evaluation(y_test, prediction, knn_c, X_test)

</details>

-   이미지 추가

#### 다) AdaboostClassifier 모델 (교차검증) 사용

<details>
  <summary>Click all Code</summary>

    <!-- # 데이터의 훈련 시 시간이 얼마나 걸릴 지 모르기 때문에 우선 1부터 조금씩 낮아지도록 훈련 진행 -->
    from sklearn.ensemble import AdaBoostClassifier

    parameters = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.7, 0.8, 0.9, 1]
    }

    g_ada = GridSearchCV(AdaBoostClassifier(), param_grid= parameters, cv=3, refit=True, return_train_score=True, n_jobs = -1)

</details>

##### 평가지표 확인

<details>
  <summary>Click all Code</summary>

    knn_c = g_knn_c.best_estimator_
    prediction = knn_c.predict(X_test)
    get_evaluation(y_test, prediction, knn_c, X_test)

</details>

#### 라) Randomforest 모델 (교차검증) 사용

<details>
  <summary>Click all Code</summary>

    <!-- # 데이터의 훈련 시 시간이 얼마나 걸릴 지 모르기 때문에 우선 1부터 조금씩 낮아지도록 훈련 진행 -->
    from sklearn.ensemble import AdaBoostClassifier

    parameters = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.7, 0.8, 0.9, 1]
    }

    g_ada = GridSearchCV(AdaBoostClassifier(), param_grid= parameters, cv=3, refit=True, return_train_score=True, n_jobs = -1)

</details>

##### 평가지표 확인

<details>
  <summary>Click all Code</summary>

    knn_c = g_knn_c.best_estimator_
    prediction = knn_c.predict(X_test)
    get_evaluation(y_test, prediction, knn_c, X_test)

</details>

#### 마) 모델 (교차검증) 사용

<details>
  <summary>Click all Code</summary>

    <!-- # 데이터의 훈련 시 시간이 얼마나 걸릴 지 모르기 때문에 우선 1부터 조금씩 낮아지도록 훈련 진행 -->
    from sklearn.ensemble import AdaBoostClassifier

    parameters = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.7, 0.8, 0.9, 1]
    }

    g_ada = GridSearchCV(AdaBoostClassifier(), param_grid= parameters, cv=3, refit=True, return_train_score=True, n_jobs = -1)

</details>

##### 평가지표 확인

<details>
  <summary>Click all Code</summary>

    knn_c = g_knn_c.best_estimator_
    prediction = knn_c.predict(X_test)
    get_evaluation(y_test, prediction, knn_c, X_test)

</details>

-   모델 검증

#### 검증 데이터 분류 후 Randomforest 모델 훈련

#### 훈련 데이터와 평가 지표 비고하여 분석

##### 검증된 모델에 대하여 predict_proba 를 이용한 target의 확률 확인

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

-   이미지 추가

##### 사용자에게 입력된 값에 대하여 결과 모델 확인

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

    <!-- # 상위 3개의 확률에 해당하는 클래스 추출 -->
    top_3_indices = np.argsort(prediction_proba[0])[-1:][::-1]
    top_3_classes = rfc.classes_[top_3_indices]

    print("Top 3 predicted classes:", top_3_classes)

    <!-- # 필터링된 결과 출력 -->
    filtered_df = pre_t_df[pre_t_df['club_main_category_id(target)'].isin(top_3_classes)]

    print(f"Rows where target matches the top 3 predicted values {top_3_classes}:")

</details>

-   출력 결과 이미지 확인

    -   pkl 파일 출력 까지 확인

</details>

### 3. AI 플로우 차트

-   플로우 차트 이미지 그림 확인

### 4. AI 훈련 모델 화면

-   AI 사전 훈련 모델 확인

### 5. 평가

-   해당 모델의 경우 RandomForest 모델을 사용하여 기존에 학습된 모델에 추가 학습이 불가합니다.
-   모델의 버젼관리를 통해 불규칙 적인 사용자의 관심사에 대하여 영상을 판단 하는 것이 중요합니다.
-
