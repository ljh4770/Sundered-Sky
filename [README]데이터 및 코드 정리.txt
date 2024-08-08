제출 데이터 및 코드파일 설명입니다.
fog_train.csv,fog_test.csv가 원 데이터입니다. 여기서 code 파일에 있는 preprocess_data code에서 몇몇 전처리를 진행한 후 train_prepro, test_prepro.csv로 저장했습니다.
이 train_prepro을 통해 몇몇 탐색적 데이터 분석(EDA)를 진행한 코드가 EDA1~3.R 입니다.
다음으로 impute.R 코드를 사용하여 결측치 대체와 파생변수 하나를 생성하여 train_final.csv, test_final.csv로 저장했습니다.
마지막으로 optuna_HurdleTreeModel code에서 추가적인 파생변수를 생성 후 모델링을 진행하여 다시  train_final.csv, test_final.csv로 저장했습니다.

모델의 예측 결과를 기존의 fog_test.csv 의 fog_test.class에 넣은 데이터는 prediction 파일 안에 HT_prediction으로 저장되어 있습니다.

[코드파일 설명]

[preprocess_data.ipynb]
- **원본 데이터 → train3**
    - 원본 데이터 로드
    - train/ test 열 이름 통일
        - fog_train.* / gof_test.*→ *
    - -99.9, -99는 결측값으로 판단하여 NaN으로 명시하여 대체
    - 지역 구분을 위해 stn_id의 첫글자, 두번째 글자에 해당하는 열생성
        - stn_id → FirstLetter, SecondLetter
    - sun10 이상치 처리
        - 값이 1이상인 경우 → NaN으로 대체
        - 야간(20:00 ~ 05:00)에 sun10의 값이 0이 아닌 경우 → 0으로 대체
    - CA 지역의 sun10 긴 구간의 결측치 CB의 sun10 값으로 대체
        - 구간 : K년 8월 26일 00:10 ~ K년 12월 23일 21:00
        - 위 구간의 CA 지역 sun10값이 결측치인 경우 → 동일 구간의 CB의 sun10 값으로 결측치 대체
    - R에서 전처리를 계속하기 위해 전처리된 csv 저장
        - `train_prepro`.csv
        - `test_prerpo`.csv


[Impute.R]

- **train3 & test3 → train6 & test6 코드 (코드정리용)**
    - `train_prepro`.csv&`test_prepro`.csv 로드 후 전처리
        - I,J,K년도를 각각 2020,2021,2022년으로 설정한 뒤 날짜와 시간을 합쳐 시계열 객체(datetime) 생성
        - stn_id & datetime을 index로 삼아 패널 데이터 프레임 생성
    - 연속된 결측 구간을 카운트하는 함수 정의(`find_na_regions`)
        - 연속된 NA 구간의 길이와 시작, 종료 위치를 반환
    - 특정 열에 선형 보간을 적용하는 함수 정의(`interpolated_column`)
        - `find_na_regions`를 사용하여 연속된 NA 구간이 n 이하일 경우에만 선형보간을 적용
    - stn_id를 기준으로 분리하고, 정해진 변수에선형 보간을 적용하는 함수(`custom_interpolation`)
        - stn_id, datetime을 기준으로 정렬하고 stn_id별로 group_by 적용
        - 그룹 내에서 `interpolated_column`을 수행하고, ungroup()
    - 변동계수가 큰 변수인 ws10_deg, ws10_ms, sun10은 n을 6, ta는 19, hm은 25로 설정한 뒤 보간
    - 나머지 변수로 선형회귀를 학습하여 ts 보간(train)
        - 더 나은 lm 성능을 위해, 풍향은 삼각변환 뒤 풍속을 곱하여 파생변수 생성
        - 월과 시간 역시 삼각변환
        - lm학습 뒤, 결측치 행 필터링 한 뒤, lm을 통해 예측하고 결측치에 대치
        - 필요한 열(기본 변수들)만 추출하여 csv 저장
            - `train_final`.csv
    - 나머지 변수로 선형회귀를 학습하여 ts 보간(test)
        - 앞에서 사용한 변수를 그대로 사용하되, 대치된 데이터로 stn_id 대신 FirstLetter 사용하여 train에서 선형회귀 학습
        - test 동일하게 전처리 및 필터링, 예측 후 대치
        - 필요한 열(기본 변수들)만 추출하여 csv 저장
            - `test_final`.csv
    - month, day를 이용하여 1년 중 몇 번째 요일인지, time, minute을 사용하여 하루의 시간을 표현 후 각각 sin cos 삼각 변환


[utils.py]

- **유틸 함수 모듈**
    - 유틸 함수 목록
    - csi_score
        - 입력 : y_true, y_pred
        - 출력 : 없음
        - 반환: CSI score
    - print_metric
        - 입력 : y_true, y_pred
        - 출력 : 정확도, F1 스코어, CSI 스코어
        - 반환 : 없음
    - show_cm :
        - 입력 :  y_true, y_pred
        - 출력 : 혼돈행렬 시각화
        - 반환 : 없음



[HurdleTreeModel.py]
- **지역별 2stage 모델 객체 파일**
    - 1 stage 모델 후보 및 2 stage 모델 후보 XGB, LGBM, RandomForest
    - params를 통해 각 stage 및 지역별 모델 하이퍼 파라미터 초기화
        - 미지정 시 default 하이퍼 파라미터
        - 기대 하이퍼 파라미터 형식
            
            각 지역별 dictionary안에 모델의 하이퍼 파라미터 정의
            
    - 1 stage 개요
        - 목적 : 입력 데이터의 class가 4인지, 1, 2, 3인지 이진 분류
        - 학습 : 입력 데이터의 class를 변환하여 전체 데이터에 대해 학습
            - 4 → 0
            - 1, 2, 3 → 1
        - 예측 : 전체 입력 데이터를  0과 1로 예측
    - 2 stage 개요
        - 목적 : 입력 데이터의 class를 1, 2, 3으로 다중 분류
        - 학습 : 입력 데이터 중 class가 1, 2, 3에 속하는 데이터를 사용하여 학습
        - 예측: 1 stage에서 1로 예측한 데이터에 대해 1, 2, 3으로 예측
    - 전체 예측 방식 (predict_oneshot 함수)
        - 1 stage 예측
        - 2 stage 예측
        - 1 stage에서 0으로 예측한 데이터는 최종 class가 4임을 의미
        - 1 stage에서 1로 예측한 데이터에 대하여 2 stage의 예측값 사용
    - 함수 구성
        - stage별 학습, 예측, 확률 예측
        - 1 stage의 F1 스코어 최대화하는 threshold 계산
        - 각 stage의 모델 지정
        - stage 별 데이터 생성
            - 1stage:  class값 변환 (1, 2, 3 → 1 / 4 → 0)
            - 2stage: class가 1, 2, 3에 속하는 행만 필터링
        - 입력데이터에 대해 1, 2 stage 예측을 순차적으로 진행하여 end-to-end로 예측
        
        ---

[HurdleTree_optuna_study.joblib]	
**optuna hyperparameter tuning 로그**

