import numpy as np
import pandas as pd

# from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier


# 두 개의 stage로 구성된 분류 모델 객체 선언
# 1 stage 목적 : 입력 데이터에 대하여 시정 변수 계급을 1, 2, 3에 속하는 지 혹은 4에 속하는 지 이진분류
# -> 학습 데이터의 class를 변환하여 학습 : 1, 2, 3 -> 1 / 4 -> 0
# 2 stage 목적 : 입력 데이터에 대하여 시정 변수 계급을 1, 2, 3으로 분류
# -> 학습 데이터 중 class가 1, 2, 3에 속하는 데이터를 사용하여 학습
class HurdleTreeModel:
    # 객체 초기화
    def __init__(self,
                 params:dict = None,
                 SEED:int = 625):
        # 랜덤 시드 설정을 위한 상수 선언
        self.SEED = SEED
        
        # 1stage F1을 최대화하는 threshold 저장
        self.cutoffs = dict(
            XGB = dict(A = -1, B = -1, C = -1, D = -1, E = -1),
            LGBM = dict(A = -1, B = -1, C = -1, D = -1, E = -1),
            RNF = dict(A = -1, B = -1, C = -1, D = -1, E = -1)
            )
        
        # 지역 / stage 별 모델의 하이퍼 파라미터 초기화
        # 하이퍼 파라미터를 지정하지 않은 경우
        self.params_1stage = dict(
            XGB = dict(A = dict(random_state = SEED),
                        B = dict(random_state = SEED),
                        C = dict(random_state = SEED),
                        D = dict(random_state = SEED),
                        E = dict(random_state = SEED)),
            LGBM = dict(A = dict(random_state = SEED),
                        B = dict(random_state = SEED),
                        C = dict(random_state = SEED),
                        D = dict(random_state = SEED),
                        E = dict(random_state = SEED)),
            RNF = dict(A = dict(random_state = SEED),
                       B = dict(random_state = SEED),
                       C = dict(random_state = SEED),
                       D = dict(random_state = SEED),
                       E = dict(random_state = SEED)),            
            )
        self.params_2stage = dict(
            XGB = dict(A = dict(random_state = SEED),
                       B = dict(random_state = SEED),
                       C = dict(random_state = SEED),
                       D = dict(random_state = SEED),
                       E = dict(random_state = SEED)),
            LGBM = dict(A = dict(random_state = SEED),
                        B = dict(random_state = SEED),
                        C = dict(random_state = SEED),
                        D = dict(random_state = SEED),
                        E = dict(random_state = SEED)),
            RNF = dict(A = dict(random_state = SEED),
                       B = dict(random_state = SEED),
                       C = dict(random_state = SEED),
                       D = dict(random_state = SEED),
                       E = dict(random_state = SEED)),            
            )
        if params != None: # 하이퍼 파라미터를 지정한 경우
            self.params_1stage = params['params_1stage']
            self.params_2stage = params['params_2stage']
        
        # 모델 초기화
        # XGB, LGBM, RandomForest로 stage 별 모델 후보군 선정
        # -> self.set_1stage_model() 함수를 호출하여 최고 성능의 모델을 최종 모델로 선언
        self.model_1stage = dict(
            XGB = dict(A = XGBClassifier(**self.params_1stage['XGB']['A'], n_jobs = -1),
                       B = XGBClassifier(**self.params_1stage['XGB']['B'], n_jobs = -1),
                       C = XGBClassifier(**self.params_1stage['XGB']['C'], n_jobs = -1),
                       D = XGBClassifier(**self.params_1stage['XGB']['D'], n_jobs = -1),
                       E = XGBClassifier(**self.params_1stage['XGB']['E'], n_jobs = -1)),
            LGBM = dict(A = LGBMClassifier(**self.params_1stage['LGBM']['A'], n_jobs = -1),
                        B = LGBMClassifier(**self.params_1stage['LGBM']['B'], n_jobs = -1),
                        C = LGBMClassifier(**self.params_1stage['LGBM']['C'], n_jobs = -1),
                        D = LGBMClassifier(**self.params_1stage['LGBM']['D'], n_jobs = -1),
                        E = LGBMClassifier(**self.params_1stage['LGBM']['E'], n_jobs = -1)),
            RNF = dict(A = RandomForestClassifier(**self.params_1stage['RNF']['A'], n_jobs = -1),
                       B = RandomForestClassifier(**self.params_1stage['RNF']['B'], n_jobs = -1),
                       C = RandomForestClassifier(**self.params_1stage['RNF']['C'], n_jobs = -1),
                       D = RandomForestClassifier(**self.params_1stage['RNF']['D'], n_jobs = -1),
                       E = RandomForestClassifier(**self.params_1stage['RNF']['E'], n_jobs = -1))         
            )
        self.model_2stage = dict(
            XGB = dict(A = XGBClassifier(**self.params_2stage['XGB']['A'], n_jobs = -1),
                       B = XGBClassifier(**self.params_2stage['XGB']['B'], n_jobs = -1),
                       C = XGBClassifier(**self.params_2stage['XGB']['C'], n_jobs = -1),
                       D = XGBClassifier(**self.params_2stage['XGB']['D'], n_jobs = -1),
                       E = XGBClassifier(**self.params_2stage['XGB']['E'], n_jobs = -1)),
            LGBM = dict(A = LGBMClassifier(**self.params_2stage['LGBM']['A'], n_jobs = -1),
                        B = LGBMClassifier(**self.params_2stage['LGBM']['B'], n_jobs = -1),
                        C = LGBMClassifier(**self.params_2stage['LGBM']['C'], n_jobs = -1),
                        D = LGBMClassifier(**self.params_2stage['LGBM']['D'], n_jobs = -1),
                        E = LGBMClassifier(**self.params_2stage['LGBM']['E'], n_jobs = -1)),
            RNF = dict(A = RandomForestClassifier(**self.params_2stage['RNF']['A'], n_jobs = -1),
                       B = RandomForestClassifier(**self.params_2stage['RNF']['B'], n_jobs = -1),
                       C = RandomForestClassifier(**self.params_2stage['RNF']['C'], n_jobs = -1),
                       D = RandomForestClassifier(**self.params_2stage['RNF']['D'], n_jobs = -1),
                       E = RandomForestClassifier(**self.params_2stage['RNF']['E'], n_jobs = -1))              
            )        
        
        # 최종 모델 dictionary 초기화
        # stage 별 최종 모델
        self.best_1stage_key = None
        self.best_2stage_key = None
        # 최종 모델의 하이퍼 파라미터
        self.best_model = dict(
            model_1stage = dict(A = None,
                                B = None,
                                C = None,
                                D = None,
                                E = None),
            cutoffs = dict(A = -1, B = -1, C = -1, D = -1, E = -1),
            model_2stage = dict(A = None,
                                B = None,
                                C = None,
                                D = None,
                                E = None),
        )
    
    # 1stage 모델 지역별 fit
    def fit_1stage(self, X:pd.DataFrame, Y:pd.DataFrame) -> None: # train 데이터 사용
        for model in self.model_1stage.keys(): # 1stage 모델 후보군
            for stn in self.model_1stage[model].keys(): # 지역 A ,B, C, D, E
                stn_idx = X.loc[X['FirstLetter'] == stn].index # 해당 지역의 데이터 인덱스 필터링
                # 해당 지역의 모델 학습
                self.model_1stage[model][stn].fit(
                    X.drop('FirstLetter', axis = 1, inplace = False).iloc[stn_idx],
                    Y.iloc[stn_idx]
                )

    # F1 스코어를 최고로하는 threshold(cut off point) 구하는 것을 vecotrization을 통해 계산 속도 향상
    # self.set_cutoff() 함수에서 필요함
    def find_f1_cutoff(self, y_true, y_proba, num_interval = 200):
        # 행 개수 할당
        n = y_true.shape[0]
        
        # 차원 조작
        y_true = np.array(y_true).reshape(-1, 1)
        y_proba = y_proba[:, 1].reshape(-1, 1)
        
        # 이진 분류될 예측값 공간 정의
        y_pred = np.zeros(shape = (n, num_interval))
        # F1 스코어를 계산할 threshold 후보 지정 ex) num_interval = 100 -> thresholds := 0.0, 0.01, ... 0.99
        thresholds = np.arange(0, num_interval, 1) / num_interval
        # numpy array의 broadcast 기능을 이용하여 모든 열에 threshold 후보 지정
        y_pred = y_pred + thresholds
        
        # 입력 받은 클래스 1일 확률을 이용하여 이진 분류
        # threshold - 확률 < 0 == threshold < 확률 == 클래스 1로 판단
        y_proba_bin = np.where(np.subtract(y_pred, y_proba) < 0, 1, 0).reshape(-1, num_interval)
        
        # 혼돈행렬 원소 계산을 위함
        sub = y_true - y_proba_bin # -> 1인 경우 FN case, -1인 경우 FP case
        add = y_true + y_proba_bin # -> 2인 경우 TP case, 0인 경우 TN case
        
        FN = np.sum(sub == 1, axis = 0)
        FP = np.sum(sub == -1, axis = 0)
        TP = np.sum(add == 2, axis = 0)    
        
        # 혼돈행렬 원소값에 기반하여 precision. recall 값 계산
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        
        # 모든 threshold 후보에 대한 F1 스코어 계산
        f1_list = 2 * precision * recall / (precision + recall + 1e-10)
        
        # F1 스코어가 최고가 되는 threshold 반환
        return np.argmax(f1_list) / num_interval
    
    # 데이터(validation data)를 바탕으로 최고 F1 스코어에 해당하는 cutoff 값 저장
    def set_cutoff(self, X:pd.DataFrame, Y:pd.Series) -> dict: # valid 데이터 사용
        for model in self.model_1stage.keys(): # 1stage 모델 후보군
            for stn in self.model_1stage[model].keys(): # 지역 A ,B, C, D, E
                stn_idx = X.loc[X['FirstLetter'] == stn].index # 해당 지역의 데이터 인덱스 필터링
                # 각 클래스 (0, 1)에 속할 확률 예측
                y_proba = self.model_1stage[model][stn].predict_proba(X.drop('FirstLetter', axis = 1, inplace = False).iloc[stn_idx])
                # vvv --- vectorization 이전 코드 --- vvv
                # score_list = np.zeros(shape = 1000) # score list 초기화
                # for i in range(0, 1000, 1):
                #     cutoff = i / 1000
                #     y_pred = np.where(y_proba[:, 1] > cutoff, 1, 0) # 조건의 참일 경우 1, 아닌 경우 0
                #     score = metric(Y.iloc[stn_idx], y_pred) # score 계산
                #     score_list[i] = score
                # 최고 성능의 cutoff 저장
                # self.cutoffs[model][stn] = np.argmax(score_list) / 100
                # ^^^ --- vectorization 이전 코드 --- ^^^
                # 최고 F1 스코어에 해당하는 threshold를 객체 변수에 저장
                self.cutoffs[model][stn] = self.find_f1_cutoff(Y.iloc[stn_idx], y_proba)
        
        return self.cutoffs
    
    # 1 stage 모델을 사용하여 예측
    def predict_1stage(self, X:pd.DataFrame, mode:str):
        # train/valid 데이터에 대해 예측 -> 모델 선정
        if mode == 'train':
            # 모델 별 반환 할 예측값 리스트를 dictionary로 선언
            pred_dict = dict(
                XGB = None,
                LGBM = None,
                RNF = None
            )
            
            # 1stage 모델 후보군 [XGB, LGBM, RNF]
            for model in self.model_1stage.keys(): # 1stage 모델 후보군
                # 반환할 예측 값 리스트
                y_pred = []
                
                for stn in self.model_1stage[model].keys(): # 지역 A ,B, C, D, E
                    stn_idx = X.loc[X['FirstLetter'] == stn].index # 해당 지역의 데이터 인덱스 필터링
                    # 각 클래스 (0, 1)에 속할 확률 예측
                    y_proba = self.model_1stage[model][stn].predict_proba(X.drop('FirstLetter', axis = 1, inplace = False).iloc[stn_idx])
                    # 해당 지역 모델의 threshold 할당
                    cutoff = self.cutoffs[model][stn]
                    # 확률 및 threshold를 이용하여 이진 분류
                    pred = np.where(y_proba[:, 1] > cutoff, 1, 0)
                    # 해당 지역의 예측값 concat
                    y_pred = y_pred + pred.tolist()
                # 해당 모델 dictionary에 할당
                pred_dict[model] = np.array(y_pred)
            
            return pred_dict
        
        # test 데이터에 대한 예측
        elif mode == 'test':
            # 반환할 예측 값 리스트
            y_pred = []
            # self.set_1stage_model() 함수로 선택한 모델을 사용하여 예측
            for stn in self.best_model['model_1stage'].keys(): # 지역 A ,B, C, D, E
                stn_idx = X.loc[X['FirstLetter'] == stn].index # 해당 지역의 데이터 인덱스 필터링
                # 각 클래스 (0, 1)에 속할 확률 예측
                y_proba = self.best_model['model_1stage'][stn].predict_proba(X.drop('FirstLetter', axis = 1, inplace = False).iloc[stn_idx])
                # 해당 지역 모델의 threshold 할당
                cutoff = self.best_model['cutoffs'][stn]
                # 확률 및 threshold를 이용하여 이진 분류
                pred = np.where(y_proba[:, 1] > cutoff, 1, 0)
                # 해당 지역의 예측값 concat
                y_pred = y_pred + pred.tolist()
            
            return np.array(y_pred)
            
    # 1stage 모델 선택
    def set_1stage_model(self, model_key:str = 'XGB') -> None:
        # 선택할 모델의 key값 입력 (XGB, LGBM, RNF)
        self.best_1stage_key = model_key
        for stn in self.best_model['model_1stage'].keys(): # 지역 A ,B, C, D, E
            # 선택한 모델의 threshold를 
            self.best_model['cutoffs'][stn] = self.cutoffs[model_key][stn]
            self.best_model['model_1stage'][stn] = self.model_1stage[model_key][stn]
    
    # 1stage class 별 확률 예측 - test 전용(학습 시 사용 X)
    def predict_proba_1stage(self, X:pd.DataFrame):
        # 반환할 예측 확률 값 리스트
        y_proba = []
        
        # self.set_1stage_model() 함수로 선택한 모델을 사용하여 예측
        for stn in self.best_model['model_1stage'].keys(): 
            stn_idx = X.loc[X['FirstLetter'] == stn].index # 해당 지역의 데이터 인덱스 필터링
            # 각 클래스 (0, 1)에 속할 확률 예측
            proba = self.best_model['model_1stage'][stn].predict_proba(X.drop('FirstLetter', axis = 1, inplace = False).iloc[stn_idx])
            # 해당 지역의 예측값 concat
            y_proba = y_proba + proba.tolist()        
        
        return np.array(y_proba)

    # 2stage 모델 지역별 fit
    def fit_2stage(self, X:pd.DataFrame, Y:pd.Series) -> None: # train 데이터
        for model in self.model_2stage.keys():
            for stn in self.model_2stage[model].keys():
                stn_idx = X.loc[X['FirstLetter'] == stn].index
                if model != 'XGB':
                    self.model_2stage[model][stn].fit(
                        X.drop('FirstLetter', axis = 1, inplace = False).iloc[stn_idx],
                        Y.iloc[stn_idx]
                    )
                
                # XGBClassifier 모듈은 target class가 0부터 시작하도록 제한하기 때문에 예외 처리
                # 원래 class에 1을 빼준 뒤 학습: 1, 2, 3 -> 0, 1, 2
                # XGB로 예측시 예측값에 1을 더하도록 구현하여 원래 class 값으로 복원
                elif model == 'XGB':
                    self.model_2stage[model][stn].fit(
                        X.drop('FirstLetter', axis = 1, inplace = False).iloc[stn_idx],
                        Y.iloc[stn_idx] - 1 
                    )                

    # 2 stage 모델을 사용하여 예측
    def predict_2stage(self, X:pd.DataFrame, mode:str) -> dict:
        if mode == 'train':
            pred_dict = dict(
                XGB = None,
                LGBM = None,
                RNF = None
            )
            
            for model in self.model_2stage.keys():
                y_pred = []
                for stn in self.model_2stage[model].keys():
                    stn_idx = X.loc[X['FirstLetter'] == stn].index
                    pred = self.model_2stage[model][stn].predict(X.drop('FirstLetter', axis = 1, inplace = False).iloc[stn_idx])
                    
                    # XGB의 경우 0, 1, 2를 도출하도록 학습했기 때문에 1을 더하여 원래의 class 값으로 복원
                    if model == 'XGB':
                        pred = pred + 1
                    y_pred = y_pred + pred.tolist()
                pred_dict[model] = y_pred
            
            return pred_dict
        
        elif mode == 'test':
            y_pred = []
            
            for stn in self.best_model['model_2stage'].keys():
                stn_idx = X.loc[X['FirstLetter'] == stn].index
                pred = self.best_model['model_2stage'][stn].predict(X.drop('FirstLetter', axis = 1, inplace = False).iloc[stn_idx])
                
                # XGB의 경우 0, 1, 2를 도출하도록 학습했기 때문에 1을 더하여 원래의 class 값으로 복원
                if self.best_2stage_key == 'XGB':
                    pred = pred + 1
                y_pred = y_pred + pred.tolist()
            
            return np.array(y_pred)  
            
    # 2stage 모델 선택
    def set_2stage_model(self, model_key:str = 'LGBM') -> None:
        self.best_2stage_key = model_key
        for stn in self.best_model['model_2stage'].keys():
            self.best_model['model_2stage'][stn] = self.model_2stage[model_key][stn]

    # 2stage class 별 확률 예측 - test 전용(학습 시 사용 X)
    def predict_proba_2stage(self, X:pd.DataFrame):
        y_proba = np.zeros(shape = (X.shape[0], 3))
        
        for stn in self.best_model['model_2stage'].keys():
            stn_idx = X.loc[X['FirstLetter'] == stn].index
            proba = self.best_model['model_2stage'][stn].predict_proba(X.drop('FirstLetter', axis = 1, inplace = False).iloc[stn_idx])
            y_proba[stn_idx,:] += proba
        
        return np.array(y_proba)

    # 데이터프레임의 class를 binary로 만듬
    def gen_1stage_data(self, df:pd.DataFrame) -> pd.DataFrame: # train, valid 따로 생성
        res_df = df.copy()
        res_df.loc[res_df['class'] != 4, 'class'] = 1 # class가 1, 2, 3 중 하나라면 1
        res_df.loc[res_df['class'] == 4, 'class'] = 0 # class가 4라면 0 
        
        return res_df.copy()
    
    # 데이터프레임의 class 중 1, 2, 3만 남김
    def gen_2stage_data(self, df:pd.DataFrame) -> pd.DataFrame: # train, valid 따로 생성
        drop_idx = df.loc[df['class'] == 4].index
        res_df = df.drop(drop_idx, axis = 0, inplace = False).copy()
        res_df.reset_index(drop = True, inplace = True)
        
        return res_df

    # self.set_1stage_model, set_2stage_model을 통해 self.best_model을 정의해주어야 동작
    def predict_oneshot(self, X:pd.DataFrame) -> np.array: # valid / test 데이터
        # 1 stage 예측
        y_pred_1stage = self.predict_1stage(X, mode = 'test')
        
        # 1 stage에서 1로 예측한 인덱스 필터링
        # 1, 2, 3에 속할 것으로 예측
        one_idx = np.argwhere(y_pred_1stage == 1).flatten()
        
        if len(one_idx) > 0:
            # 2 stage 예측: 1, 2, 3으로 예측
            y_pred_2stage = self.predict_2stage(X, mode = 'test')
        else:
            y_pred_2stage = np.array([])
        
        # 1 stage의 0은 원래 class에서 4를 의미
        # 0 -> -1 -> 4
        # 1 ->  0 -> 0
        # 0으로 계산된 행은 2 stage의 예측값(1, 2, 3)으로 채움
        y_pred = (y_pred_1stage - 1) * -4

        # 1 stage에서 1로 예측한 인덱스를 2 stage 예측값으로 채움
        for idx in one_idx:
            y_pred[idx] = y_pred_2stage[idx]

        # 최족 예측값 반환
        return y_pred