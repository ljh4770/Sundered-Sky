import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score

# 공모전 평가 지표
def csi_score(y_true, y_pred):
    # 혼돈행렬 생성
    cm = confusion_matrix(y_true, y_pred)
    cm[-1, -1] = 0 # C44 제외

    # 분자 := 대각원소합 (C44 제외)
    H = np.diagonal(cm).sum() 

    # calculate score: H / (H + M + F)
    score = H / (cm.sum() + 1e-7) # 0으로 나누는 것 방지

    return score

# 평가 지표 출력함수 - 정확도, f1스코어, CSI 스코어
def print_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average = 'macro')
    csi = csi_score(y_true, y_pred)
    print(f"Accuracy : {acc: .4%}")
    print(f"F1  score: {f1: .4%}")
    print(f"CSI score: {csi: .4%}")
    
# 혼돈행렬 시각화
def show_cm(y_true, y_pred, figsize: tuple = (8, 8)) -> None:
    # 혼돈행렬 생성
    cm = confusion_matrix(y_true, y_pred, labels = [i for i in range(1, 5)])
    
    # 틱 라벨 생성
    ticklabels = ['v' + str(i).zfill(2) for i in range(1, 5, 1)]
    
    fig, ax = plt.subplots(figsize = figsize) # plot 크기 지정
    
    # 혼돈행렬 시각화
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
                xticklabels = ticklabels,
                yticklabels = ticklabels)
    
    # x, y축 이름 지정
    ax.set_xlabel("Prediction", fontsize = 15)
    ax.set_ylabel("True", fontsize = 15)
    
    # 제목 지정
    ax.set_title("Confusion Matrix",
                 fontsize = 20)
    
    # plot 표시
    fig.show()
    
    return None
    
