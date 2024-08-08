library(lubridate)
library(forecast)
library(xts)
library(zoo)
library(ggplot2)
library(skimr)
library(GGally)
library(dplyr)
library(corrplot)
library(plm)
library(data.table)
library(magrittr)
library(tidyr)

setwd(getwd())

######################### EDA1 #########################
##WS10_deg,WS10_ms,RE EDA codes

train = fread('../data/train_prepro.csv')

train %<>%
  mutate(
    stn_id = as.factor(stn_id))
train %<>% mutate(year=case_when(year=='I'~2020,year=='J'~2021,year=='K'~2022))

train$datetime <- make_datetime(year = train$year, month = train$month, 
                                day = train$day, hour = train$time, min = train$minute)
train$re <- as.factor(train$re)
train <- train %>%
  mutate(
    wind_x = ws10_ms * cos(ws10_deg * pi / 180),
    wind_y = ws10_ms * sin(ws10_deg * pi / 180))
#wind_x <- ws10_ms * cos(ws10_deg * pi / 180) # 동쪽으로 부는 바람의 세기
#wind_y <- ws10_ms * sin(ws10_deg * pi / 180) # 북쪽으로 부는 바람의 세기
# datetime 변수를 POSIXct 타입으로 변환
train$datetime <- as.POSIXct(train$datetime)
# 월 변수를 삼각함수로 변환
train$month_sin <- sin(2 * pi * train$month / 12)
train$month_cos <- cos(2 * pi * train$month / 12)

# 시각 변수를 삼각함수로 변환
train$hour_sin <- sin(2 * pi * train$time / 24)
train$hour_cos <- cos(2 * pi * train$time / 24)

train_selected <- train %>% select(wind_x,wind_y,ta,hm,sun10,ts,vis1)
# 히스토그램 생성
train_selected %>%
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = value)) +
  facet_wrap(~ variable, scales = "free") +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  theme_minimal()

cor_matrix <- cor(train_selected, use = "complete.obs")
print(cor_matrix)
corrplot(cor_matrix, method = "color", type = "upper", 
         addCoef.col = "black",
         title = paste("Correlation Matrix for all station"), 
         mar = c(0, 0, 2, 0))




# re별 변수 플랏. 
## boxplot

# Boxplot 시각화 함수
plot_boxplot <- function(data, variable) {
  ggplot(data, aes(x = re, y = get(variable), fill = re)) +
    geom_boxplot() +
    labs(title = paste("Boxplot of", variable, "by rain event"),
         x = "re",
         y = variable) +
    theme_minimal()
}

# 변수 목록
variables <- c("ta", "ts", "hm", "sun10", "wind_x", "wind_y")
train_to_vis <- as.data.frame(train)

# 각 변수에 대해 boxplot 생성
for (variable in variables) {
  print(plot_boxplot(train_to_vis, variable))
}
##violin plot
# Violin plot 시각화 함수
plot_violin <- function(data, variable) {
  ggplot(data, aes(x =re, y = get(variable), fill = re)) +
    geom_violin() +
    labs(title = paste("Violin Plot of", variable, "by re"),
         x = "re",
         y = variable) +
    theme_minimal()
}

# 각 변수에 대해 violin plot 생성
for (variable in variables) {
  print(plot_violin(train_to_vis, variable))
}

## RE vs class EDA 

# class 변수의 분포 시각화 함수
plot_class_distribution <- function(data) {
  ggplot(data, aes(x = class, fill = re)) +
    geom_bar(position = "dodge") +
    labs(title = "Distribution of Class by Rain Event",
         x = "Class",
         y = "Count",
         fill = "Rain Event") +
    theme_minimal() +
    theme(legend.position = "top")
}

# 시각화 실행
print(plot_class_distribution(train_to_vis[train_to_vis$class < 4 & !is.na(train_to_vis$class), ]))
print(plot_class_distribution(train_to_vis[train_to_vis$class==4,]))

print(table(train$class, train$re))

class_re_df = as.data.frame(table(train$class, train$re))

# 비율 계산 (각 class에 대해 re == 1의 비율)
class_re_df <- class_re_df %>%
  group_by(Var1) %>%
  mutate(total = sum(Freq),
         proportion = Freq / total) %>%
  filter(Var2 == 1) %>%
  select(Var1, proportion)

# 테이블 출력
print(class_re_df)


# re가 1인 데이터 필터링
filtered_data <- train %>% filter(re == 1 )

# class 분포 시각화
ggplot(filtered_data, aes(x = factor(class, ordered = TRUE))) +
  geom_bar() +
  labs(title = "Class Distribution for re == 1",
       x = "Class",
       y = "Count") +
  theme_minimal()





####PLM

# plm 적합 후 결과 확인
fixed_effects_model <- plm(ts ~ wind_x + wind_y + re + ta + sun10 + hm , 
                           data = train, 
                           index = c("stn_id", "datetime"), 
                           model = "within")
# 모델 요약
summary(fixed_effects_model)










