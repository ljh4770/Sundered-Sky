library(data.table)
library(magrittr)
library(tidyr)
library(dplyr)
library(lubridate)
library(forecast)
library(xts)
library(zoo)
library(plm)

setwd(getwd())
######################### 전처리 ######################### 
train = fread('../data/train_prepro.csv')
# 날짜 및 시간 결합하여 시계열 객체 생성
train %<>%
  mutate(stn_id = as.factor(stn_id),
         year=case_when(year=='I'~2020,year=='J'~2021,year=='K'~2022))
train$datetime <- make_datetime(year = train$year, month = train$month, day = train$day, hour = train$time, min = train$minute)

# 패널 데이터 프레임 생성
ptrain <- pdata.frame(train, index = c("stn_id", "datetime"),stringsAsFactors = T)
ptrain$re <- as.factor(ptrain$re)
ptrain$datetime <- as.POSIXct(ptrain$datetime)
ptrain$class = factor(ptrain$class,levels = c(1,2,3,4), ordered = TRUE)

# 요일 합산 함수
MDtoD <- function(year, month, day){
  year = (year == 2020)
  tmp <- cumsum(c(0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30))
  day = tmp[month] + day + year * (month > 2)
  return(day)
}

################### 선형보간 함수 제작 ################### 
# 연속된 NA 구간 길이와 시작, 종료 위치를 반환하는 함수
find_na_regions <- function(x) {
  rle_result <- rle(is.na(x))
  lengths <- rle_result$lengths
  values <- rle_result$values
  region_info <- data.frame(
    start = cumsum(lengths) - lengths + 1,
    end = cumsum(lengths),
    length = lengths,
    is_na = values)
  region_info <- region_info[region_info$is_na, ]
  return(region_info)
}

# 특정 열에 대해 선형 보간을 적용하는 함수
interpolate_column <- function(df, col_name, n) {
  na_regions <- find_na_regions(df[[col_name]])
  interpolated_values <- zoo::na.approx(df[[col_name]], na.rm = FALSE)
  for (i in 1:nrow(na_regions)) {
    if (na_regions$length[i] <= n) {
      df[[col_name]][na_regions$start[i]:na_regions$end[i]] <- interpolated_values[na_regions$start[i]:na_regions$end[i]]
    }}
  return(df)
}

# 모든 변수에 대해 선형 보간을 적용하는 함수
custom_interpolation <- function(data, variables, n) {
  data <- data %>% arrange(stn_id, datetime)
  
  interpolated_data <- data %>%
    group_by(stn_id) %>%
    group_modify(~ {
      for (var in variables) {
        .x <- interpolate_column(.x, var,n)
      }
      .x
    }) %>% ungroup()
  return(interpolated_data)
}

# 연속적인 결측치 확인 함수
check_consecutive_na <- function(data, column) {
  # NA 여부 확인
  na_run <- is.na(data[[column]])
  
  # Run Length Encoding (RLE) 적용
  rle_na <- rle(na_run)
  
  # 연속적인 NA의 길이와 위치 확인
  consecutive_na_lengths <- rle_na$lengths[rle_na$values]
  consecutive_na_positions <- which(rle_na$values)
  
  # 결과 데이터 프레임 생성
  result <- data.frame(
    start_position = unlist(lapply(consecutive_na_positions, function(x) sum(rle_na$lengths[1:(x - 1)]) + 1)),
    length = consecutive_na_lengths
  )
  # 길이가 1보다 큰 경우만 필터링
  #result <- result[result$length > 6, ]
  return(result)
}

################### 보간 ################### 
# 보간할 변수들
variables_6 <- c("ws10_deg", "ws10_ms", "sun10")
# 함수 호출
ptrain <- custom_interpolation(ptrain, variables_6, 6)
ptrain <- custom_interpolation(ptrain, "ta", 19)
ptrain <- custom_interpolation(ptrain, "hm", 25)




######################### LM을 사용한 ts 보간 ######################### 

################### 1. train ts <- ta
ptrain <- ptrain %>%
  mutate(wind_x = ws10_ms * cos(ws10_deg * pi / 180),
         wind_y = ws10_ms * sin(ws10_deg * pi / 180))

# 월 변수를 삼각함수로 변환
ptrain$month_sin <- sin(2 * pi * ptrain$month / 12)
ptrain$month_cos <- cos(2 * pi * ptrain$month / 12)

# 시각 변수를 삼각함수로 변환
ptrain$hour_sin <- sin(2 * pi * ptrain$time / 24)
ptrain$hour_cos <- cos(2 * pi * ptrain$time / 24)

# 데이터 프레임으로 변환
ptrain <- as.data.frame(ptrain)
# lm 모델 학습
ts_lm <- lm(ts ~ wind_x + wind_y + re + ta + sun10 + hm +
              month_sin + month_cos + hour_sin + hour_cos + stn_id, 
            data = ptrain)
# 결측치를 포함한 행 필터링
ptrain_na <- ptrain %>% filter(is.na(ts))
predictors <- c("wind_x", "wind_y", "re", "ta", "sun10", "hm",
                "month_sin", "month_cos", "hour_sin", "hour_cos", "stn_id")
X_na <- ptrain_na[, predictors]

# 결측치 예측
predicted_values <- predict(ts_lm, newdata = X_na)
# 예측된 값을 결측치에 대치
ptrain$ts[is.na(ptrain$ts)] <- predicted_values

# 필요한 열만 선택하여 새로운 데이터프레임 생성
selected_columns <- c("stn_id", "year", "month", "day", "time", "minute",
                      "FirstLetter", "SecondLetter", "ws10_deg", "ws10_ms",
                      "ta", "re", "hm", "sun10", "ts", "vis1", "class")
train_selected <- train %>% select(all_of(selected_columns))
################### ################### ################### ################### 
train_selected %<>%
  mutate(MDay = MDtoD(year, month, day), TM = time * 60 + minute) %>%
  mutate(cosMD = cos(MDay * 2 * pi / 365), sinMD = sin(MDay * 2 * pi / 365), 
         cosTM = cos(TM * 2 * pi / 1440), sinTM = sin(TM * 2 * pi /1440))
################### ################### ################### ################### 
write.csv(train_selected, file = "../data/train_final.csv", row.names = FALSE)
################### ################### ################### ################### 

################### 2. test ts <- ta 
# lm 모델 학습
ts_lm2 <- lm(ts ~ wind_x + wind_y + re + ta + sun10 + hm +month_sin 
             + month_cos + hour_sin + hour_cos + FirstLetter, data = ptrain)
test = fread('../data/test_prepro.csv')
test %<>%
  mutate(FirstLetter = as.factor(FirstLetter),
         re = as.factor(re),
         wind_x = ws10_ms * cos(ws10_deg * pi / 180),
         wind_y = ws10_ms * sin(ws10_deg * pi / 180),
         month_sin = sin(2 * pi * month / 12),
         month_cos = cos(2 * pi * month / 12),
         hour_sin = sin(2 * pi * time / 24),
         hour_cos =cos(2 * pi * time / 24))

# 결측치를 포함한 행 필터링
test_ts_na <- test %>% filter(is.na(ts))
predictors <- c("wind_x", "wind_y", "re", "ta", "sun10", "hm","month_sin",
                "month_cos", "hour_sin", "hour_cos", "FirstLetter")
# 예측할 새로운 데이터셋 생성
X_na <- test_ts_na %>% select(all_of(predictors))
# 결측치 예측
predicted_values <- predict(ts_lm2, newdata = X_na)
# 예측된 값을 결측치에 대치
test$ts[is.na(test$ts)] <- predicted_values
# write
selected_columns <- c("stn_id", "year", "month", "day", "time", "minute",
                      "FirstLetter", "SecondLetter", "ws10_deg", "ws10_ms",
                      "ta", "re", "hm", "sun10", "ts", "class")

test_selected <- test %>% select(all_of(selected_columns))
################### ################### ################### ################### 
test_selected %<>%
  mutate(MDay = MDtoD(year, month, day), TM = time * 60 + minute) %>%
  mutate(cosMD = cos(MDay * 2 * pi / 365), sinMD = sin(MDay * 2 * pi / 365), 
         cosTM = cos(TM * 2 * pi / 1440), sinTM = sin(TM * 2 * pi /1440))
################### ################### ################### ################### 
write.csv(test_selected, file = "../data/test_final.csv", row.names = FALSE)
################### ################### ################### ################### 