library(dplyr)
library(corrplot)
library(ggplot2)
library(plotly)
library(data.table)


setwd(getwd())
######################### EDA3 ######################### 
##ta,sun10,ts EDA codes

df=fread('../data/train_prepro.csv')
df=as.data.frame(df)
df$class=as.factor(df$class)
colnames(df)


df_model=df %>% select(year,month,day,time,minute,FirstLetter,ws10_ms,ta,re,hm,sun10,ts,class)



##class 결측 제거
df<- df %>% filter(!is.na(class))

##Binary로 변수 추가
df <- df %>% 
  mutate(binary = case_when(
    class %in% c(1,2,3)~'Y',
    class %in% c(4)~'N'
  ))
df$binary=as.factor(df$binary)


##계절 추가
df <- df %>%
  mutate(season = case_when(
    month %in% c(3, 4, 5) ~ "Spring",
    month %in% c(6, 7, 8) ~ "Summer",
    month %in% c(9, 10, 11) ~ "Fall",
    month %in% c(12, 1, 2) ~ "Winter"
  ))
df$season<-as.factor(df$season)


##상,하반기 추가
df <- df %>% mutate(half=case_when(
  month %in% c(1,2,3,4,5,6)~'상반기',
  month %in% c(7,8,9,10,11,12)~'하반기'))
df$half=as.factor(df$half)

unique(df$month)



##TA:기온

######density plot
ggplot(df, aes(x = ta)) +
  geom_histogram(aes(y = ..density..), binwidth = 0.5, fill = "lightblue", color = "black", alpha = 0.6) +
  geom_density(color = "blue", fill = "blue", alpha = 0.2) +
  labs(title = "density of ta",
       x = "ta",
       y = "Density") +
  theme_minimal()


##season별
ggplot(df, aes(x = ta, fill = season)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of ta by season",
       x = "New Column",
       y = "Density") +
  theme_minimal()

##상,하반기 별->상,하반기로 plot이 잘 나뉘어짐
ggplot(df, aes(x = ta, fill = half)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of ta by half",
       x = "New Column",
       y = "Density") +
  theme_minimal()


##class별
ggplot(df, aes(x = ta, fill = class)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of ta by class",
       x = "New Column",
       y = "Density") +
  theme_minimal()


##안개발생 여부- 안개가 발생한 case의 분포를 보면 발생하지 않았을 때와 비교하여 0-10도 부근에서 밀도가 높음
ggplot(df, aes(x = ta, fill = binary)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of ta by fog occurence",
       x = "New Column",
       y = "Density") +
  theme_minimal()

##Violin plot by class

#안개가 발생하는 경우의 온도의 분포를 보면 안개가 발생하지 않는 경우보다 상대적으로 0도에서 10도 사이의 밀도가 높음
ggplot(df, aes(x = class, y = ta, fill =class)) +
  geom_violin(alpha = 0.6) +
  labs(title = "Violin Plot of ta by Class",
       x = "Class",
       y = "ta",
       fill = "Class") +
  theme_minimal()

#안개발생 여부
ggplot(df, aes(x = binary, y = ta, fill =binary)) +
  geom_violin(alpha = 0.6) +
  labs(title = "Violin Plot of ta by fog occurence",
       x = "fog",
       y = "ta",
       fill = "fog") +
  theme_minimal()


#지역별 차이-> 좀 있어보이긴함
ggplot(df, aes(x = FirstLetter, y = ta, fill =FirstLetter)) +
  geom_violin(alpha = 0.6) +
  labs(title = "Violin Plot of ta by region",
       x = "region",
       y = "ta",
       fill = "region") +
  theme_minimal()





##Sun10

##안개 발생 여부와 상관없이 상당히 치우쳐진 분포임
##분포를 보면 왼쪽으로 치우져진 분포형태,
ggplot(df, aes(x = sun10)) +
  geom_histogram(aes(y = ..density..), binwidth = 0.01, fill = "lightblue", color = "black", alpha = 0.6) +
  geom_density(color = "blue", fill = "blue", alpha = 0.2) +
  labs(title = "density of sun10",
       x = "sun10",
       y = "Density") +
  theme_minimal()


##안개발생 여부에 따른 violin plot
ggplot(df, aes(x = binary, y = sun10, fill =binary)) +
  geom_violin(alpha = 0.6) +
  labs(title = "Violin Plot of sun10 by fog occurence",
       x = "fog",
       y = "sun10",
       fill = "fog") +
  theme_minimal()




##ts


######density plot
ggplot(df, aes(x = ts)) +
  geom_histogram(aes(y = ..density..), binwidth = 0.5, fill = "lightblue", color = "black", alpha = 0.6) +
  geom_density(color = "blue", fill = "blue", alpha = 0.2) +
  labs(title = "density of ts",
       x = "ts",
       y = "Density") +
  theme_minimal()


##season별
ggplot(df, aes(x = ts, fill = season)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of ts by season",
       x = "New Column",
       y = "Density") +
  theme_minimal()

##상,하반기 별
ggplot(df, aes(x = ts, fill = half)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of ts by half",
       x = "New Column",
       y = "Density") +
  theme_minimal()


##class별
ggplot(df, aes(x = ts, fill = class)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of ts by class",
       x = "New Column",
       y = "Density") +
  theme_minimal()


##안개발생 여부별
ggplot(df, aes(x = ts, fill = binary)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of ts by fog occurence",
       x = "ts",
       y = "Density") +
  theme_minimal()


##Violin plot by class

ggplot(df, aes(x = class, y = ts, fill =class)) +
  geom_violin(alpha = 0.6) +
  labs(title = "Violin Plot of ts by Class",
       x = "Class",
       y = "ts",
       fill = "Class") +
  theme_minimal()

#안개발생 여부-> 안개가 발생하는 경우,지면 온도 0-20도 사이의 밀도가 발생하지 않는 경우보다 상대적으로 높음
ggplot(df, aes(x = binary, y = ts, fill =binary)) +
  geom_violin(alpha = 0.6) +
  labs(title = "Violin Plot of ts by fog occurence",
       x = "fog",
       y = "ts",
       fill = "fog") +
  theme_minimal()


#지역별 차이
ggplot(df, aes(x = FirstLetter, y = ts, fill =FirstLetter)) +
  geom_violin(alpha = 0.6) +
  labs(title = "Violin Plot of ts by region",
       x = "region",
       y = "ts",
       fill = "region") +
  theme_minimal()





##시계열적인 특성
df$class=as.factor(df$class)
df$year=as.factor(df$year)
df$month=as.factor(df$month)
df$day=as.factor(df$day)
df$time=as.factor(df$time)
df$minute=as.factor(df$minute)
df=df[complete.cases(df),]

S=split(df,df$stn_id)


vari=S$AA$sun10


layout(matrix(c(1,1,2,3),2,2,byrow = T))
plot.ts(vari) ; title('ts')
acf(vari,lag=1008)
pacf(vari,lag=144)

##모든 수치형 변수들이 time과 month에 대한 계절성을 지님
