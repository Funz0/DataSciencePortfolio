# Alejandro Cepeda
# LIS 4805 Final Project Pt 2
# Song Genre Classification

# outside package used:
citation(package = "funModeling")

# source of outside code: 
# audio feature description for music_data kable: https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features
# caret package model validation: https://remiller1450.github.io/s230f19/caret3.html 

# set working directory
setwd("~/Documents/GitHub/Spotify_Genre_Classification")

# libraries
library(tidyverse)
library(caret)
library("funModeling")
library(ggpubr)
library(corrplot)

##### Data Prep #####
# read in data
music_data <- read.csv("SpotifyFeatures.csv", stringsAsFactors=F)
str(music_data)
head(music_data)
summary(music_data)
# storing dataset description
music_data.type <- lapply(music_data, class)
music_data.var_desc <- c("Song Genre",
                         "Song Artist",
                         "Song Name",
                         "Song unique ID",
                         "Song Popularity (0-100) where higher is better",
                         "A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.",
                         "Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.",
                         "Duration of song in milliseconds",
                         "Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.",
                         "Predicts whether a track contains no vocals. \"Ooh\" and \"aah\" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly \"vocal\". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.",
                         "The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C<U+266F>/D<U+266D>, 2 = D, and so on. If no key was detected, the value is -1.",
                         "Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.",
                         "The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.",
                         "Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.",
                         "Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.",
                         "The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.",
                         "A measurement used in music to indicate meter, written as a fraction with the bottom number indicating the kind of note used as a unit of time and the top number indicating the number of units in each measure.",
                         "A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).")
music_data.var_names <- colnames(music_data)
data.desc <- as_tibble(cbind(music_data.var_names,music_data.type,music_data.var_desc))
colnames(data.desc) <- c("Variable","Class","Description")
kable(data.desc)

# check for missing/NA values 
colSums(is.na(music_data))
music_data <- music_data[!duplicated(music_data$track_id),]

# drop unnecessary cols
spotify_raw <- music_data[,-c(2:4)]

# convert characters to factors
spotify_clean <- spotify_raw %>% 
  mutate(key = factor(spotify_raw$key),
         time_signature = factor(spotify_raw$time_signature),
         mode = factor(spotify_raw$mode))
# drop old genre variable
#spotify_clean <- spotify_clean[,-1]

# find which genres to keep
#spotify_clean %>% 
#  count(track_genre) %>%
#  kable()

# storing genre levels to keep
#genres <- list("Blues","Classical","Country","Disco","Hip-Hop","Jazz","Metal","Electronic","Reggae","Reggaeton")
# drop unwanted genres from data
#spotify_clean <- spotify_clean[spotify_clean$track_genre %in% genres,]
#spotify_clean$track_genre <- droplevels(spotify_clean$track_genre)

# display clean dataset
knitr::kable(head(spotify_clean), caption="Clean Spotify Song Dataset")

##### Data Analysis #####
# storing audio features for analysis 
spotify_feat <- spotify_clean %>% 
  select(where(is.numeric)) 

summary(spotify_feat)

# plot number of songs per selected genre
#spotify_clean %>% 
#  ggplot(aes(x=track_genre, fill=track_genre))+
#  geom_bar()+
#  coord_polar()+
#  theme_void()

# density plot of numeric features
plot(spotify_feat)
# plot factor features
key_bp <- ggplot(spotify_clean, aes(x=key)) +
  geom_bar(aes(fill=key)) +
  theme(axis.text.x=element_blank())
mode_bp <- ggplot(spotify_clean, aes(x=mode)) +
  geom_bar(aes(fill=mode)) +
  theme(axis.text.x=element_blank())
time_sig_bp <- ggplot(spotify_clean, aes(x=time_signature)) +
  geom_bar(aes(fill=time_signature)) +
  theme(axis.text.x=element_blank())

# display all factor features
ggarrange(key_bp, mode_bp, time_sig_bp,
          widths=c(0.5,0.5), heights=c(0.5,0.5))

# correlation plot of feature selection
feat_corr <- cor(spotify_feat)
corrplot(feat_corr,
         type="upper",
         diag=FALSE,
         tl.srt=45,
         tl.col="black")
# remove acousticness (highest negative correlation)
spotify_clean$acousticness <- NULL
# dropping the following variables (not valuable for model)
spotify_clean$mode <- NULL
spotify_clean$time_signature <- NULL

##### Train/Test split #####
# creating index and creating train/test split
set.seed(123)
index <- createDataPartition(spotify_clean$track_genre, p=0.60, list=FALSE)
spotify_train <- spotify_clean[index,]
spotify_test <- spotify_clean[-index,]

# training control for models
ctrl <- trainControl(method="cv", number=10, verboseIter=FALSE)

##### Random Forest #####
library(ranger)
set.seed(123)
# random forest model
rf_model <- ranger(track_genre~.,
                   spotify_train,
                   importance="impurity",
                   verbose=0)
rf_model
# storing best tunegrid params
tunegrid_rf <- expand.grid(mtry=rf_model$mtry, splitrule=rf_model$splitrule, min.node.size=rf_model$min.node.size)
# train rf model w 10 fold cv
rf_train <- train(track_genre~., 
                  spotify_train, 
                  method="ranger", 
                  tuneGrid=tunegrid_rf, 
                  trControl=ctrl)
print(rf_train)
# training confusion matrix
confusionMatrix(rf_train$finalModel$predictions, spotify_train$track_genre)

# compute accuracy on test set
rf_pred <- predict(rf_train, newdata=spotify_test)
summary(rf_pred)
# test pred confusion matrix
rf_cm <- confusionMatrix(rf_pred, spotify_test$track_genre)
rf_cm
# RF used a lot of memory (~300MB)

# plot results with "vip" package
library(vip)
(rf_varImp<- vip(rf_model, geom="point", horizontal=FALSE,
              aes=list(color="blue", shape=17, size=5)) +
              theme_light())

##### Gradient Boosting #####
library(gbm)
set.seed(123)
# training gb model with cv
gbm_model <- train(track_genre~., 
                   spotify_train,
                   method="gbm",
                   trControl=ctrl,
                   verbose=0)
print(gbm_model)
# train set predictions
gbm_train_pred <- predict(gbm_model)
gbm_train_result <- data.frame(spotify_train$track_genre, gbm_train_pred)
# training confusion matrix
gbm_train_cm <- confusionMatrix(spotify_train$track_genre, as.factor(gbm_train_pred))
print(gbm_train_cm)
# test set predictions
gbm_pred <- predict(gbm_model, newdata=spotify_test)
gbm_result <- data.frame(spotify_test$track_genre, gbm_pred)
# computing confusion matrix on test preds
gbm_cm <- confusionMatrix(spotify_test$track_genre, as.factor(gbm_pred))
print(gbm_cm)
# less memory usage (~20MB) but worse acc

# gbm variable importance
(gbm_varImp <- vip(gbm_model, geom="point", horizontal=FALSE,
                 aes=list(color="blue", shape=17, size=5)) +
                 theme_light())

##### Gradient Boosting v2 #####
# finding best tuning params
params <- gbm_model$bestTune

# training new gbm model with best tuning params
gbm_model2 <- train(track_genre~., 
                    spotify_train,
                    method="gbm",
                    tuneGrid=params,
                    trControl=ctrl,
                    verbose=0)
print(gbm_model2)
# train set predictions
gbm_train_pred <- predict(gbm_model2)
gbm_train_result <- data.frame(spotify_train$track_genre, gbm_train_pred)
# training confusion matrix
gbm_train_cm <- confusionMatrix(spotify_train$track_genre, as.factor(gbm_train_pred))
print(gbm_train_cm)
# test set predictions
gbm_pred <- predict(gbm_model2, newdata=spotify_test)
gbm_result <- data.frame(spotify_test$track_genre, gbm_pred)
# computing confusion matrix on test preds
gbm2_cm <- confusionMatrix(spotify_test$track_genre, as.factor(gbm_pred))
print(gbm2_cm)

# gbm variable importance
(gbm2_varImp <- vip(gbm_model, geom="point", horizontal=FALSE,
                   aes=list(color="blue", shape=17, size=5)) +
    theme_light())

##### Model Result Comparison #####
# test variable importance by model
ggarrange(rf_varImp, gbm_varImp, gbm2_varImp,
          widths=c(0.5,0.5,0.5), heights=c(0.5,0.5,0.5), nrow=3)
# test confusion matrix by model
kable(rf_cm$table)
kable(gbm_cm$table)
kable(gbm2_cm$table) 