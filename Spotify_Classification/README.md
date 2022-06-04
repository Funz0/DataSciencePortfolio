Spotify Predictive Analysis
================
Alejandro Cepeda
5/23/2022

***GRAMMAR CHECK ALL MARKDOWN TEXT BEFORE UPLOADING***

## Context

![](spotify-logo.png)

Music (listening and playing) is one of my favorites pastimes and go-to
therapy session to de-stress. From a very young age, I have always
enjoyed diverse genres of music, which then prompted me to search for
and discover new genres I may enjoy. That of course could be a hit or
miss depending on what I discover. To combat the struggle of manually
conducting searches for new or previously unknown genres of music, I ask
the following question: **What determines a track’s genre category?**.
Knowing this may help me determine what makes a track enjoyable or not
based on my own tastes and hopefully others who read this! To answer
this, I downloaded a Spotify dataset from
[Kaggle.com](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db),
which was gathered using the Spotify API, mined and saved by Kaggle user
Zaheen Hamidani. This dataset contains track information and the audio
features associated with their respective track, sourced from Spotify’s
API
[documentation](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features):

## Data Description

-   **genre**: Song genre
-   **artist_name**: Song artist
-   **track_name**: Song name
-   **track_id**: Song unique ID
-   **popularity**: Song popularity (0-100) where higher is better
-   **acousticness**: A confidence measure from 0.0 to 1.0 of whether
    the track is acoustic. 1.0 represents high confidence the track is
    acoustic.
-   **danceability**: Describes how suitable a track is for dancing
    based on a combination of musical elements including tempo, rhythm
    stability, beat strength, and overall regularity. A value of 0.0 is
    least danceable and 1.0 is most danceable.
-   **duration_ms**: Duration of track in milliseconds
-   **energy**: Energy is a measure from 0.0 to 1.0 and represents a
    perceptual measure of intensity and activity. Typically, energetic
    tracks feel fast, loud, and noisy. For example, death metal has high
    energy, while a Bach prelude scores low on the scale. Perceptual
    features contributing to this attribute include dynamic range,
    perceived loudness, timbre, onset rate, and general entropy.
-   **instrumentalness**: Predicts whether a track contains no vocals.
    “Ooh” and “aah” sounds are treated as instrumental in this context.
    Rap or spoken word tracks are clearly “vocal”. The closer the
    instrumentalness value is to 1.0, the greater likelihood the track
    contains no vocal content. Values above 0.5 are intended to
    represent instrumental tracks, but confidence is higher as the value
    approaches 1.0.
-   **key**: The estimated overall key of the track. Integers map to
    pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C#/D#,
    2 = D, and so on. If no key was detected, the value is -1.
-   **liveness**: Detects the presence of an audience in the recording.
    Higher liveness values represent an increased probability that the
    track was performed live. A value above 0.8 provides strong
    likelihood that the track is live.
-   **loudness**: The overall loudness of a track in decibels (dB).
    Loudness values are averaged across the entire track and are useful
    for comparing relative loudness of tracks. Loudness is the quality
    of a sound that is the primary psychological correlate of physical
    strength (amplitude). Values typical range between -60 and 0 db.
-   **mode**: Mode indicates the modality (major or minor) of a track,
    the type of scale from which its melodic content is derived. Major
    is represented by 1 and minor is 0.
-   **speechiness**: Speechiness detects the presence of spoken words in
    a track. The more exclusively speech-like the recording (e.g. talk
    show, audio book, poetry), the closer to 1.0 the attribute value.
    Values above 0.66 describe tracks that are probably made entirely of
    spoken words. Values between 0.33 and 0.66 describe tracks that may
    contain both music and speech, either in sections or layered,
    including such cases as rap music. Values below 0.33 most likely
    represent music and other non-speech-like tracks.
-   **tempo**: The overall estimated tempo of a track in beats per
    minute (BPM). In musical terminology, tempo is the speed or pace of
    a given piece and derives directly from the average beat duration.
-   **time_signature**: A measurement used in music to indicate meter,
    written as a fraction with the bottom number indicating the kind of
    note used as a unit of time and the top number indicating the number
    of units in each measure.
-   **valence**: A measure from 0.0 to 1.0 describing the musical
    positiveness conveyed by a track. Tracks with high valence sound
    more positive (e.g. happy, cheerful, euphoric), while tracks with
    low valence sound more negative (e.g. sad, depressed, angry).

## Packages and Data Collection

The following are the packages used throughout this project:

``` r
# libraries
library(tidyverse)
library(corrplot)

library(caret)
```

Let’s read in the dataset from the working directory:

``` r
# read in data
spotify <- read_csv("SpotifyFeatures.csv", show_col_types = FALSE)
tibble(glimpse(spotify))
```

    ## Rows: 232,725
    ## Columns: 18
    ## $ genre            <chr> "Movie", "Movie", "Movie", "Movie", "Movie", "Movie",~
    ## $ artist_name      <chr> "Henri Salvador", "Martin & les fées", "Joseph Willia~
    ## $ track_name       <chr> "C'est beau de faire un Show", "Perdu d'avance (par G~
    ## $ track_id         <chr> "0BRjO6ga9RKCKjfDqeFgWV", "0BjC1NfoEOOusryehmNudP", "~
    ## $ popularity       <dbl> 0, 1, 3, 0, 4, 0, 2, 15, 0, 10, 0, 2, 4, 3, 0, 0, 0, ~
    ## $ acousticness     <dbl> 0.61100, 0.24600, 0.95200, 0.70300, 0.95000, 0.74900,~
    ## $ danceability     <dbl> 0.389, 0.590, 0.663, 0.240, 0.331, 0.578, 0.703, 0.41~
    ## $ duration_ms      <dbl> 99373, 137373, 170267, 152427, 82625, 160627, 212293,~
    ## $ energy           <dbl> 0.9100, 0.7370, 0.1310, 0.3260, 0.2250, 0.0948, 0.270~
    ## $ instrumentalness <dbl> 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.23e-01, 0.0~
    ## $ key              <chr> "C#", "F#", "C", "C#", "F", "C#", "C#", "F#", "C", "G~
    ## $ liveness         <dbl> 0.3460, 0.1510, 0.1030, 0.0985, 0.2020, 0.1070, 0.105~
    ## $ loudness         <dbl> -1.828, -5.559, -13.879, -12.178, -21.150, -14.970, -~
    ## $ mode             <chr> "Major", "Minor", "Minor", "Major", "Major", "Major",~
    ## $ speechiness      <dbl> 0.0525, 0.0868, 0.0362, 0.0395, 0.0456, 0.1430, 0.953~
    ## $ tempo            <dbl> 166.969, 174.003, 99.488, 171.758, 140.576, 87.479, 8~
    ## $ time_signature   <chr> "4/4", "4/4", "5/4", "4/4", "4/4", "4/4", "4/4", "4/4~
    ## $ valence          <dbl> 0.8140, 0.8160, 0.3680, 0.2270, 0.3900, 0.3580, 0.533~

    ## # A tibble: 232,725 x 18
    ##    genre artist_name    track_name track_id popularity acousticness danceability
    ##    <chr> <chr>          <chr>      <chr>         <dbl>        <dbl>        <dbl>
    ##  1 Movie Henri Salvador C'est bea~ 0BRjO6g~          0      0.611          0.389
    ##  2 Movie Martin & les ~ Perdu d'a~ 0BjC1Nf~          1      0.246          0.59 
    ##  3 Movie Joseph Willia~ Don't Let~ 0CoSDzo~          3      0.952          0.663
    ##  4 Movie Henri Salvador Dis-moi M~ 0Gc6TVm~          0      0.703          0.24 
    ##  5 Movie Fabien Nataf   Ouverture  0IuslXp~          4      0.95           0.331
    ##  6 Movie Henri Salvador Le petit ~ 0Mf1jKa~          0      0.749          0.578
    ##  7 Movie Martin & les ~ Premières~ 0NUiKYR~          2      0.344          0.703
    ##  8 Movie Laura Mayne    Let Me Le~ 0PbIF9Y~         15      0.939          0.416
    ##  9 Movie Chorus         Helka      0ST6uPf~          0      0.00104        0.734
    ## 10 Movie Le Club des J~ Les bisou~ 0VSqZ3K~         10      0.319          0.598
    ## # ... with 232,715 more rows, and 11 more variables: duration_ms <dbl>,
    ## #   energy <dbl>, instrumentalness <dbl>, key <chr>, liveness <dbl>,
    ## #   loudness <dbl>, mode <chr>, speechiness <dbl>, tempo <dbl>,
    ## #   time_signature <chr>, valence <dbl>

## Data Preparation

Prior to conducting EDA, there is already some preexisting knowledge
regarding the variables within the dataset. Since this data was gathered
using an API, it is always a good practice to check for duplicate
observations (or rows) to ensure uniformity. The variables
`artist_name`, `track_name`, and `track_id` will not be needed for the
purpose of this project, thus the best course of action may be to remove
them. Additionally, there are variables such as `genre`, `key`, `mode`,
and `time_signature` that may be best treated as **multi-level**
variables, meaning that each observation of the aforementioned variables
have defined categories throughout the dataset. To conduct all of these
data cleaning steps, I created a pipeline using the dplyr’s `%>%`
operator (loaded from the tidyverse package).

``` r
# pipeline to clean, drop and reformat variables
spotify_clean <- spotify %>%
  # remove duplicate tracks
  distinct(track_id, .keep_all=TRUE) %>%
  # select relevant variables
  select(-c(artist_name, track_name, track_id)) %>%
  # convert character variables to factor
  mutate(genre = as_factor(genre),
         key = as_factor(key),
         mode = as_factor(mode),
         time_signature = as_factor(time_signature),
         # convert duration to minutes instead
         duration_min = duration_ms/60000) %>%
  select(-duration_ms)
```

Before proceeding any further, inspecting the genres within the dataset
may be helpful in the case of a misleading genre being present.

``` r
# ensure genres are properly written
tibble(levels(spotify_clean$genre))
```

    ## # A tibble: 27 x 1
    ##    `levels(spotify_clean$genre)`
    ##    <chr>                        
    ##  1 Movie                        
    ##  2 R&B                          
    ##  3 A Capella                    
    ##  4 Alternative                  
    ##  5 Country                      
    ##  6 Dance                        
    ##  7 Electronic                   
    ##  8 Anime                        
    ##  9 Folk                         
    ## 10 Blues                        
    ## # ... with 17 more rows

``` r
# recode Children's Music genres as one
levels(spotify_clean$genre)[levels(spotify_clean$genre) == "Children’s Music"] <- "Children's Music"

# verify recoding of levels
tibble(levels(spotify_clean$genre))
```

    ## # A tibble: 26 x 1
    ##    `levels(spotify_clean$genre)`
    ##    <chr>                        
    ##  1 Movie                        
    ##  2 R&B                          
    ##  3 A Capella                    
    ##  4 Alternative                  
    ##  5 Country                      
    ##  6 Dance                        
    ##  7 Electronic                   
    ##  8 Anime                        
    ##  9 Folk                         
    ## 10 Blues                        
    ## # ... with 16 more rows

Success! We can now continue with our cleaning process.

As the final step of the cleaning process, let’s check for any missing
values:

``` r
# check for missing data
colSums(is.na(spotify_clean))
```

    ##            genre       popularity     acousticness     danceability 
    ##                0                0                0                0 
    ##           energy instrumentalness              key         liveness 
    ##                0                0                0                0 
    ##         loudness             mode      speechiness            tempo 
    ##                0                0                0                0 
    ##   time_signature          valence     duration_min 
    ##                0                0                0

Great news, 0 NAs were found!

## Exploratory Data Analysis

Now that our data is clean, we can conduct some EDA and determine
whether any further manipulation to be made on the data as well as
selecting the most appropriate features for the models.

``` r
# number of tracks per genre
spotify_clean %>%
  group_by(genre) %>%
  count()
```

    ## # A tibble: 26 x 2
    ## # Groups:   genre [26]
    ##    genre           n
    ##    <fct>       <int>
    ##  1 Movie        7802
    ##  2 R&B          5353
    ##  3 A Capella     119
    ##  4 Alternative  9095
    ##  5 Country      7383
    ##  6 Dance        7982
    ##  7 Electronic   9149
    ##  8 Anime        8935
    ##  9 Folk         8048
    ## 10 Blues        8496
    ## # ... with 16 more rows

``` r
# reorder variables
spotify_clean <- spotify_clean %>%
  # remove A Capella from data
  filter(!genre %in% "A Capella") %>%
  droplevels() %>%
  select(genre, key, mode, time_signature, everything())
```

### Numerical Features

To ensure the data possesses proper center and spread, let’s take a look
at each possible features by genre. First let’s visualize the
distribution of each possible feature across all genres in the dataset:

``` r
# store numeric variable names in a vector
feature_names <- names(spotify_clean)[5:15]

# density plot of numeric features by genre
spotify_clean %>%
  select(c(genre, feature_names)) %>%
  # convert data to long format based on features
  pivot_longer(cols = feature_names) %>%
  ggplot(aes(x = value, fill = genre)) +
  geom_density() +
  facet_wrap(~name, ncol = 3, scales = "free") +
  labs(title = "Spotify Audio Feature Density Across Genres",
       x = "", y = "density") +
  theme(axis.text.x = element_text(size = 8, angle = 40),
        axis.text.y = element_blank())
```

![](Spotify_Classification_files/figure-gfm/Numerical%20Features-1.jpeg)<!-- -->

Based on the density plots above, `duration_min`, `instrumentalness`,
`liveness`, `loudness`, `popularity`, and `speechiness` require
normalization to ensure a well distributed dataset.

**Track Duration**

``` r
# look for outliers
spotify_clean %>%
  ggplot(aes(y = duration_min)) +
  geom_boxplot() + 
  coord_flip() +
  ggtitle("Duration (in minutes)")
```

![](Spotify_Classification_files/figure-gfm/Track%20Duration-1.jpeg)<!-- -->

``` r
# store outliers based on 4th whisker
duration_outliers <- boxplot(spotify_clean$duration_min, 
                             plot = FALSE, range = 4)$out

# remove outliers
spotify_clean <- spotify_clean %>%
  filter(!duration_min %in% duration_outliers)

spotify_clean %>%
  ggplot(aes(y = duration_min)) +
  geom_boxplot() + 
  coord_flip() +
  ggtitle("Duration(in minutes) - no outliers")
```

![](Spotify_Classification_files/figure-gfm/Track%20Duration-2.jpeg)<!-- -->

**Instrumentalness**

``` r
# find duration outliers
spotify_clean %>%
  ggplot(aes(y = instrumentalness)) +
  geom_boxplot() +
  coord_flip() +
  ggtitle("Instrumentalness")
```

![](Spotify_Classification_files/figure-gfm/Instrumentalness-1.jpeg)<!-- -->

``` r
# determine number of genres with no instrumentalness
no_instrument <- spotify_clean %>%
  summarize(sum(instrumentalness == 0))

# look at genres with 0 instrumentalness
spotify_clean %>%
  ggplot(aes(x=log(instrumentalness), fill=genre)) +
  geom_density()
```

![](Spotify_Classification_files/figure-gfm/Instrumentalness-2.jpeg)<!-- -->

After the analysis above, the option I believe to be most useful to the
model performance is to remove this feature altogether as there are
quite a large number of tracks with 0 `instrumentalness` detected. I
also stored the values within the data with no instruments detected and
will determine later if the 57791 tracks are worth dropping to have
`instrumentalness` as a predictor variable for the models. This will be
determined during the [correlation](#correlation) step.

``` r
# drop instrumentalness
spotify_clean <- spotify_clean %>%
  select(-instrumentalness)
```

**Liveness**

**Loudness**

**Popularity**

**Speechiness**

``` r
# find speechiness outliers
spotify_clean %>%
  ggplot(aes(y = speechiness)) +
  geom_boxplot(coef = 4) +
  coord_flip() +
  labs(title = "Speechiness")
```

![](Spotify_Classification_files/figure-gfm/Speechiness-1.jpeg)<!-- -->

``` r
# store outliers within duration_min
speechiness_outliers <- boxplot(spotify_clean$speechiness, 
                             plot = FALSE, range = 4)$out
# remove outliers
spotify_clean <- spotify_clean %>%
  filter(!speechiness %in% speechiness_outliers)

# verify outliers are removed
spotify_clean %>%
  ggplot(aes(y = speechiness)) +
  geom_boxplot(coef = 4) +
  coord_flip() +
  labs(title = "Speechiness, less outliers")
```

![](Spotify_Classification_files/figure-gfm/Speechiness-2.jpeg)<!-- -->

### Categorical Features

``` r
# categorical variable distribution
genre_bar <- ggplot(spotify_clean)
# key
genre_bar + geom_bar(aes(key)) 
```

![](Spotify_Classification_files/figure-gfm/Categorical%20Features-1.jpeg)<!-- -->

``` r
# mode
genre_bar + geom_bar(aes(mode))
```

![](Spotify_Classification_files/figure-gfm/Categorical%20Features-2.jpeg)<!-- -->

``` r
# time_signature
genre_bar + geom_bar(aes(time_signature))
```

![](Spotify_Classification_files/figure-gfm/Categorical%20Features-3.jpeg)<!-- -->

**Key**

**Mode**

**Time Signature**

``` r
# drop mode and time signature (too biased/vague)
spotify_clean <- spotify_clean %>%
  select(-c(mode, time_signature))
```

#### One-Hot Encoding

### Correlation

…

``` r
# correlation plot of numerical features (add categorical once encoding is done)
spotify_clean %>%
  select(feature_names) %>%
  cor() %>%
  corrplot(method="color", type="upper",
           tl.col = "black", tl.cex=0.9, tl.srt=45)
```

``` r
# remove loudness (highest positive & 2nd highest negative corr)
spotify_clean$loudness <- NULL
```

## Feature Engineering

### Train/Test Split

``` r
# creating index and creating train/test split
set.seed(123)
index <- createDataPartition(spotify_final$genre, p=0.70, list=FALSE)
spotify_train <- spotify_final[index,]
spotify_test <- spotify_final[-index,]

# training control for models
ctrl <- trainControl(method="cv", number=10, verboseIter=FALSE)
```

## Modeling

### Nearest Neighbors

### Random Forest

``` r
set.seed(123)
# random forest model
rf_model <- ranger(genre~.,
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
(rf_varImp<- vip(rf_model, geom="point", horizontal=FALSE,
              aes=list(color="blue", shape=17, size=5)) +
              theme_light())
```

### Gradient Boosting

``` r
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
```

### Model Comparison

``` r
# test variable importance by model
ggarrange(rf_varImp, gbm_varImp, gbm2_varImp,
          widths=c(0.5,0.5,0.5), heights=c(0.5,0.5,0.5), nrow=3)
# test confusion matrix by model
kable(rf_cm$table)
kable(gbm_cm$table)
kable(gbm2_cm$table)
```
