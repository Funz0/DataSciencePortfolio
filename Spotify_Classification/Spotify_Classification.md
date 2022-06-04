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

Before proceeding any further, inspecting the genres within the dataset
may be helpful in the case of a misleading genre being present.

Success! We can now continue with our cleaning process.

As the final step of the cleaning process, let’s check for any missing
values:

Great news, 0 NAs were found!

## Exploratory Data Analysis

Now that our data is clean, we can conduct some EDA and determine
whether any further manipulation to be made on the data as well as
selecting the most appropriate features for the models.

### Numerical Features

To ensure the data possesses proper center and spread, let’s take a look
at each possible features by genre. First let’s visualize the
distribution of each possible feature across all genres in the dataset:

Based on the density plots above, `duration_min`, `instrumentalness`,
`liveness`, `loudness`, `popularity`, and `speechiness` require further
normalization.

**Track Duration**

**Instrumentalness**

After the analysis above, the option I believe to be most useful to the
model performance is to remove this feature altogether as there are
quite a large number of tracks with 0 `instrumentalness` detected. I
also stored the values within the data with no instruments detected and
will determine later if the 57791 tracks are worth dropping to have
`instrumentalness` as a predictor variable for the models. This will be
determined during the [correlation](#correlation) step.

**Liveness**

**Loudness**

**Popularity**

**Speechiness**

### Categorical Features

**Key**

**Mode**

**Time Signature**

#### One-Hot Encoding

### Correlation

…

## Feature Engineering

### Train/Test Split

## Modeling

### Nearest Neighbors

### Random Forest

### Gradient Boosting

### Model Comparison
