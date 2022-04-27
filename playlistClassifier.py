import spotipy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


from spotipy.oauth2 import SpotifyClientCredentials

#Spotify Login Information
client_credentials_manager = SpotifyClientCredentials(client_id='ea9244872cc2404cbd8dc55a89745d8f', client_secret='0841567385b24f2dbaf6c93a663556cc')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


#Getting my top played music of 2020
good_playlist = sp.user_playlist("1182050600", "37i9dQZF1EMcyDariIUh4m")
good_tracks = good_playlist["tracks"]
good_songs = good_tracks["items"]

while good_tracks['next']:
    good_tracks = sp.next(good_tracks)
    for item in good_tracks["items"]:
        good_songs.append(item)
good_ids = [] 
for i in range(len(good_songs)):
    good_ids.append(good_songs[i]['track']['id'])
    features = []
for i in range(0,len(good_ids),50):
    audio_features = sp.audio_features(good_ids[i:i+50])
    for track in audio_features:
        features.append(track)
        features[-1]['target'] = 1

#Getting a playlist I dont like from Spotify Techno/House Music
bad_playlist = sp.user_playlist("1182050600", "2MSE9BQC2U1i3U4NNltxOw")
bad_tracks = bad_playlist["tracks"]
bad_songs = bad_tracks["items"]

while bad_tracks['next']:
    bad_tracks = sp.next(bad_tracks)
    for item in bad_tracks["items"]:
        bad_songs.append(item)
bad_ids = [] 
for i in range(len(bad_songs)):
    bad_ids.append(bad_songs[i]['track']['id'])
for i in range(0,len(bad_ids),50):
    audio_features = sp.audio_features(bad_ids[i:i+50])
    for track in audio_features:
        features.append(track)
        features[-1]['target'] = 0

trainingData = pd.DataFrame(features)
trainingData.head()

train, test = train_test_split(trainingData, test_size = 0.25)
print("Training size: {}, Test size: {}".format(len(train),len(test)))

features = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness","duration_ms"]

#Split the data into x and y test and train sets passing them into different classifiers
x_train = train[features]
y_train = train["target"]

x_test = test[features]
y_test = test["target"]


#Decision Tree
c = DecisionTreeClassifier(min_samples_split=100)
dt = c.fit(x_train, y_train)
y_pred = c.predict(x_test)
score = accuracy_score(y_test, y_pred) * 100
print("Accuracy using Decision Tree: ", round(score, 1), "%")

#Nearest Neighbour
knn = KNeighborsClassifier(3)
knn.fit(x_train, y_train)
knn_pred = c.predict(x_test)
score = accuracy_score(y_test, knn_pred) * 100
print("Accuracy using Knn Tree: ", round(score, 1), "%")

#Neural MLP 
mlp = MLPClassifier()
mlp.fit(x_train, y_train)
mlp_pred = mlp.predict(x_test)
score = accuracy_score(y_test, mlp_pred) * 100
print("Accuracy using MLP : ", round(score, 1), "%")

#Random Forest
forest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
forest.fit(x_train, y_train)
forest_pred = forest.predict(x_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, forest_pred) * 100
print("Accuracy using random forest: ", round(score, 1), "%")

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, max_depth=1, random_state=0)
gbc.fit(x_train, y_train)
predicted = gbc.predict(x_test)
score = accuracy_score(y_test, predicted)*100
print("Accuracy using Gbc: ", round(score, 1), "%")


#weird techno music : 2MSE9BQC2U1i3U4NNltxOw
#old country music : 37i9dQZF1DWYP5PUsVbso9
#my favourite music : 37i9dQZF1EMcyDariIUh4m
#Using a classifier to search through a new playlist for songs I would like  
playlistToFindSongsYouLikeIn = sp.user_playlist("1182050600", "37i9dQZF1EMcyDariIUh4m")

newPlaylist_tracks = playlistToFindSongsYouLikeIn["tracks"]
newPlaylist_songs = newPlaylist_tracks["items"] 
while newPlaylist_tracks['next']:
    newPlaylist_tracks = sp.next(newPlaylist_tracks)
    for song in newPlaylist_tracks["items"]:
        newPlaylist_songs.append(song)
        
newPlaylist_song_ids = [] 
print(len(newPlaylist_songs))
for i in range(len(newPlaylist_songs)):
    newPlaylist_song_ids.append(newPlaylist_songs[i]['track']['id'])
    
newPlaylist_features = []
j = 0
for i in range(0,len(newPlaylist_song_ids),50):
    audio_features = sp.audio_features(newPlaylist_song_ids[i:i+50])
    for track in audio_features:
        track['song_title'] = newPlaylist_songs[j]['track']['name']
        track['artist'] = newPlaylist_songs[j]['track']['artists'][0]['name']
        j= j + 1
        newPlaylist_features.append(track)
print(len(newPlaylist_features))

playlistToLookAtFeatures = pd.DataFrame(newPlaylist_features)

pred = knn.predict(playlistToLookAtFeatures[features])

likedSongs = 0
i = 0
for prediction in pred:
    if(prediction == 1):
        print (playlistToLookAtFeatures["song_title"][i] + ", By: "+ playlistToLookAtFeatures["artist"][i])
      #  sp.user_playlist_add_tracks("1287242681", "7eIX1zvtpZR3M3rYFVA7DF", [test['id'][i]])
        likedSongs= likedSongs + 1
    i = i +1
print("")
print("You should like ",'{:.2f}%'.format((likedSongs/i)*100),"% of that playlist !")
print("")