import pandas as pd
import numpy as np
import tweepy
from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import appCredentials 

## TWITTER API AUTHENTICATION ##

class Twitter_Authenticator():
    def get_authentication(self):
        auth = OAuthHandler(appCredentials.CONSUMER_KEY,appCredentials.CONSUMER_SECRET)
        auth.set_access_token(appCredentials.ACCESS_TOKEN,appCredentials.ACCESS_TOKEN_SECRET)
        return auth 

## TWEET LISTENER AND WRITE TO FILE ##

class Twitter_Listener(StreamListener):
    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename,'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on data: %s" % str(e))
        return True
    
    def on_error(self, status):
        ##terminate on bad call
        if status==420:
            return False
        print(status)

## TWEET STREAMER ##

class Twitter_Streamer():
    def __init__(self):
        self.twitter_auth = Twitter_Authenticator()
    def stream_tweet(self, fetched_tweets_filename, hash_tag_list):
        listener = Twitter_Listener(fetched_tweets_filename)
        auth = self.twitter_auth.get_authentication()
        stream = Stream(auth, listener)
        stream.filter(track=hash_tag_list)

## GET CLIENT TWEET DATA ##

class Twitter_Client():
    def __init__(self,twitter_user=None):
        self.auth = Twitter_Authenticator().get_authentication()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user
    
    def client_api(self):
        return self.twitter_client

    def get_user_tweets(self,num_tweets):
        tweets=[]
        for tweet in Cursor(self.twitter_client.user_timeline,id = self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets
    def get_user_friends(self,num_friends):
        friends=[]
        for friend in Cursor(self.twitter_client.friends,id = self.twitter_user).items(num_friends):
            friends.append(friend)
        return friends
    def get_home_timeline(self,num_tweets):
        tweets=[]
        for tweet in Cursor(self.twitter_client.home_timeline,id = self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweet
