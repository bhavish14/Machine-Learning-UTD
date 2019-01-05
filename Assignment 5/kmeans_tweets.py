'''
    This script allows a user to cluster tweets based upon their similarity. 
    It is assumed that the user provides the following files:
        seeds: txt
            Initial cluster centers {tweet ids}
        list of tweets: json 
            tweets that are to be clustered
        output: txt
            Cluster id and the list of tweets in each cluster will be written into it
        
    This script requires that `json`, `sys`, `random` and `numpy` be installed within the Python environment you 
    are running this script in.

'''


import json
import sys
import random
import numpy as np


class tweetsCluster:    
    '''
        A class used to cluster the tweets according to the similarity measure between them.
    
        ...
        
        Attributes
        ----------
        num_clusters: int
            number of clusters to be used in the algorithm (default = 25)
        seeds: list int
            stores the initial centroids of the clusters. 
        tweets: dict {int, str}
            stores the id and the tweets
        tweets_keys: list int
            stores the ids of the tweets
        cluster_distance: list int
            stores the distance of each point from each of the cluster
        final_cluster: list id
            stores the ids{tweets} of the cluster centers 
    
    
        Methods
        -------
        tokenize_tweet(id)
            Given the id of a tweets, it returns the tokenized version of it.
        compute_jaccard(id1, id2)
            Given a pair of tweets, it computes the distance between them using the formula: 
            1 - |A intersection B| / |A union B|.
        find_distance():
            Computes the distance between each of the tweets to each of the clusters and stores.
            it in the cluster_distance.
        assign_to_clusters():
            assigns the tweets to each of the clusters based on the distance. 
        recompute_centroid():
            Recomputes the centeroid of each of the clusters post the assignment step. 
        compute_sse():
            Computes the SSE of the model
        kmeans_cluster():
            Initializes the K-means parameters and invokes the other functions for clustering. 
    '''
    
    def __init__(self, initial_seed_path, tweets_data_path, output_path, number_of_clusters = 25):
        '''
            Parameters
            ----------
            initial_seed_path: str
                absolute path of the seeds file.
            tweets_data_path: str
                absolute path of the tweets.json file. 
            output_path: str
                absoulte path of the expected output.txt file. 
            number_of_clusters: int, optional
                number of clusters to be used in the model.
        '''
        
        
        self.num_clusters = int(number_of_clusters)
        self.seeds = []
        self.clusters = []
        self.tweets = {}

        src_handle = open(initial_seed_path, 'r')

        # Seeds List
        for item in src_handle:
            self.seeds.append(int(item.split(',')[0]))

        # Tweets List
        for item in open(tweets_data_path, 'r'):
            temp = json.loads(item)
            self.tweets[temp['id']] =  temp['text']

        self.tweet_keys = list(self.tweets.keys())
        self.output = open(output_path, 'w')

        # x = cluster; y = tweet
        self.cluster_distance = []
        self.final_cluster = {}


    def tokenize_tweet(self, id):
        '''
            Given the id of a tweets, it returns the tokenized version of it.
        
            Parameters
            ----------
            id: int
                id of the tweet

            Returns
            -------
            list
                A list of words of the tweet.
        '''
        return (self.tweets[id].split())

    def compute_jaccard(self, id1, id2):
        '''
            Given a pair of tweets, it computes the distance between them using the formula: 
            1 - |A intersection B| / |A union B|.
        
            Parameters
            ----------
            id1: int
                id of the first tweet
            id2: int
                id of the second tweet
        

            Returns
            -------
            distance: int
                distance between the tweet and the cluster
        '''
       
       
        tweet1 = set(self.tokenize_tweet(id1))
        tweet2 = set(self.tokenize_tweet(id2))

        distance = 1 - (len(tweet1 & tweet2) / len(tweet1 | tweet2))
        return distance

    def find_distance(self):
        '''
            Computes the distance between each of the tweets to each of the clusters and stores.
            it in the cluster_distance.
        '''
        distance_matrix = []
        for index in range(len(self.tweet_keys)):
            tweet_distance = []
            for cluster_id in self.clusters:
                tweet_distance.append(
                    self.compute_jaccard(cluster_id, self.tweet_keys[index])
                )
            distance_matrix.append(tweet_distance)

        self.cluster_distance = np.array(distance_matrix)


    def assign_to_clusters(self):
        '''
            assigns the tweets to each of the clusters based on the distance. 
        '''
        for index, item in enumerate(self.cluster_distance):
            c = np.unravel_index(np.argmin(item, axis = 0), item.shape)
            if c[0] in self.final_cluster:
                t = self.final_cluster[c[0]]
                t.append(self.tweet_keys[index])
                self.final_cluster[c[0]] = t
            else:
                self.final_cluster[c[0]] = [self.tweet_keys[index]]

    def recompute_centroids(self):
        '''
            Recomputes the centeroid of each of the clusters post the assignment step. 
        '''
        new_clusters = []
        for item in sorted(self.final_cluster):
            cluster = self.clusters[item]
            nodes = self.final_cluster[item]
            distance = []
            for node1 in nodes:
                distance_sum = 0
                temp_nodes = set(nodes)- set([node1])
                for node2 in temp_nodes:
                    distance_sum += self.compute_jaccard(node1, node2)
                distance.append(distance_sum)
            distance = np.array(distance)
            index = np.unravel_index(np.argmin(distance, axis = 0), distance.shape)[0]
            new_clusters.append(
                nodes[index]
            )
        self.clustes = new_clusters

    def compute_sse(self):
        '''
            Computes the SSE of the model
        
            Returns
            -------
            sse: int
                The sum of squared errors of the model
        '''
        sse = 0
        for item in sorted(self.final_cluster):
            cluster = self.clusters[item]
            nodes = self.final_cluster[item]
            for u in nodes:
                distance = self.compute_jaccard(cluster, u)
                sse += pow(distance, 2)
        return sse

    def kmeans_cluster(self):
        '''
            Initializes the K-means parameters and invokes the other functions for clustering. 
        '''
        for item in self.seeds:
            self.clusters.append(int(item))

        self.find_distance()
        self.assign_to_clusters()
        self.recompute_centroids()
        error = self.compute_sse()
        self.output.write("SSE Error: %f \n\n" % (error))
        self.output.write("Cluster \t\t [tweets]")
        for index, item in enumerate(sorted(self.final_cluster)):
            nodes = str(index) + "\t\t"
            for x in self.final_cluster[item]:
                nodes = nodes + str(x) + ", "
            self.output.write(
                "\n%s\n" % (nodes)
            )
           


def main():
    number_of_clusters = sys.argv[1]
    initial_seed_path = sys.argv[2]
    tweets_data_path = sys.argv[3]
    output_path  = sys.argv[4]
    cluster_obj = tweetsCluster(
        initial_seed_path, tweets_data_path,
        output_path, number_of_clusters
    )

    cluster_obj.kmeans_cluster()

if __name__ == "__main__":
    main()



# python kmeans_tweets.py 25 InitialSeeds.txt Tweets.json output.txt
