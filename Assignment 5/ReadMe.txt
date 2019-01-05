ReadMe:

This script allows a user to cluster tweets based upon their similarity. It is assumed that the user provides the following files:
        seeds: txt
            Initial cluster centers {tweet ids}
        list of tweets: json 
            tweets that are to be clustered
        output: txt
            Cluster id and the list of tweets in each cluster will be written into it
        
This script requires that `json`, `sys`, `random` and `numpy` be installed within the Python environment you are running this script in.

The script can be executed as follows
	python kmeans_tweets.py [number_of_clusters] <initial_seed_file_path> <tweets_file_path> <output_file_path>