import boto
import pandas as pd

class S3Conn(object):
    '''
    Class to handle S3 connections
    
    Requires credentials in ~/.boto:

        [profile yournamehere]
        aws_access_key_id = a_key
        aws_secret_access_key = a_secret

        [profile anothernamehere]
        aws_access_key_id = another_key
        aws_secret_access_key = another_secret

        [Credentials]
        aws_access_key_id = default_key
        aws_secret_access_key = default_secret
    '''
    def __init__(self, bucket_name='my-data-bucket', profile_name='adam'):
        self.bucket_name = bucket_name
        self.profile_name = profile_name
        self.s3path = None
        self.conn = boto.connect_s3(profile_name=profile_name)
        
        try:
            self.bucket = self.conn.get_bucket(self.bucket_name)
        except:
            print "Bucket {} not found.".format(bucket_name)
    
    def ls_keys(self, contains=''):
        '''
        Print all keys containing the string 'contains'
        '''
        for key in self.bucket.list():
            if contains in key.name:
                print key.name.encode('utf-8')
    
    def csv_to_df(self, s3path):
        '''
        Returns a csv file on s3 as a pandas dataframe.
        '''
        self.s3path = s3path
        mykey = self.bucket.get_key(self.s3path)
    
        if mykey is None:
            print 's3path {} does not exist. Returning None.'.format(s3path)
            return None
    
        # generate a temporary signed url to access the data
        myurl = mykey.generate_url(600, query_auth=True)
        
        # TODO: Allow for reading only part of the file
        df = pd.read_csv(myurl)
    
        return df
    
def demo():
    s3path = 'path/to/data.csv'
    s3 = S3Conn()
    df = s3.csv_to_df(s3path)
    print df.head()