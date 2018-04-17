""" 
	This code demonstrate converting a training file containing list of sentences into tfrecords 
	Each Line contains following
	sentence_1,sentence_2,label

	label tells us whether two sentences are close and how close they are
"""
import tensorflow as tf
import numpy as np
import sys

source_file_name='/home/cdpai/ibm-icd/SimilarTicket_UseCase/doc2vec_demo/training_set.2.csv'

lines = open(source_file_name).readlines()

preprocessed_lines = [x.replace('\n','') for x in lines ]

splitted_lines = np.array([(x.split('SARWESH')[0].encode('utf-8'),x.split('SARWESH')[1].encode('utf-8'),float(x.split('SARWESH')[2])) for x in preprocessed_lines ])

# for label

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# _bytes is used for string/char values

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

target_file_name=source_file_name+'.tfrecord.bin'

writer = tf.python_io.TFRecordWriter(target_file_name)

# Attempt 2  - Each Line is a seperate example

for sent1,sent2,label in splitted_lines:
	#print("{} {} {}".format(sent1,sent2,label))
	features = {
		'sentence1': _bytes_feature(sent1),
		'sentence2': _bytes_feature(sent2),
		'label': _bytes_feature(label) #_float_feature(float(label))
	}
	example = tf.train.Example(features=tf.train.Features(feature=features))
	writer.write(example.SerializeToString())


writer.close()

sys.exit(1)

# Attempt 1 - Wrong, all records serialized as one example, creates issues while reading


features = {
	'sentence1': _bytes_feature(splitted_lines[:,0]),
	'sentence2': _bytes_feature(splitted_lines[:,1]),
	'label': _float_feature(splitted_lines[:,2].astype(float))
}

example = tf.train.Example(features=tf.train.Features(feature=features))

writer.write(example.SerializeToString())

writer.close()
