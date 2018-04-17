import tensorflow as tf
import numpy as np
import sys

source_file_name='/home/cdpai/ibm-icd/SimilarTicket_UseCase/doc2vec_demo/training_set.2.csv.tfrecord.bin'

# Attempt 3

"""
	Had To upgrade to 1.7.1 from 1.2.1 to use this API
	This Code Snippet Reads TFRecord File and creates batches using tf.data.Dataset api.
"""


def parse_one_example(example_proto):
	features = {
		"sentence1":tf.VarLenFeature(tf.string),
		"sentence2":tf.VarLenFeature(tf.string),
		"label":tf.VarLenFeature(tf.string)
	}
	parsed_feature = tf.parse_single_example(example_proto,features=features)
	return [parsed_feature['sentence1'],parsed_feature["sentence2"],parsed_feature['label']]

def parse_examples(examples):
	features = {
		"sentence1":tf.VarLenFeature(tf.string),
		"sentence2":tf.VarLenFeature(tf.string),
		"label":tf.VarLenFeature(tf.string)
	}
	parsed_example = tf.parse_example(examples,features=features)
	return [parsed_example['sentence1'].values,parsed_example['sentence2'].values,parsed_example['label'].values]

dataset = tf.data.TFRecordDataset(source_file_name)

iterator = dataset.shuffle(100).batch(10).map(parse_examples).make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
	#print(sess.run(iterator.initializer))
	for i in range(1):
		tensor_sent1,tensor_sent2,tensor_label=sess.run(next_element)
		for i in zip(tensor_sent1,tensor_sent2,tensor_label):
			print(i)

sys.exit(1)


"""
	TFRecord is simple record serializer , to read the data we can load the whole dataset in memory and 
	then shuffle etc else we need to seperate it as small files and then read that small files and shuffle and feed [TODO]
"""

# Attempt 2 - Success Reads one record at a time

record_iterator = tf.python_io.tf_record_iterator(path=source_file_name)

for string_record in record_iterator:
	example = tf.train.Example()
	example.ParseFromString(string_record)
	#print(dir(example.features.feature['sentence1']))
	print(example.features.feature['sentence1'].bytes_list.value[0])
	print(example.features.feature['sentence2'].bytes_list.value[0])
	print(example.features.feature['label'].float_list.value[0])
	break

sys.exit(1)

# Attempt 1  - Will parse One Example Untested.

with tf.Session() as sess:
	features = {
		"sentence_1":tf.VarLenFeature(tf.string),
		"sentence_2":tf.VarLenFeature(tf.string),
		"label":tf.FixedLenFeature([],tf.float32)
	}

	filename_queue = tf.train.string_input_producer([source_file_name], num_epochs=1)

	reader = tf.TFRecordReader()
	
	_,serialized_example = reader.read(filename_queue)

	features = tf.parse_single_example(serialized_example, features=features)
	
	sentence_1 = features['sentence_1']
	
	print(sess.run(sentence_1))

