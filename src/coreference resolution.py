import xml.etree.ElementTree as ET
import nltk
from subprocess import call
import os
import sys

input_file = sys.argv[1]
# Running the coreferencing tool
os.chdir('../data/stanford-corenlp-full-2018-02-27/')

cmd = "java -Xmx5g -cp stanford-corenlp-3.7.0.jar:stanford-english-corenlp-models-3.7.0.jar:* edu.stanford.nlp.pipeline.StanfordCoreNLP -props edu/stanford/nlp/coref/properties/deterministic-english.properties -file ../"+input_file
os.system(cmd)
os.chdir('../../src/')

# Parsing the xml file to remove coreferences.
tree = ET.parse('../data/stanford-corenlp-full-2018-02-27/' + input_file + '.xml')
root = tree.getroot() 

file = open('../data/'+ input_file,'r')
lines = file.readlines()

file_out = open('../data/processed_'+ input_file,'w')

# Function to create a dictionary where key is the main word and values contains a list of words which are coreferenced as the main word.
def create_dict():
	count = 0

	cor_dict = dict()

	# Parse the xml using tags and nesting
	for child in root: 									# tag = document
		for child1 in child:							# tag = coreference
			if child1.tag=='coreference':
				for child2 in child1:					# tag = coreference
					cor_list = list()
					count = 0
					for child3 in child2:				# tag = mention
						count+=1
						for child4 in child3:			# tag = text
							if child4.tag=='text':
								if(count==1):			# The first text is the main text hence skip it
									name = child4.text
									continue

								postag = nltk.pos_tag(nltk.word_tokenize(child4.text))	# Add only the pronouns
								if("PRP" in postag[0][1]):
									cor_list.append(child4.text)
								
					cor_dict[name] = cor_list			# Add the list to the dictionary
	
	return cor_dict

# Function to process input text file and create preprocesses text file
def process_input(cor_dict):
	global lines

	index_dict = dict()
	
	# Creating a dictionary where the main word is the key and values is the line number where the word appeared first.
	# This is done since we want to replace the word only in the places which appear after the word first appeared.
	for key,values in cor_dict.iteritems():
		count_lines = 0
		for line in lines:
			count_lines+=1
			if key in line:
				index_dict[key] = count_lines
				break

	# Create a new text file with coreference resolved.
	print index_dict
	count_keys = 0
	for key,values in cor_dict.iteritems():

		for value in values:
			lines1 = list()
			count_lines = 0
			for line in lines:	# For every main word we create a new list of lines and take the latest one to write to the file.
				count_lines+=1
				
				if(index_dict[key]>count_lines):# If the line count is less than the line at which the word appeared first then simply skip
					print index_dict[key],count_lines
					lines1.append(line)# Append the line as it is and then continue
					continue

				words = line.split()
				for i in range(len(words)):
					if(value==words[i]):	# At all the places the pronouns occur, repalce them with main word.
						words[i] = key

				processed_line = " ".join(words)
				
				lines1.append(processed_line)
				
			lines = lines1
		count_keys+=1

	# Finally write to the new file
	for line in lines:
		file_out.write(line)
		file_out.write("\n")

# Main function
def main():

	cor_dict = create_dict()
	process_input(cor_dict)
	print cor_dict

main()
