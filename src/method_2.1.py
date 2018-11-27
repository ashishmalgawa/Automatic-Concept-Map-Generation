############################################################
#Authors        : Ashish, Hussain, Abhishek, Smit
#Last_Modified  : 7-4-2018 
#Version        : 2
#Usage          : python startTest.py [input_file_path]
############################################################


from subprocess import check_output
import os
import nltk
import spotlight
import pysolr
from io import open
from scipy import spatial #for cosine similarity
import networkx as nx
import sys
from datetime import datetime

#---GLOBAL Variables starts
SIMILARITY_THRESHOLD=-1
SIMILARITY_THRESHOLD2=0
CURRENT_PATH=os.getcwd()
#---GLOBAL Variables ends
# Author: Abhishek-----Starts----
def loader(f_name):
    print "Loading of data start\n"
    file = open(f_name, 'r',encoding="utf-8")
    lines = file.read().lower()
    print "Loading of data stop\n"

    return lines


def annotate(lines):
    
    print "annotate start"
    l_dict = dict()

   # print len(lines)

    for line in lines:
        ann_list = list()
        try:
            annotations = spotlight.annotate('http://model.dbpedia-spotlight.org/en/annotate', line)
        except:
          #  print "No annotations found for: ", line
          #  print "\n"
            continue
        for ann in annotations:
        #    print ann['surfaceForm']
            ann_list.append(ann['surfaceForm'].lower())
       # print "\n"
        l_dict[line[:-1]] = ann_list

    print "annotate stop"
    return l_dict

# Author: Abhishek-----Ends----

# break
#----Author : Ashish start-------
def getWordVector(word):
    glove_solr = pysolr.Solr('http://localhost:8983/solr/glove', timeout=10)
    searchQuery = 'id:' + word
    results = glove_solr.search(searchQuery, rows=10)
    vector=[]
    for result in results:
        vector = map(float,result["vector"].split(" "))
    return vector

def convertWordToVectors(sentencedict):
    wordVectorDict={}
    for sentence in sentencedict.keys():
        wordVectors = []
        for word in sentencedict[sentence]:
            #for multi word expression
            multi_word_list=word.split(" ")
            multiWordVectors=[]
            for w in multi_word_list:
                tempWordVector=getWordVector(w)
                if(len(tempWordVector)>0): #if word is available in glove
                    multiWordVectors.append(tempWordVector)
            if len(multiWordVectors) == 0:
                continue #if none of the words are available in glove from the multiword expression
            wordVector = [sum(x) for x in zip(*multiWordVectors)]
            wordVector = map(lambda x: x / len(multiWordVectors), wordVector)
            wordVectors.append((word,wordVector))
        wordVectorDict[sentence]=wordVectors
    return wordVectorDict

def getMultiWordVector(word):
    multi_word_list=word.split(" ")
    multiWordVectors=[]
    for w in multi_word_list:
        tempWordVector=getWordVector(w)
        if(len(tempWordVector)>0): #if word is available in glove
            multiWordVectors.append(tempWordVector)
    wordVector = [sum(x) for x in zip(*multiWordVectors)]
    wordVector = map(lambda x: x / len(multiWordVectors), wordVector)
    return wordVector

def relationSimilarity(openIEDict,annotatedDict):
    similarityDict={}
    for sentence in openIEDict.keys():
        similarityList=[]
        for wordPairs in openIEDict[sentence]:
            #print "smit",wordPairs
            # word 1
            flag1 = 0
            flag2 = 0
            
            word1=wordPairs[0]
            word2=wordPairs[2]
            
            max_similarity1 = SIMILARITY_THRESHOLD2 #starts with similarity threshold
            max_similarity2 = SIMILARITY_THRESHOLD2 #starts with similarity threshold
            
            for word in annotatedDict[sentence]:
                #for 1st word
                flg = 0
                try:
                    cosine_similarity = 1 - spatial.distance.cosine(getMultiWordVector(word), getMultiWordVector(wordPairs[0]))
                except:
                    # continue
                    flg = 1
                if(flg==0):
                    if cosine_similarity > max_similarity1:
                        word1=word
                        flag1 = 1
                        max_similarity1=cosine_similarity
            
            #for 2nd word
                try:
                    cosine_similarity = 1 - spatial.distance.cosine(getMultiWordVector(word), getMultiWordVector(wordPairs[2]))
                except:
                    continue
                
                if cosine_similarity > max_similarity2:
                    word2=word
                    flag2 = 1
                    max_similarity2=cosine_similarity
             
            if flag1 == 0 and flag2 == 0:
                continue #not in any of the annotated words
            multi_word_list=wordPairs[0].split(" ")
            multiWordVectors=[]
            for w in multi_word_list:
                tempWordVector=getWordVector(w)
                if(len(tempWordVector)>0): #if word is available in glove
                    multiWordVectors.append(tempWordVector)
            if len(multiWordVectors) == 0:
                continue #if none of the words are available in glove from the multiword expression
            wordVector1 = [sum(x) for x in zip(*multiWordVectors)]
            wordVector1 = map(lambda x: x / len(multiWordVectors), wordVector1)
            # word2
            multi_word_list=wordPairs[1].split(" ")
            multiWordVectors=[]
            for w in multi_word_list:
                tempWordVector=getWordVector(w)
                if(len(tempWordVector)>0): #if word is available in glove
                    multiWordVectors.append(tempWordVector)
            if len(multiWordVectors) == 0:
                continue #if none of the words are available in glove from the multiword expression
            wordVector2 = [sum(x) for x in zip(*multiWordVectors)]
            wordVector2 = map(lambda x: x / len(multiWordVectors), wordVector2)
            
            cosine_similarity = 1 - spatial.distance.cosine(wordVector1, wordVector2)
            if cosine_similarity > SIMILARITY_THRESHOLD:
                similarityList.append(([word1,wordPairs[1],word2],cosine_similarity))
        similarityDict[sentence]=similarityList
    return similarityDict
                    

def calculateSimilarity(wordVectorDict):
    similarityDict={}
    for sentence in wordVectorDict.keys():
        wordVectors=wordVectorDict[sentence]
        similarityList=[]
        for i in range(len(wordVectors)):
            for j in xrange(i+1,len(wordVectors)):
                cosine_similarity = 1 - spatial.distance.cosine(wordVectors[i][1], wordVectors[j][1])
                if cosine_similarity > SIMILARITY_THRESHOLD:
                    similarityList.append((wordVectors[i][0],wordVectors[j][0],cosine_similarity))
        similarityDict[sentence]=similarityList
    return similarityDict

def outputFilePath():
    if not os.path.exists(CURRENT_PATH+"/../output/Results_v2.1"):
        os.mkdir(CURRENT_PATH+"/../output/Results_v2.1")
    print CURRENT_PATH+"/../output/Results_v2.1/graph.gexf"
    return CURRENT_PATH+"/../output/Results_v2.1/graph.gexf"
#----Author : Ashish ends-------

#----Author : Hussain start-------
def checkContain(string1,string2):
    list1=string1.split(" ")
    list2=string2.split(" ")
    
    flag=True
    index=-1
    for item in list1:
        if item in list2 and list2.index(item)>index :
            index=list2.index(item)
        else:
            flag=False
            break
    return flag

def callOpenIE(lines):
    print "openie start"
    linesList1=nltk.sent_tokenize(lines)
    linesList=[]
    for line in linesList1:
        linesList.append(line[:-1])
    
    #print "linesList",linesList
    os.chdir('../data/stanford-corenlp-full-2018-02-27')

    finalRelationList=[]
    sent_relation_triples_dict={}
    TEMPFILEPATH="../../temInput.txt"
    
    filey=open(TEMPFILEPATH,"w")
    filey.write(lines)
    filey.close()

    cmdOpenIE="java -mx4g -cp \"*\" edu.stanford.nlp.naturalli.OpenIE "+TEMPFILEPATH
    ouputString=check_output(cmdOpenIE,shell=True)

    outputLines=ouputString.split("\n")
    
    for opLines in outputLines:
        opClauses=opLines.split("\t")
      #  print(opClauses)
        if len(opClauses)!=4:
            continue
        confidence=opClauses[0]
        subject=opClauses[1]
        predicate=opClauses[2]
        obJect=opClauses[3]
        
        for line in linesList:
            if checkContain(subject,line) and checkContain(obJect,line) and checkContain(predicate,line):
                tempList=[]
                tempList.append(subject)
                tempList.append(predicate)
                tempList.append(obJect)
                if line not in sent_relation_triples_dict:
                    sent_relation_triples_dict[line]=[]
                    sent_relation_triples_dict[line].append(tempList)
                else:
                    sent_relation_triples_dict[line].append(tempList)
    
    print "openie stop"
    return sent_relation_triples_dict

    
#----Author : Hussain end---------


def main():
    lines = loader(sys.argv[1])
    
    startTime = datetime.now()
    sent_relation_triples_dict = callOpenIE(lines)
    aaa =  str(datetime.now() - startTime )
    h, m, s = [float(i) for i in aaa.split(':')]
    time = 3600*h + 60*m + s
    
    print "open time",time
    
    startTime = datetime.now()
    annotatedDict = annotate(nltk.sent_tokenize(lines))
    aaa =  str(datetime.now() - startTime )
    h, m, s = [float(i) for i in aaa.split(':')]
    time = 3600*h + 60*m + s
    
    print "annotate time",time
    similarityDict = relationSimilarity(sent_relation_triples_dict,annotatedDict)
    G = nx.Graph()
    startTime = datetime.now()
    for key in similarityDict:
        list_rel = similarityDict[key]
        for l in list_rel:
            relation = l[0]
            relation_score = l[1]
            G.add_edge(relation[0],relation[2],hase=relation[1])
    
    nx.write_gexf(G, outputFilePath())
    aaa =  str(datetime.now() - startTime )
    h, m, s = [float(i) for i in aaa.split(':')]
    time = 3600*h + 60*m + s
    
    print "graph generation time",time

if __name__=="__main__":
    main()
