# -*- coding: utf-8
#############################
######## Text2Predic8 #######
#############################
import csv
import os
from os import path
from posixpath import join
import pprint
import re
import json
from typing import Hashable
from nltk.tokenize import sent_tokenize
from nltk.tree import *
from functions import *
import nltk
from stanfordcorenlp import StanfordCoreNLP

# from pycorenlp import StanfordCoreNLP
# nltk.download('punkt')
#english_vocab = set(w.lower() for w in nltk.corpus.words.words())
# java_path = "C:\Program Files\Java\jdk1.8.0_171\bin\java.exe"
# os.environ['JAVAHOME'] = java_path

vbse = False  # VerBoSE mode for error reporting and tracing

localBranch = ''
basePath = "./data/"
# basePath = "./data/msr/"
# basePath = "/Users/clarkhu/Downloads/fyp/data/"
# localBranch = "covid/"
localPath = basePath  # localPath + inPath
inPath = localPath + localBranch  # "test/"
outPath = localPath + localBranch

# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

# nlp = StanfordCoreNLP(r'/Users/clarkhu/Downloads/fyp/stanford-corenlp-4.2.2', lang='en')
nlp = StanfordCoreNLP('http://localhost', port=9000)
# import stanza
# from stanza.server import CoreNLPClient
# stanza.download('en')
# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos')
# pp = pprint.PrettyPrinter(indent=4)

global document_triples
global sentence_triples
global sentence_triplesPP
global set_of_raw_concepts

data = []
document_triples = []
causal_triples = []
causal_triples_v2 = []
causal_triples_list = []
sentence_triples = []
sentence_triplesPP = []

fullDocument = ""

global skip_over_previous_results
skip_over_previous_results = False

concept_tags = {'NN', 'NNS', 'PRP', 'PRP$'}
relation_tags = {'VB'}  # 动词基本形式 Verb, base form
illegal_concept_nodes = {
    "-RRB-", "-LRB-", "-RSB-" "-LSB-" "Unknown", "UNKNOWN", ",", ".", "?", "'s", "'", "''"}
# causal_words = '.ecause|cause.|unless|and|or|Therefore|as a result|consequently'
# causal_words_list = ['because', 'cause', 'caused', 'unless',
#                      'and', 'or', 'Therefore', 'as a result', 'consequently']
vb_tags_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def trim_concept_chain(text):  # very long chains only, phrases
    if vbse:
        print("   Trim c_c ", end="")
    "Extract nouns and Preps from coreferents that are entire phrases."
    str = nltk.word_tokenize(text.replace("_", " "))
    #print(" !!", str, end="!! ")
    tagged = nltk.pos_tag(str)
    ret = '_'.join([word for word, tag in tagged[:-1]
                   if tag in concept_tags] + [tagged[-1][0]])
    if vbse:
        print(" ->!!", ret, end="!! mirT ")
    return ret
# trim_concept_chain('cloth_captured_from_a_flapping_flag_it')


def processDocument(text):  # full document
    # print('text:',text)
    global sentence_number
    global sentence_triples
    global sentence_triplesPP
    global set_of_raw_concepts
    list_of_sentences = sent_tokenize(text)
    # print("********************")
    # print(list_of_sentences)
    # print("********************")
# with CoreNLPClient(
# annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'], timeout=30000,
# memory='16G') a
# s client:
    # ann = client.annotate(text)
    sents = sent_tokenize(text)
    # print('sents:',sents)
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize, ssplit, parse, dcoref',
        'outputFormat': 'json', })
    # print(type(nlp.annotate(text, properties={
    #     'annotators': 'tokenize, ssplit, parse, dcoref', 'outputFormat': 'json'})))
    # ff_test = json.dumps(output)
    # with open('output.txt', 'w') as f:
    #     f.write(ff_test)
    # print(type(output))
    if output == "CoreNLP request timed out. Your document may be too long.":
        print("***** Timeout of the Stanford Parser *****")
        list_of_sentences = []  # No parsed output to process
        pass

    if output == 'Could not handle incoming annotation':
        print("***** Could not handle incoming annotation *****")
        list_of_sentences = []  # No parsed output to process
        pass

    # elif isinstance(output, dict):
    #     try:
    #         coref = output['corefs'] ## OCCASIONALY DODGY RESULTS - ??? finds no corefs????
    #         # print('coref:',coref)
    #     except IndexError:
    #         coref = None
    #     # print('#####output',output)

    elif type(output) is str or type(output) is unicode:
        # print('type of output:',type(output))
        try:
            output = json.loads(output, strict=False)
            coref = output['corefs']
        except:
            coref = None

    else:
        print("** Stanford Parser Error - type:", type(output), end="")

    for i in range(len(list_of_sentences)):
        # for each sentence
        sentence_triples = []
        sentence_triplesPP = []
        sentence_number += 1  # 0

        # print("\nSENT#", sentence_number, ":", sents[i].strip(), end="   ")
        if sents[i][0:14] == "CR Categories:":    # skip keyword list.
            break
        try:
            sent1 = output['sentences'][i]['parse']
        except IndexError:
            sent1 = None
        if sent1 is not None:
            try:
                sent2 = CoreferenceResolution(coref, sent1)
            except IndexError:
                sent2 = None

        # Read a bracketed tree string and return the resulting tree.
        tree = ParentedTree.fromstring(sent2)
        tree_leaves = [x.lower() for x in tree.leaves() if isinstance(x, str)]
        tree_subtrees = tree.subtrees()
        # sentenceOfword = nltk.tokenize.word_tokenize(list_of_sentences[i])
        # wordOftagged = nltk.pos_tag(sentenceOfword)
        # triggerWordList = findTriggerWords(wordOftagged)
        # causal_triples = findPreBack(triggerWordList)

        order_vcv_list = findTriggerWords_v2(tree_leaves, tree_subtrees)
        causal_triples = findPreBack_v2(order_vcv_list)

        # causal_triples = findTriggerWords_v3(tree_leaves,tree_subtrees)

        # causal_triples= findTriggerWords_v4(tree,tree_leaves)

        Positions = getListPos(tree)  # 返回树中 子树的位置，结果为列表,（0，1）
        # print('Positions:',Positions)

        Positions_depths = getOrderPos(Positions)
        # print('#######Positions_depths',Positions_depths)

        Positions_leaves = getLeafPos(tree)
        # print('Positions_leaves:',Positions_leaves)

        # print('**********Positions_leaves:',Positions_leaves,'************')

        # find the children of S
        # TODO implement new set of rule
        # locate all VP's in the sentence.
        posOfVP = findPosOfSpecificLabel(
            tree, "VP", Positions_depths, Positions_leaves)
        # print("***Position of VP**** ", posOfVP, "*****")

        posOfNN = findPosOfSpecificLabel(
            tree, "NN", Positions_depths, Positions_leaves)
        # print('posOfNN:',posOfNN)

        # NPs = list(tree.subtrees(filter=lambda x: x.label()=='NP'))
        # print('NPs:',NPs)

        # VPs_str = [" ".join(vp.leaves()) for vp in list(tree.subtrees(filter=lambda x: x.label()=='VP'))]
        # print('VPs_str:',VPs_str)

        # print('NPs:',NPs)
        # print("***Position of IN**** ", posOfIN, "*****")
        # print('tree:',tree)
        # for i in posOfIN:
        #     PinTree = PositionInTree(i, Positions_depths)
        #     print('PinTree:',PinTree)
        #     T_child = findChildNodes(PinTree, i, Positions_depths)
        #     print("child ", T_child)

        ####################
        ######  VP  ########
        ####################
        if posOfVP == None:
            if vbse:
                print("Gotcha no VP")
        else:
            for z in posOfVP:   # iterative over VP's
                Triple = []
                Verb = ""
                NextStep = True

                PosInTree = PositionInTree(z, Positions_depths)
                # print('PosInTree:',PosInTree)
                child = findChildNodes(PosInTree, z, Positions_depths)
                # print('child:',child)
                if vbse:
                    print("child ", child)
                for x in child:
                    if checkLabel(tree, PositionInTree(x, Positions_depths)) == "VP":
                        if vbse:
                            print("True VP1 ")
                        NextStep = False
                        # break out and stop working with this VP
                    else:
                        if vbse:
                            print("non-VP1 ", end="")
                ###########################
                # If next step still equals true then there is no VP child of the current VP and
                # we can procede to the next step.
                if NextStep:
                    VerbTree = child[0]
                    Verb = child[0]
                    Verb = findLeavesFromNode(PositionInTree(
                        Verb, Positions_depths), Positions_leaves)
                    Verb = checkLabelLeaf(tree, Verb)
                    if vbse:
                        print("Verb:", Verb)
                    Subject = "Unknown"

                    LeftSibling = findLeftSiblingCurrentLevel(
                        z, Positions_depths)
                    LeftSiblingPos = PositionInTree(
                        LeftSibling, Positions_depths)
                    # print(checkLabel(tree, LeftSiblingPos))
                    RunCheck = True
                    try:
                        LeftSiblingLabel = checkLabel(tree, LeftSiblingPos)
                        if vbse:
                            print("Try left-sibling ", end=" ")
                    except:
                        RunCheck = False
                        if vbse:
                            print("No left-sibling", end=" ")

                    if RunCheck and LeftSiblingLabel == "NP":
                        leaves = findLeavesFromNode(
                            LeftSiblingPos, Positions_leaves)
                        Subject = leaves[len(leaves) - 1]
                        # print(Subject)
                        Subject = checkLabelLeaf(tree, Subject)
                        if vbse:
                            print("true NP1, leavels",
                                  leaves, "Subject:", Subject)
                        # print(Subject)
                    else:
                        # If left sibling isnt a NP then check parent and its NP, repeat until you find NP.
                        CurrentVP = z  # this will change later to x or something when I loop
                        # through all of the VP's
                        cont = True
                        counter = 0  # why?

                        while cont == True and counter < 10:
                            counter += 1
                            # get parent of this VP
                            Parent = findParentNode(PositionInTree(
                                CurrentVP, Positions_depths), Positions_depths)
                            if vbse:
                                print("?????????????Parent",
                                      Parent, "???????????????")
                            # print(CurrentVP)
                            # print(Parent)
                            if vbse:
                                print("This is the parent node")
                            # now that we have parent, check its leftsibling
                            ParentLeftSibling = findLeftSiblingCurrentLevel(
                                Parent, Positions_depths)
                            # print(ParentLeftSibling)
                            # now check the label of the parents left sibling, if it is an NP then use the above code, if it is node then repeat the process
                            ParentLeftSiblingPOS = PositionInTree(
                                ParentLeftSibling, Positions_depths)
                            RunCheck = True
                            try:
                                ParentLeftSiblingPOSLabel = checkLabel(
                                    tree, ParentLeftSiblingPOS)
                            except:
                                RunCheck = False

                            if RunCheck and ParentLeftSiblingPOSLabel == "NP":
                                if vbse:
                                    print("trueNP2 ")
                                leaves = findLeavesFromNode(
                                    ParentLeftSiblingPOS, Positions_leaves)
                                if vbse:
                                    print("Leaves", leaves)
                                Subject = leaves[len(leaves) - 1]
                                if vbse:
                                    print("Subject", Subject)
                                Subject = checkLabelLeaf(tree, Subject)
                                if vbse:
                                    print("Subject", Subject)
                                cont = False
                                break
                            else:
                                CurrentVP = Parent

                    # now that I have the subject and Verb I should combine these together and create a double.
                    if Subject.count("_") >= 2:
                        #print(Subject,"->", end="")
                        Subject = trim_concept_chain(Subject)
                        #print(Subject,"   ", end="")
                    Triple.append(Subject)

                    Triple.append(Verb)
                    if vbse:
                        print("*************Partial Triple", Triple)

                    # now locate the OBJECT - if there is one.
                    Obj = "Unknown-Obj "
                    # reuse some of the code from previous rule to find closest NP on the right of the verb.
                    ListOfNP = findPosOfSpecificLabel(
                        tree, "NP", Positions_depths, Positions_leaves)
                    PosOfVerbTree = Positions.index(
                        Positions_depths[child[0][0]][child[0][1]])
                    if vbse:
                        print(PosOfVerbTree)
                        print(ListOfNP)
                    index = []
                    if ListOfNP:  # dod
                        for x in ListOfNP:
                            index.append(Positions.index(
                                Positions_depths[x[0]][x[1]]))
                    if vbse:
                        print(index)
                    closest = 0
                    currentDif = 100000
                    for y in index:
                        diff = y - PosOfVerbTree
                        if (diff > 0 and diff < currentDif):
                            currentDif = diff
                            closest = y

                    # check if closest has an NP child, if it does work from this node instead
                    loop = True
                    count = 0
                    currentNode = findPosInOrderList(
                        Positions[closest], Positions_depths)
                    if vbse:
                        print("currentNode", currentNode)
                    while loop and count < 10:
                        currentNodePOS = PositionInTree(
                            currentNode, Positions_depths)
                        currentNodeChild = findChildNodes(
                            currentNodePOS, currentNode, Positions_depths)
                        currentNodeChildTreePOS = PositionInTree(
                            currentNodeChild[0], Positions_depths)

                        if(currentNodeChildTreePOS in Positions_leaves):
                            loop = False
                            break
                        elif checkLabel(tree, currentNodeChildTreePOS) == "NP":
                            currentNode = currentNodeChild[0]
                        else:
                            leaves = findLeavesFromNode(
                                currentNodePOS, Positions_leaves)
                            Obj = checkLabelLeaf(tree, leaves[len(leaves)-1])
                            loop = False
                            break
                    if vbse:
                        print("Obj=", Obj, " ")

                    if Obj != ".":
                        if Obj.count("_") >= 2:         # trim coreference Phrases
                            #print(Obj,"=>", end="")
                            Obj = trim_concept_chain(Obj)
                            #print(Obj, end="   ")
                        Triple.append(Obj)
                        #print(" TRIPLE: ", end="")
                        #print(Triple, end="")
                        sentence_triples.append(Triple)  # end PosOfVP for loop

        ####################
        ######  PP  ########
        ####################

        PosOfPP = findPosOfSpecificLabel(
            tree, "PP", Positions_depths, Positions_leaves)
        if vbse:
            print("$$$$$$$ PosOfPP", PosOfPP, "$$$$$$$")
        #global sentence_triplesPP
        # sentence_triplesPP = []
        if PosOfPP is None:
            if vbse:
                print("No PP found")
        else:
            if vbse:
                print("posOfPP", PosOfPP)
            for z in PosOfPP:
                Triple = []
                Preposition = ""
                NextStep = True
                PosInTree = PositionInTree(z, Positions_depths)
                child = findChildNodes(PosInTree, z, Positions_depths)
                for x in child:
                    if checkLabel(tree, PositionInTree(x, Positions_depths)) == "PP":
                        if vbse:
                            print("CheckLabel is True")
                        NextStep = False
                    else:
                        if vbse:
                            print("CheckLabel is False")
                if NextStep:
                    Preposition = child[0]
                    Preposition = findLeavesFromNode(PositionInTree(
                        Preposition, Positions_depths), Positions_leaves)
                    if vbse:
                        print(tree)
                        print(Preposition)
                    if vbse:
                        print("Preposition index:", Preposition)

                    if vbse:
                        print("Preposition", Preposition)
                    if type(Preposition) == list:
                        if len(Preposition[0]) > 1:  # ERROR from here
                            Preposition = [Preposition[0]]
                    if vbse:
                        print("2 tree[Preposition]", tree[Preposition])

                    Preposition = checkLabelLeaf(tree, Preposition)

                    # find NP on the left
                    PosPPTree = Positions.index(
                        Positions_depths[child[0][0]][child[0][1]])
                    closest = 0
                    currentDif = -1000000
                    if vbse:
                        print("index", index)
                        print("PosPPTree", PosPPTree)

                    """if isinstance(index, int):  # ulgy hack
                        index = [index]
                        #print("is int")   """

                    try:
                        index
                    except NameError:  # no Verb
                        index = []

                    for y in index:   # position of V
                        diff = y - PosPPTree
                        if (diff < 0 and diff > currentDif):
                            currentDif = diff
                            closest = y
                    # now that you have closest NP get the children
                    leaves = findLeavesFromNode(
                        Positions[closest], Positions_leaves)

                    # add the right most leaf to the triple
                    leafLabel = checkLabelLeaf(tree, leaves[len(leaves) - 1])
                    Triple.append(leafLabel)
                    Triple.append(Preposition)

                    # now get NP on the right
                    closest = 0
                    currentDif = 100000
                    for y in index:
                        diff = y - PosPPTree
                        if (diff > 0 and diff < currentDif):
                            currentDif = diff
                            closest = y

                    # check if closet has an NP child, if it does work from child
                    leafLabel = "UNKNOWN"
                    loop = True
                    count = 0
                    if vbse:
                        print(" closest=", closest, "Positions",
                              Positions, len(Positions), end=" ")

                    if closest >= len(Positions):
                        closest = (len(Positions)-1)  # No NP
                    currentNode = findPosInOrderList(
                        Positions[closest], Positions_depths)
                    if vbse:
                        print("currentNode:", currentNode)

                    while (currentNode != None) and (loop and count < 10):         # Why 10? 10 attempts?
                        # check if child is a leaf node first
                        # ClosestPosInOrderList = findPosInOrderList(Positions[closest],Positions_depths)
                        # childOfClosest = findChildNodes(Positions[closest], ClosestPosInOrderList, Positions_depths)
                        # childOfClosestTreePOS = PositionInTree(childOfClosest[0], Positions_depths)
                        currentNodePOS = PositionInTree(
                            currentNode, Positions_depths)
                        currentNodeChild = findChildNodes(
                            currentNodePOS, currentNode, Positions_depths)
                        if currentNodeChild == []:
                            pass
                        else:
                            currentNodeChildTreePOS = PositionInTree(
                                currentNodeChild[0], Positions_depths)
                            if (currentNodeChildTreePOS in Positions_leaves):
                                loop = False
                                break
                            elif checkLabel(tree, currentNodeChildTreePOS) == "NP":
                                currentNode = currentNodeChild[0]
                            else:
                                leaves = findLeavesFromNode(
                                    currentNodePOS, Positions_leaves)
                                leafLabel = checkLabelLeaf(
                                    tree, leaves[len(leaves) - 1])
                                loop = False
                                break
                        count += 1

                    Triple.append(leafLabel)
                    sentence_triplesPP.append(Triple)

        # *********************************
        # *        POST PROCESSING        *
        # *********************************
        # print('**********Positions_leaves:',Positions_leaves,'************')

        # find the children of S
        # TODO implement new set of rule
        # locate all VP's in the sentence.

        # TODO post processing
        x = 0
        sentence_triples_copy = sentence_triples.copy()
        sentence_triplesPP_copy = sentence_triplesPP.copy()
        for x in sentence_triples_copy:
            if x[0] in illegal_concept_nodes or x[2] in illegal_concept_nodes:
                sentence_triples.remove(x)
            elif x[0] == 'Unknown' or x[2] == 'Unknown':
                sentence_triples.remove(x)
            elif x[0] == ',' or x[2] == ',':
                sentence_triples.remove(x)
            elif (not re.match(r'^\w+$', x[0]) or not re.match(r'^\w+$', x[1]) or not re.match(r'^\w+$', x[2])):
                sentence_triples.remove(x)

        for x in sentence_triplesPP_copy:  # remove invalid triples
            if x[0] in illegal_concept_nodes or x[2] in illegal_concept_nodes:
                sentence_triplesPP.remove(x)
            elif x[0] == 'Unknown' or x[2] == 'Unknown':
                sentence_triplesPP.remove(x)
            elif x[0] == ',' or x[2] == ',':
                sentence_triplesPP.remove(x)
            elif (not re.match(r'^\w+$', x[0]) or not re.match(r'^\w+$', x[1]) or not re.match(r'^\w+$', x[2])):
                sentence_triplesPP.remove(x)
        x = 0
        # for vb_triple in sentence_triples:  ## Phrasal verb composition
        #     for prp_triple in sentence_triplesPP:  # X vb Y followed by  X prp Y in same sentence
        #         if (prp_triple[0]== vb_triple[0] and prp_triple[2]== vb_triple[2] and
        #             (vb_triple[1] +" " +prp_triple[1]) in sents[i]): # sequence bv prp in the text
        #             if prp_triple in sentence_triplesPP:       #already removed?
        #                 sentence_triplesPP.remove(prp_triple)
        #             if vb_triple in sentence_triples:
        #                 sentence_triples.remove(vb_triple)
        #             sentence_triples.append([prp_triple[0], vb_triple[1]+"_"+prp_triple[1], prp_triple[2]])
        # print(vb_triple[1]+"_"+prp_triple[1], end="")

        # print('tree_leaves:', tree_leaves)
        print("sentence_triples: ", sentence_triples)
        print("sentence_triplesPP: ", sentence_triplesPP)
        print('causal_triples:', causal_triples)
        # print('vb_list_v2:',vb_list)
        # print("causal_triples:", causal_triples)
        # print('causal_triples_v2:', causal_triples_v2)
        print('##############################')

        tree.draw()  # show display parse tree

        document_triples.append(sentence_triples)
        document_triples.append(sentence_triplesPP)
        # causal_triples_list.append([causal_triples_v2])
        causal_triples_list.append([causal_triples])
        # print('$$$$$$$$$$$$$$',causal_triples_list)

    return


# *********************************
# *   OUTPUT PREPARATION        *
# *********************************


def generate_output_CSV_file(fileName):
    global document_triples
    testList = BringListDown1D(document_triples)
    causalList = BringListDown1D(causal_triples_list)
    if vbse:
        print(testList)
    heading = [["NOUN", "VERB/PREP", "NOUN"]]
    causal_heading = [["EFFECT", "CAUSAL_WORDS", "CAUSE"]]
    with open(outPath + fileName+".dcorf.csv", 'w', encoding="utf8") as resultFile:
        write = csv.writer(resultFile, lineterminator='\n')
        write.writerows(heading)
        write.writerows(testList)
    with open(outPath + fileName+".causal.csv", 'w', encoding="utf8") as causalFile:
        write = csv.writer(causalFile, lineterminator='\n')
        write.writerows(causal_heading)
        write.writerows(causalList)
    return


def processAllTextFiles():
    global inPath
    global document_triples
    global causal_triples_list
    global sentence_number
    global set_of_raw_concepts
    global skip_over_previous_results
    fileList = os.listdir(inPath)
    txt_files = [i for i in fileList if i.endswith('.txt')]
    # txt_files = [i for i in fileList]
    # txt_files.remove('Cache.txt')  # System file for Text 2 Predic8
    csv_files = [i for i in fileList if i.endswith('.csv')]

    for fileName in txt_files:
        set_of_raw_concepts = set()
        sentence_number = 0
        # print("\n####################################################")
        ###print("FILE ", inPath, "&&", fileName)
        if skip_over_previous_results and path.isfile(inPath + fileName + ".dcorf.csv"):
            print(" £skippy ", end="")
            continue
        global data
        data = []
        document_triples = []
        try:
            file = open(inPath+fileName, "r",
                        encoding="utf8", errors='replace')
        except Exception as err:
            print("Erro {}".format(err))
        #############
        # full_sentences = file.read()

        # processDocument(full_sentences[0])

        ##############################################################
        # '''
        full_document = file.read()
        full_document_list = full_document.split()
        text_chunk_size = 400
        text_chunk_start = 0
        text_chunk_end = text_chunk_size
        while text_chunk_start < len(full_document_list):
            z = full_document_list[text_chunk_start:text_chunk_end]
            documentSegment = " ".join(z)
            # print('documentSegment:',documentSegment)
            processDocument(documentSegment)
            text_chunk_start += text_chunk_size - 2
            text_chunk_end = text_chunk_end + text_chunk_size
            # if text_chunk_start >= 4500:
            #     print("End Of Document Truncated", end=" ")
            if text_chunk_end > len(full_document_list):
                text_chunk_end = min(
                    len(full_document_list), len(full_document_list))
            else:
                if text_chunk_start < len(full_document_list):
                    while (text_chunk_end >= text_chunk_start + (text_chunk_size - 100)) and \
                        ("." not in full_document_list[text_chunk_end] or
                         "?" not in full_document_list[text_chunk_end] or
                         ":" not in full_document_list[text_chunk_end]):  # split chunks @ full-stop, where reasonable
                        text_chunk_end -= 1
                        # '''
        ##############################################################
        # uses documentSegment, documentTriples

        # uses documentSegment, documentTriples
        generate_output_CSV_file(fileName)
        set_of_unique_concepts = set()
        for l in document_triples:
            for a, b, c in l:
                set_of_unique_concepts.add(a)
                set_of_unique_concepts.add(c)
        print("CONCEPT COUNT", ",", fileName, ",", len(set_of_unique_concepts))


print("Type   processAllTextFiles()   to generate graphs from ", inPath)
processAllTextFiles()
