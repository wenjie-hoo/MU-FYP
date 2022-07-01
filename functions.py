from bdb import effective
import pdb
import re
from this import s
from unittest.mock import sentinel


vbse = False  # VerBoSE mode for error reporting and tracing
causal_words = '.ecause|cause|cause.|unless|\
                Therefore|as a result|consequently'
causal_words_list = ['because', 'cause','caused','hence','therefore']
vb_tags_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']    

def findTriggerWords_v4(tree,tree_leaves):
    # print('*****************************')
    # print('tree_leaves:',tree_leaves)
    NPs = [" ".join(np.leaves()) for np in list(tree.subtrees(filter=lambda x: x.label()=='NP'))]
    # print('NPs:',NPs)
    for words in tree.leaves():
        if words == causal_words_list[1]:
            pass
    for np in NPs:
        pass

def findTriggerWords_v3(tree_leaves,tree_subtrees):
    # print('*****************************')
    # print('tree_leaves:',tree_leaves)
    causal_triple = []
    vp_list = []
    np_list = []
    causal = ''
    effect = ''
    causal_word = ''
    vp_causal_list =[]
    effect_list =[]
    causal_list  = []
    because_tag = False
    hence_tag = False
    caused_tag= False
    therefore_tag = False
    for subtree in tree_subtrees:
        if subtree.label() == 'VP':
            vp_list.append(subtree.leaves())
        if subtree.label() == 'NP':
            np_list.append(subtree.leaves())
        for words in subtree.leaves():
            if words == causal_words_list[0]:
                because_tag = True
            elif words == causal_words_list[1]:
                caused_tag = True
            elif words == causal_words_list[2]:
                hence_tag = True
            elif words == causal_words_list[3]:
                therefore_tag = True
                break
            
    if because_tag == True:
        i = -1
        # print('vp_list:',vp_list)
        for sent in vp_list:
            i +=1
            for word in sent:
                if word == causal_words_list[0]:
                    vp_causal_list.append(sent)
                    print('i:',i)
                    effect_list=vp_list[i]
                    pass
                else:
                    continue
        if len(vp_causal_list) != 0:
            for word in vp_causal_list[len(vp_causal_list)-1]:
                causal_list.append(word)
                if word in causal_words_list:
                    break
                else:
                    continue            
            causal_word = causal_list[-1]
            causal_list.pop()
        causal  = ' '.join(map(str, causal_list))
        effect = ' '.join(map(str, effect_list))
        causal_triple =[causal,causal_word,effect]
        # print('because tag causal triple:',causal_triple)
    elif caused_tag == True:
        np_causal_list = []
        print('np_list:',np_list)
        for sent in np_list:
            for word in sent:
                if word in causal_words_list:
                    np_causal_list=sent
        if len(np_causal_list) != 0:
            re_np_causal_list = np_causal_list[::-1]
            for word in np_causal_list:
                causal_list.append(word)
                if word == causal_words_list[1]:
                    causal_word = causal_words_list[1]
                    # print('causal_list:',causal_list)
                    break
            for word in re_np_causal_list:
                effect_list.append(word)
                if word == causal_words_list[1]:
                    break
            effect_list.pop()
            effect_list.reverse()
            causal_list.pop()
            causal =' '.join(map(str, causal_list))
            effect = ' '.join(map(str, effect_list))
            causal_triple = [causal,causal_word,effect]
            # print('caused_tag causal_triple:',causal_triple)
        elif len(np_causal_list) < 3:
            pass
        else:    
            causal = ' '.join(map(str, np_list[-2]))
            causal_word = causal_words_list[1]
            effect = ' '.join(map(str, np_list[-1]))
            causal_triple = [causal,causal_word,effect]
    elif hence_tag == True:
        # in progress
        pass
        
    elif therefore_tag == True:
        # in progress
        pass
               
def findTriggerWords_v2(tree_leaves, tree_subtrees):
    vb_list = []
    vcv_list = []
    for vb_word in tree_subtrees:
        if vb_word.label().startswith('NN') or \
            vb_word.label().startswith('NNS') or\
            vb_word.label().startswith('NNP')  or\
            vb_word.label().startswith('NNPS'):
            vb_list.append(vb_word.leaves()[0])
    for i in range(len(tree_leaves)):
        if tree_leaves[i] in causal_words_list and\
            tree_leaves[i] not in vb_list:
            vb_list.append(tree_leaves[i])
        if tree_leaves[i] in vb_list:
            vcv_list.append(tree_leaves[i])
    return vcv_list

def findPreBack_v2(listOfwords):
    causalWords = []
    len_listOfwords=len(listOfwords)
    for i in range(len_listOfwords):
        if len_listOfwords ==1:
            continue
        if listOfwords[i] in causal_words_list:
            if i == 0:
                a = listOfwords[i], listOfwords[i+1]
                for i in range(len(a)):
                    causalWords.append(a[i])
                continue
            if i > 0 and i != (len(listOfwords)-1):
                b = listOfwords[i-1], listOfwords[i], listOfwords[i+1]
                for i in range(len(b)):
                    causalWords.append(b[i])
                continue
            else:
                c = listOfwords[i-1], listOfwords[i]
                for i in range(len(c)):
                    causalWords.append(c[i])
    if len(causalWords) == 2:
        causalWords.insert(0,'')
    # if len(causalWords) == 0:
    #     pass
    return causalWords

def findTriggerWords(wordOftagged):
    triggerlPos = []
    for i in range(len(wordOftagged)):
        if re.match(causal_words, wordOftagged[i][0])or \
            re.match('VB|VB.|NN', wordOftagged[i][1], re.IGNORECASE):
            triggerlPos.append(wordOftagged[i])
            print('triggerlPos:',triggerlPos)
    return triggerlPos

def findPreBack(listOfwords):
    causalWords = []
    try:
        for i in range(len(listOfwords)):
            if re.match(causal_words, listOfwords[i][0], re.IGNORECASE):
                if i > 0 and i != (len(listOfwords)-1):
                    a = listOfwords[i-1], listOfwords[i], listOfwords[i+1]
                    for j in range(len(a)):
                        causalWords.append(a[j][0])
                elif i == 0:
                    b = listOfwords[i], listOfwords[i+1]
                    for j in range(len(b)):
                        causalWords.append(b[j][0])
                else:
                    c = listOfwords[i-1], listOfwords[i]
                    for j in range(len(c)):
                        causalWords.append(c[j][0])
    except IndexError:
        causalWords = []
        causalWords.append('No causal words found')
    return causalWords

def getListPos(tree):
    # print('tree.treepositions:',tree.treepositions())
    return tree.treepositions()
    

def PositionInTree(PositionInOrderList, OrderPos):
    pos = OrderPos[PositionInOrderList[0]][PositionInOrderList[1]]
    return pos

def getLeafPos(tree):
    # import json
    # t1 = json.dumps(tree)
    # with open('test.txt', 'w') as f:
    #     f.write(t1)
    leaves = [tree.leaf_treeposition(n) for n, x in enumerate(tree.leaves())] #tree.leaves返回叶子，列表。n计数器
    # t2 = json.dumps(leaves)
    # with open('test1.txt', 'w') as f:
    #     f.write(t2)
    return  leaves


def getOrderPos(PosList):
    #temp_list = []
    work_list = []
    deepest = 0         #树的最大深度
    length = len(PosList)
    # print('length:', length)
    #find deepest point
    for i in PosList:
        if len(i)>deepest:
            deepest = len(i)

    for i in range(deepest):
        temp_list= []
        #check values
        for x in range(length):
            if len(PosList[x]) == i:
                temp_list.append(PosList[x])
                # print(temp_list)
        work_list.append(temp_list)

    return (work_list)


def checkLabel(tree, Position):
    return tree[Position].label()


def checkLabelLeaf(tree, Position):
    #returns the string contained within a leaf node at the given position
    #print("CLL", tree, "Pos", Position)   #
    if isinstance(Position, list): 
        if len(Position) > 1:
            print("############### Error List #####")
            print(tree[Position[0]])
            # exit(1)
            return tree[Position[0]]
#        print(tree)
#        print(Position, end=" ", flush = True)
#        print(tree[Position])
#        exit(1)
    return tree[Position]



def findPosInOrderList(Position, orderPos):
    #returns the position in the ordered array of the current node
    #used to find the left/right sibling as well as child and parent nodes

    #Make sure Position is a tuple
    #print("in FPiOL", end="")
    if not type(Position)=="tuple":
        Position = tuple(Position)

    #print("xyz", end="")
    for i in range(len(orderPos)):
        part = orderPos[i]
        for j in range(len(part)):
            if part[j]==Position:
                return [i,j]


def findLeftSiblingCurrentLevel(PositionInOrderList, orderPos):
    #returns the position in the ordered list of nodes of the current positions left sibling
    #PositionInOrderList is the position of the current node in the ordered list of nodes
    #orderPos is the list of ordered node positions

    try:
        x = PositionInOrderList[0]
        y = PositionInOrderList[1]
        #print(x)
        #print(y)
        return [x,y-1]
    except:
        print("There was an error: findLeftSiblingCurrentLevel")



def findRightSiblingCurrentLevel(PositionInOrderList, orderPos):
    #returns the position in the ordered list of nodes of the current positions right sibling
    # PositionInOrderList is the position of the current node in the ordered list of nodes
    # orderPos is the list of ordered node positions

    try:
        x = PositionInOrderList[0]
        y = PositionInOrderList[1]
        return[x,y+1]
    except:
        print("There was an error: findRightSiblingCurrentLevel")


def findChildNodes(PositionInTree,PositionInOrderList, orderPos):
    #Takes the position of the current node in the ordered list of nodes
    #Takes the position of the current node in the tree
    #Returns a list of the children nodes, and their position in the ordered list
    #Position in tree is taken from the ordered list, it is the sequence of moves needed to go through the tree to get to the node is a tuple e.g (0,1,0)
    #PositionInOrderList is its position in the ordered list of nodes

    #pos = PositionInTree
    searchArea = PositionInOrderList[0]+1
    #print(pos)
    #print(searchArea)
    listOfChild = []
    #print("Functions.py//searchArea", searchArea)
    #print("Functions.py//orderPos", orderPos)

    if searchArea >= len(orderPos):
        searchArea = len(orderPos)-1

    #search through the lower level to find the children nodes
    for i in orderPos[searchArea]:
      #  if set(PositionInTree)<set(i):
      if  i[:len(PositionInTree)]==PositionInTree:
            temp = findPosInOrderList(i,orderPos)
            listOfChild.append(temp)
           # listOfChild.append(i)

   # print(PositionInTree)
    return listOfChild

def findLeavesFromNode(PositionInTree, LeafPos):
    #takes the position in the tree and returns a list of the positions of the leaf nodes that can be gotten to from the current node
    listOfLeaves = []
    for i in LeafPos:
        if i[:len(PositionInTree)]==PositionInTree:
            listOfLeaves.append(i)
    #print(listOfLeaves)
    return listOfLeaves


def findParentNode(PositionInTree, orderPos):
    parentPos = PositionInTree[:-1]
    PosInOrderList = findPosInOrderList(parentPos,orderPos)
    return PosInOrderList

def findParentNodeB(PositionInTree, orderPos):
    parentPos = PositionInTree[:-1]
   # PosInOrderList = findPosInOrderList(parentPos, orderPos)
    return parentPos

def findPosOfSpecificLabel(tree,label,orderPos, leafPos):
    #tree is the parsed tree
    #label is the specific label that you are searching for
    #orderPos is the list of all of the nodes in order of depth
    #leafPos is the a list of all of the positions of the leaf nodes (so that you do not try to check the label of the leaf node)
    #returns a list of positions containing the specified label
    foundLabel = False
    listOfPos = []
    for x in range(len(orderPos)):
        #print(str(x) + "this is the x value")
        for y in range(0, len(orderPos[x])):
            if not orderPos[x][y] in leafPos:
                check = checkLabel(tree,orderPos[x][y])
                if check ==label:
                    #print("found")
                    foundLabel = True
                    listOfPos.append([x,y])

    if foundLabel==False:
        if vbse:
            print("FUNCTIONS.py//Unsuccessful findPosOfSpecificLabel for",label, "  ")
        return None 
    else:
        # print('listofpos:::',listOfPos)
        return listOfPos


def CoreferenceResolution(corefs, parse_sent):
    #corefs is the list of all of the   , straight from the output of the corenlp
    #parse_sent is the sentence that will be turned into a parse tree, from the output
    #of the corenlp
    import  re
    import json
    amountCoref = len(corefs)

    for x in (corefs):
        if vbse: print("Coref: ", x)
            
        #lengthOfList = len(corefs[str(x+1)])
        lengthOfList = len(corefs[x])
    

        #HeadWord = corefs[str(x+1)][0]['text'] #This is the word you replace other words with
        HeadWord = corefs[x][0]['text'].replace(' ', '_')  # Eoin inserted April ‘19
        # print('888888888888',HeadWord)

        #HeadWord = corefs[x][0]['text'] //previously by dod
        
        #Loop through the rest of the corefs in the current set and
        #get a list of their words and then replace
        for y in range(1, lengthOfList):
            #print(corefs[str(x+1)][y]['text'])
            #word = corefs[str(x+1)][y]['text']
            word = corefs[x][y]['text']
            
            try:
                parse_sent = re.sub(r'\b'+word+r'\b', HeadWord+"_"+word, parse_sent) #illegal word or HeadWord
                
            except:
                print(" Timeout / Illegal RegEx ")
                #parse_sent = ""

    return (parse_sent)

def fiterEmptyList(List):
    causalList=[]
    for i in List:
        for j in i:
            if j !=[]:
                causalList.append(j)
    print('causalList:',causalList)
                

def BringListDown1D(List):
    flat_list = [item for sublist in List for item in sublist if item!=[]]  
    # print("flat_list:",flat_list)
    return flat_list

def RuleVBD(Tree, LIstOfVBD, OrderPos, Positions, PositionLeaves):
    #list of vbd is the list of all of the vbd found in the tree
    #orderpos is the ordered list of positions
    #positions is the unordered list of positions
    #tree is the tree
    #position leaves is the position of all of the leaves in the tree

    ListOfNP = findPosOfSpecificLabel(Tree, "NP", OrderPos, PositionLeaves)
    index = []
    for x in ListOfNP:
        index.append(Positions.index(OrderPos[x[0]][x[1]]))

    SetOfTriples = []
    print(LIstOfVBD)
    for x in LIstOfVBD:
        Triple = []
        verbWord = findChildNodes(OrderPos[x[0]][x[1]],x,OrderPos)
        posofVBDtree =Positions.index(OrderPos[x[0]][x[1]])

        #find out which NPs are on the left by subtracting if negative it is on the left
        closest = 0
        currentDif = -1000000
        for y in index:
            diff = y-posofVBDtree
            if(diff<0 and diff>currentDif):
                currentDif = diff
                closest = y

        #then check is closest has a parent that is an NP
        closestParent  = findParentNodeB(Positions[closest],OrderPos)
        if checkLabel(Tree,closestParent)=="NP":
            #obtain all leaf nodes from this parent
            leaves = findLeavesFromNode(closestParent,PositionLeaves)
            #then search through these positions and find the first that is an actual noun, to do this
            #you need to find out what the label of the parent node is.
            for z in leaves:
                parentOfLeaf = findParentNodeB(z, OrderPos)
                labelOfParent = checkLabel(Tree,parentOfLeaf)
                print(labelOfParent)
                if(labelOfParent=="NNS" or labelOfParent=="NN" or labelOfParent=="NNP" or labelOfParent=="NNPS" or labelOfParent=="PRP" or labelOfParent=="PRP$"):
                    Triple.append(checkLabelLeaf(Tree, z))
                    print(z)
                    print("I added something to the triple")
                    break
            #then append the verb that you are working with to the triple
            Triple.append(checkLabelLeaf(Tree, OrderPos[verbWord[0][0]][verbWord[0][1]]))

        else:
            #if the parent isnt an NP get child of current NP
            leaves = findLeavesFromNode(Positions[closest],PositionLeaves)
            for z in leaves:
                parentOfLeaf = findParentNodeB(z, OrderPos)
                labelOfParent = checkLabel(Tree, parentOfLeaf)
                if (labelOfParent=="NNS" or labelOfParent=="NN" or labelOfParent=="NNP" or labelOfParent=="NNPS"or labelOfParent=="PRP" or labelOfParent=="PRP$"):
                    Triple.append(checkLabelLeaf(Tree, z))
                    break
            #then append the verb you are working with.
            Triple.append(checkLabelLeaf(Tree, OrderPos[verbWord[0][0]][verbWord[0][1]]))

        #now get the first noun on the right of the verb
        closest = 0
        currentDif = 100000
        for y in index:
            diff = y-posofVBDtree
            if(diff>0 and diff<currentDif):
                currentDif = diff
                closest = y

        #check if closest has an NP parent
        closestParent = findParentNodeB(Positions[closest],OrderPos)
        if checkLabel(Tree, closestParent)=="NP":
            leaves = findParentNode(closestParent,PositionLeaves)
            #get the right most leaf
            Triple.append(checkLabelLeaf(Tree,leaves[len(leaves)-1]))
        else:
            #if the parent is not an NP just get right most child from current node
            leaves = findLeavesFromNode(Positions[closest],PositionLeaves)
            Triple.append(checkLabelLeaf(Tree,leaves[len(leaves)-1]))

        SetOfTriples.append(Triple)
    return SetOfTriples

# Eoin

import pickle
from nltk.stem.wordnet import WordNetLemmatizer
import warnings


def findChildNodesB(tree, position_in_tree):
    pos_set = set(getListPos(tree))
    i = 0
    children = []

    while True:
        child = position_in_tree + (i,)

        if child in pos_set:
            children.append(child)
        else:
            break

        i += 1

    return children


def phrasalVerbs(tree, cap=None, remove_branches=False):
    def phrasalVerbsHelper(vb_pos, tree, cap):
        # NOTE: vb_pos will remain constant throughout removal of leaves below

        print("Starting phrasal verb conversion.")

        with open("pos_of_phrasal_verbs.p", 'rb') as f:
            pos_dict = pickle.load(f)    # dictionary where keys are verbs and values are lists of tuples of full phrasal verbs with corresponding POS tags

        positions_leaves = getLeafPos(tree)
        verb_leaf = findLeavesFromNode(vb_pos, positions_leaves)[0]
        verb = checkLabelLeaf(tree, verb_leaf)

        key = WordNetLemmatizer().lemmatize(verb, "v")      # puts verb found in text into root form (the form the dictionary keys are in)

        ret = verb
        leaves_to_remove = []

        try:
            possible_matches = pos_dict[key]    # finds all possible full phrasal verbs which begin with the given verb
        except KeyError:
            return (False, verb, [])                # gives back verb unchanged with no leaves to remove if not a phrasal verb

        for full_phrase, labels in possible_matches:
            full_phrase_split = full_phrase.split(' ')      # the full phrasal verb split into individual words

            zipped = list(zip(labels[1:], full_phrase_split[1:]))

            for label, word in zipped:        # for each label in this particular full phrasal verb's POS breakdown and each word in the split version
                i = vb_pos[-1] + 1

                while True:
                    try:
                        start = vb_pos[:-1] + (i,)
                        start_label = checkLabel(tree, start)
                    except IndexError:
                        return (False, verb, [])

                    if start_label == 'NP':
                        i += 1
                    else:
                        leaves = findLeavesFromNode(start, positions_leaves)    # the leaves under the first non-NP sister of the VB or VBD
                        if leaves_to_remove:
                            leaves = list(filter(lambda l: l > leaves_to_remove[-1], leaves))
                        break

                if cap:
                    leaves = list(filter(
                        lambda l: 0 <= positions_leaves.index(verb_leaf) - positions_leaves.index(l) <= cap,
                        leaves
                    ))

                for leaf in leaves:
                    parent_of_leaf_label = checkLabel(tree, leaf[:-1])

                    if parent_of_leaf_label == label:     # when a leaf with the correct POS tag is found
                        leaf_label = checkLabelLeaf(tree, leaf)

                        if leaf_label == word:                       # if that leaf is the correct word for that full phrasal verb
                            print("Got a possible match.")
                            ret += '_' + leaf_label              # append it to the verb
                            leaves_to_remove.append(leaf)
                        else:
                            ret = verb                  # the word wasn't correct, so this full phrasal verb is not correct
                            leaves_to_remove = []       # we hence reset our tracking of the verb and the leaves to be removed
                            print("Match was bad")
                        break

                if ret == verb:     # this full phrasal verb is not correct, so we try another possibility
                    break

            if ret != verb:     # if we have found a match, we stop trying possibilities
                print("Got an actual match")
                break

        # if no match was found, warn the user
        if ret == verb:
            warnings.warn('Phrasal verb "{0}" detected but no conversion performed.'.format(verb), Warning)

        return (True, ret, leaves_to_remove)
    # end of def

    def removeBranches(tree, leaves_to_remove, vb_positions):
        for leaf in leaves_to_remove:
            while True:
                if findChildNodesB(tree, leaf):
                    break
                else:
                    tree.__delitem__(leaf)        # delete it
                    print("Removed a leaf.")

                    l = len(leaf)
                    for j in len(vb_positions):       # for each as-yet-unexamined verb
                        later_vb_pos = vb_pos

                        # if it shares a prefix with this leaf whose length is one less than the leaf's position's length
                        if len(later_vb_pos) >= l:
                            if all([leaf[k] == later_vb_pos[k] for k in range(l - 1)]):
                                # put that level back by one
                                vb_positions[j] = later_vb_pos[:(l - 1)] + (later_vb_pos[l - 1] - 1,) + later_vb_pos[l:]

                    leaf = leaf[:-1]

        return (tree, vb_positions)

    order_list = getOrderPos(getListPos(tree))
    positions_leaves = getLeafPos(tree)

    vb_positions = []
    for label in ["VB", "VBD", "VBG"]:
        xs = findPosOfSpecificLabel(tree, label, order_list, positions_leaves)
        vb_positions += xs if xs else []
    vb_positions = [PositionInTree(vb, order_list) for vb in vb_positions]      # converts from order positions to positions in tree
    vb_positions.sort()       # ensures rightmost verbs appear later in the list

    changed_list = []

    for i in range(len(vb_positions)):
        vb_pos = vb_positions[i]
        changed, new_verb, leaves_to_remove = phrasalVerbsHelper(vb_pos, tree, cap)

        # replace the old instance of the verb with the new "underscored" version
        print('Replacing with "{0}"'.format(new_verb))
        tree.__setitem__(findLeavesFromNode(vb_pos, getLeafPos(tree))[0], new_verb)

        if remove_branches:
            tree, vb_positions[(i + 1):] = removeBranches(tree, leaves_to_remove, vb_positions[(i + 1):])

        if changed:
            changed_list.append(new_verb)

    return (changed_list, tree)
