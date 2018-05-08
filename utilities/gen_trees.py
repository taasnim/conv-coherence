from __future__ import absolute_import
import glob, os, csv, re
import sys
#import nltk, string
#from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def insert_node(x=[],node="100"):
	tree_4 = []

	#addin new branch
	for item in x:
		tmp = [k for k in item] 
		tmp.append("1" + node)
		tree_4.append(tmp)
	
	for item in x:
		for i in item:
			#keep old_branches
			old_branches = [branch for branch in item if branch != i]
			#compute new sub trees
			new_subTrees = adding(x=i, new_node=node)
			
			for tree in new_subTrees:
				tmp = [branch for branch in tree if branch not in old_branches]
				#print tmp
				# each new_braches, adding to the old branches 
				tree_4.append(old_branches + tmp )

	return tree_4


def adding(x="123",new_node="4"):
	res = []
	res.append([x + new_node])
	n = len(x)

	tmp = []
	for i in range(1,n):
		tmp = []
		tmp.append(x)
		tmp.append(x[0:i] + new_node)

		res.append(tmp)


	return res


def gen_Tree3():
	return [["123"],["13","12"]]

def gen_Tree4():
	x3 = gen_Tree3() 
	tree_4 = insert_node(x=x3,node="4")
	return tree_4

def gen_Tree5():

	x4 = gen_Tree4() 
	tree_5 = insert_node(x=x4,node="5")

	#print ['1234','1235'] in tree_5
	return tree_5

def gen_Tree6():
	x = gen_Tree5() 
	tree_6 = insert_node(x=x,node="6")
	
	#print ['1246', '1245','1234'] in tree_6

	return tree_6

def gen_tree_branches(n=3):
	if n == 3:
		return [["123"],["12","13"]]
	if n == 4:
		return remove_duplication(x=gen_Tree4())
	if n == 5:
		return remove_duplication(x=gen_Tree5())
	if n == 6:
		return remove_duplication(x=gen_Tree6())
	if n >6:
		print("Have not supported larger threads (num of commnet > 6) yet...!")
		return None

def remove_duplication(x=[]):
	sort_x = []
	for i in x:
		sort_x.append('.'.join(sorted(i)))
	
	uniq_x = list(set(sort_x))

	res = []
	for i in uniq_x:
		res.append(i.split("."))

	return res

def adding_child(tree=[],nChild=4):

	if len(tree) ==0:
		return gen_Tree4()
	
	x_tree = [] #concert to strung 
	for x in tree:
		x_tree.append(''.join(x))
	
	res = []

	for x in x_tree:
		tmp = []
		tmp = [y for y in x_tree if y!=x]

		new = adding(x=x,new_node=str(nChild))
		for y in new:
			res.append(y + tmp)

	res = remove_x(remove_duplication(res))
	#adding all first and all-previous
	bl2=[]
	bl1 = "1"
	for i in range(2,nChild+1):
		bl2.append("1" + str(i))
		bl1 =bl1 + str(i)

	#res.append([''.join(range(1,nChild+1))])

	
	res += [bl2]
	res += [[bl1]]
	
	return res


def check_edge(branch=""):
	if "15" in branch:
		return False
	if "16" in branch:
		return False
	#if "14" in branch:
	#	return False
	if "25" in branch:
		return False
	if "26" in branch:
		return False
	#if "36" in branch:
	#	return False
	return True

# remove 15,16,26
def remove_x(tree=[]):
	res =[]
	for i in tree:
		tmp = '-'.join(i)
		if check_edge(tmp) == True:
			res = res + [i]
	for i in res:
		print(i)
	
	
	return res


def prune_trees(n=6):

	#get all first
	bl2=[]
	for i in range(2,n+1):
		bl2.append("1" + str(i))

	if n == 3:
		return [["123"],["12","13"]]
	if n == 4:
		return [bl2] + remove_x(remove_duplication(x=gen_Tree4()))
	if n == 5:
		return [bl2] + remove_x(remove_duplication(x=gen_Tree5()))

	if n == 6:
		return [bl2] + remove_x(remove_duplication(x=gen_Tree6()))

	if n >6:
		print("Have not supported larger threads (num of commnet > 6) yet...!")
		return None


	
#--------------------------------------------------
#nltk.download('punkt') # if necessary...
'''

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]



def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]



def extend_trees(posts=[],current_trees=[],n=4):
	#extend tree to 4 childs

	#pick two most similairty 
	distances = []
	for i in range(1,n):
		distances.append(cosine_sim(posts[i-1],posts[n-1]))
	
	replys = [i+1 for i in np.argsort(distances)[-2:]]
	print "Distance post " + str(n) + " to previous: " + str(distances)
	print "-------- post " + str(n) + " to " + str(sorted(replys))

	#print replys
	#print n
	res = []

	for tree in current_trees:
		#print "-------------"
		#print tree
		for last in  replys:
			res += [insert(current_tree=tree,last=last,new_node=n)]

			#now have to insert to the tree
	print res
	return res


def insert(current_tree=[],last=10,new_node=11):
	
	res = []
	
	if last ==1:
		res = current_tree
		res.append("1"+str(new_node))
		return res
	else:
		tmp1 = []
		tmp2 = []

		for branch in current_tree:
			if str(last) in branch:
				idx = branch.index(str(last)) + 1
				#print i[0:idx]
				tmp2 = branch[0:idx] + str(new_node)
				tmp1 = branch
				break
		
		if len(tmp2) > len(tmp1):
			res = [k for k in current_tree if k!=tmp1]
			res.append(tmp2)
		else:
			res = [k for k in current_tree]
			res.append(tmp2)

	return sorted(res)




def get_top_possible_trees(file=""):

	cmtIDs  = [line.rstrip('\n') for line in open(file + ".commentIDs")]
	cmtIDs = [int(i) for i in cmtIDs]

	texts = [line.rstrip('\n') for line in open(file + ".TEXT")]

	assert len(cmtIDs) == len(texts)

	nPOST = max(cmtIDs)
	posts = []
	for pID in range(1,nPOST +1):
		idx = [ i for i, item in enumerate(cmtIDs) if item==pID ]
		
		post = [ j for i,j in enumerate(texts) if i in idx ]
		
		posts.append(' '.join(post))

	if nPOST == 4:
		return extend_trees(current_trees=gen_tree_branches(n=3),posts=posts,n=4)

	if nPOST == 5:
		tmp = extend_trees(current_trees=gen_tree_branches(n=3),posts=posts,n=4)
		
		return extend_trees(current_trees=tmp,posts=posts,n=5)

		
	if nPOST == 6:
		tmp = extend_trees(current_trees=gen_tree_branches(n=3),posts=posts,n=4)
		tmp1 = extend_trees(current_trees=tmp,posts=posts,n=5)
		
		return extend_trees(current_trees=tmp1,posts=posts,n=6)
		
	return [["123"],["12","13"]]


#k = get_top_possible_trees(file="../final_data/CNET/threads/Thread_6552.txt.text.all.parsed.ner")

#for i in k:
#	print i

#k = remove_x(tree=gen_tree_branches(n=6))

'''
'''
k = prune_trees(n=5)

for i in k:
	print i

print len(k)

k = adding_child(tree=['12','13','145'], nChild=6)
#k=remove_duplication(x=k)

for i in k:
	print i
print "-------------------------------"

print len(k)



x = gen_tree_branches(int(sys.argv[1]))
for i in x:
	print i
print "-------------------------------"
print len(x)

#print "-------------------------------"
'''



	


