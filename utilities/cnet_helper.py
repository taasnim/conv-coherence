from __future__ import absolute_import
from utilities import gen_trees
import numpy as np
from keras.preprocessing import sequence
import itertools


def init_vocab(emb_size):
    vocabs =['0','S','O','X','-']
    np.random.seed(2017)
    E    = 0.01 * np.random.uniform( -1.0, 1.0, (len(vocabs), emb_size))
    E[0] = 0

    return vocabs, E

#=====================================================================
#loading data by branch
#
#=====================================================================
def load_data_by_branch_new(filelist="list_of_grid.txt", perm_num=20, maxlen=15000, w_size=3, vocabs=None, emb_size=300,
                            fn=None):
    vocab_idmap = {}
    idmap_vocab = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i
        idmap_vocab[i] = vocabs[i]

    # loading entiry-grid data from list of pos document and list of neg document
    list_of_files = [line.rstrip('\n') for line in open(filelist)]

    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []
    f_track = []  # tracking branch
    pair_id = 0
    zero_padding = np.zeros((w_size), dtype=np.int)

    for file_id, file in enumerate(list_of_files):
        # print(file)

        # loading commentIDs
        cmtIDs = [line.rstrip('\n') for line in open(file + ".commentIDs")]
        cmtIDs = [int(i) for i in cmtIDs]

        org_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]
        egrids = [line.rstrip('\n') for line in open(file + ".EGrid")]

        f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]
        f_list = {}
        for line in f_lines:
            x = line.split()
            entity = x[0]
            x = x[1:]  # remnove the entity name
            f_list[entity] = x

        # f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]

        for branch in org_tree:  # reading branch, each branch is considered as a document
            f_track.append(pair_id)  # keep track branch for each thread

            branch = [int(id) for id in branch.split('.')]  # convert string to integer, branch
            idxs = [idx for idx, cmtID in enumerate(cmtIDs) if cmtID in branch]

            # grid_1 = "0 " * w_size
            grid_1 = zero_padding
            for idx, line in enumerate(egrids):
                # e_trans = get_eTrans_by_Index_word_entity(e_trans=line, idxs=idxs, vocabs=vocabs, f_list=f_list,feats=fn)  # merge the grid of positive document
                e_trans = get_eTrans_by_Index_extended(e_trans=line, idxs=idxs, vocab_idmap=vocab_idmap,
                                                       f_list=f_list,
                                                       feats=fn)

                if len(e_trans) != 0:
                    # print e_trans
                    # grid_1 = grid_1 + e_trans + " " + "0 " * w_size
                    grid_1 = np.concatenate((grid_1, e_trans, zero_padding), axis=0)

            # print grid_1

            p_count = 0
            for i in range(1, perm_num + 1):  # reading the permuted docs
                permuted_lines = [p_line.rstrip('\n') for p_line in open(file + ".EGrid" + "-" + str(i))]
                grid_0 = zero_padding

                for idx, p_line in enumerate(permuted_lines):
                    # e_trans_0 = get_eTrans_by_Index_word_entity(e_trans=p_line, idxs=idxs, vocabs=vocabs, f_list=f_list, feats=fn)
                    e_trans_0 = get_eTrans_by_Index_extended(e_trans=p_line, idxs=idxs,
                                                             vocab_idmap=vocab_idmap,
                                                             f_list=f_list,
                                                             feats=fn)
                    if len(e_trans_0) != 0:
                        # grid_0 = grid_0 + e_trans_0 + " " + "0 " * w_size
                        grid_0 = np.concatenate((grid_0, e_trans_0, zero_padding), axis=0)

                # if grid_0 != grid_1:  # check the duplication
                if np.array_equal(grid_0, grid_1) == False:
                    p_count = p_count + 1
                    sentences_0.append(padding_len_extended(grid_0, maxlen, w_size))
            # print(len(grid_1))
            for i in range(0, p_count):
                sentences_1.append(padding_len_extended(grid_1, maxlen, w_size))

        pair_id += 1

    assert len(sentences_0) == len(sentences_1)

    return sentences_1, sentences_0, f_track


def get_eTrans_by_Index(e_trans="", idxs=None, f_list={}, feats=None):

    x = e_trans.split()
    entity = x[0]
    x = x[1:] # remove the first 

    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of occurrence of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""

    final_sent = []

    for idx in idxs:
        final_sent.append(x[idx])  #id in file starts at 1
    if feats==None: #coherence model without features    
        return ' '.join(final_sent)

    f = f_list[entity]
    x_f = []
    for sem_role in x:
        new_role = sem_role;
        if new_role != '-':
            for i in feats:
                if i ==0 : #adding salience
                    if e_occur == 1:
                        new_role = new_role + "F01"
                    elif e_occur == 2:
                        new_role = new_role + "F02"
                    elif e_occur == 3:
                        new_role = new_role + "F03"
                    elif e_occur >3 :
                        new_role = new_role + "F04"
                else:
                    new_role = new_role + "F" + str(i) + f[i-1] # num feat = idx + 1
  
        x_f.append(new_role)

    return ' '.join(x_f)



#=====================================================================
#loading data by tree/sentecne structure
#
#=====================================================================


# =====================================================================
# loading data by branch with word entity
#
# =====================================================================
def load_data_by_branch_word_entity(filelist="list_of_grid.txt", perm_num=20, maxlen=15000, w_size=11,
                                    vocabs=None, emb_size=300, fn=None):
    vocab_x = []
    vocab_x.append("0")
    vocab_x.append("-")
    for ent in vocabs:
        vocab_x.append(ent + "_S")
        vocab_x.append(ent + "_O")
        vocab_x.append(ent + "_X")

    vocab_idmap = {}
    idmap_vocab = {}
    for i in range(len(vocab_x)):
        vocab_idmap[vocab_x[i]] = i
        idmap_vocab[i] = vocab_x[i]

    # loading entiry-grid data from list of pos document and list of neg document
    list_of_files = [line.rstrip('\n') for line in open(filelist)]

    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []
    f_track = []  # tracking branch
    pair_id = 0
    zero_padding = np.zeros((w_size), dtype=np.int)

    for file_id, file in enumerate(list_of_files):
        # print(file)

        # loading commentIDs
        cmtIDs = [line.rstrip('\n') for line in open(file + ".commentIDs")]
        cmtIDs = [int(i) for i in cmtIDs]

        org_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]
        egrids = [line.rstrip('\n') for line in open(file + ".EGrid")]

        f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]
        f_list = {}
        for line in f_lines:
            x = line.split()
            entity = x[0]
            x = x[1:]  # remnove the entity name
            f_list[entity] = x

        # f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]

        for branch in org_tree:  # reading branch, each branch is considered as a document
            f_track.append(pair_id)  # keep track branch for each thread

            branch = [int(id) for id in branch.split('.')]  # convert string to integer, branch
            idxs = [idx for idx, cmtID in enumerate(cmtIDs) if cmtID in branch]

            # grid_1 = "0 " * w_size
            grid_1 = zero_padding
            for idx, line in enumerate(egrids):
                # e_trans = get_eTrans_by_Index_word_entity(e_trans=line, idxs=idxs, vocabs=vocabs, f_list=f_list,feats=fn)  # merge the grid of positive document
                e_trans = get_eTrans_by_Index_word_entity_extended(e_trans=line, idxs=idxs, vocab_idmap=vocab_idmap,
                                                                   f_list=f_list,
                                                                   feats=fn)

                if len(e_trans) != 0:
                    # print e_trans
                    # grid_1 = grid_1 + e_trans + " " + "0 " * w_size
                    grid_1 = np.concatenate((grid_1, e_trans, zero_padding), axis=0)

            # print grid_1

            p_count = 0
            for i in range(1, perm_num + 1):  # reading the permuted docs
                permuted_lines = [p_line.rstrip('\n') for p_line in open(file + ".EGrid" + "-" + str(i))]
                grid_0 = zero_padding

                for idx, p_line in enumerate(permuted_lines):
                    # e_trans_0 = get_eTrans_by_Index_word_entity(e_trans=p_line, idxs=idxs, vocabs=vocabs, f_list=f_list, feats=fn)
                    e_trans_0 = get_eTrans_by_Index_word_entity_extended(e_trans=p_line, idxs=idxs,
                                                                         vocab_idmap=vocab_idmap,
                                                                         f_list=f_list,
                                                                         feats=fn)
                    if len(e_trans_0) != 0:
                        # grid_0 = grid_0 + e_trans_0 + " " + "0 " * w_size
                        grid_0 = np.concatenate((grid_0, e_trans_0, zero_padding), axis=0)

                # if grid_0 != grid_1:  # check the duplication
                if np.array_equal(grid_0, grid_1) == False:
                    p_count = p_count + 1
                    sentences_0.append(padding_len_extended(grid_0, maxlen, w_size))
            # print(len(grid_1))
            for i in range(0, p_count):
                sentences_1.append(padding_len_extended(grid_1, maxlen, w_size))

        pair_id += 1

    assert len(sentences_0) == len(sentences_1)

    vocab_info = {"vocab_x": vocab_x,
                  "vocab_idmap": vocab_idmap,
                  "idmap_vocab": idmap_vocab}

    return sentences_1, sentences_0, f_track, vocab_info



def get_eTrans_by_Index_word_entity(e_trans="", idxs=None, vocabs=None, f_list={}, feats=None):
    x = e_trans.split()
    entity = x[0]
    x = x[1:]  # remove the first

    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O')  # counting the number of occurrence of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""

    final_sent = []

    for idx in idxs:
        role = x[idx]  # id in file starts at 1

        if role != "-":
            if entity in vocabs:
                final_sent.append(entity + "_" + role)
            else:
                final_sent.append("ovv_" + role)

        else:
            final_sent.append("-")

    if feats == None:  # coherence model without features
        return ' '.join(final_sent)

    f = f_list[entity]
    x_f = []
    for sem_role in x:
        new_role = sem_role;
        if new_role != '-':
            for i in feats:
                if i == 0:  # adding salience
                    if e_occur == 1:
                        new_role = new_role + "F01"
                    elif e_occur == 2:
                        new_role = new_role + "F02"
                    elif e_occur == 3:
                        new_role = new_role + "F03"
                    elif e_occur > 3:
                        new_role = new_role + "F04"
                else:
                    new_role = new_role + "F" + str(i) + f[i - 1]  # num feat = idx + 1

        x_f.append(new_role)

    return ' '.join(x_f)


##*********************************************
##******** For Tree level *********************

def load_data_by_tree_tensor(filelist="list_of_grid.txt", perm_num=20, maxlen=10000, w_size=3, vocabs=None,
                             max_entity=50, max_depth=20, max_branch=5):
    # loading entiry-grid data from list of pos document and list of neg document
    # Max Entity = dim1 = 200
    # Max Level  = dim2 = 300
    # Max Branch = dim3 = 20

    vocab_x = []
    for ent in vocabs:
        vocab_x.append(ent + "_S")
        vocab_x.append(ent + "_O")
        vocab_x.append(ent + "_X")

    vocab_x.append("-")
    vocab_x.append("0")

    vocab_idmap = {}
    idmap_vocab = {}
    for i in range(len(vocab_x)):
        vocab_idmap[vocab_x[i]] = i
        idmap_vocab[i] = vocab_x[i]

    list_of_files = [line.rstrip('\n') for line in open(filelist)]

    # process postive gird, convert each file to be a sentence
    tensor_1 = []
    tensor_0 = []
    zero_padding = np.zeros((w_size, max_branch), dtype=np.int)

    for file_id, file in enumerate(list_of_files):
        # print(file)
        depths = [int(line) for line in open(file + ".depth")]
        egrids = [line.rstrip('\n') for line in open(file + ".EGrid")]

        pos_tensor = zero_padding
        for idx, line in enumerate(egrids):
            role_matrix = get_eTrans_by_Depth_Tensor(e_trans=line, depths=depths,
                                                     vocab_idmap=vocab_idmap, max_depth=max_depth,
                                                     max_branch=max_branch)  # merge the grid of positive document
            if role_matrix is not None:
                pos_tensor = np.vstack((pos_tensor, role_matrix, zero_padding))
                # pos_tensor.append(role_matrix)

        # padding entity here
        # print(np.shape(pos_tensor))
        # print(len(pos_tensor))
        # print(pos_tensor)

        tree_tensor_1 = padding_entity(pos_tensor, maxlen, max_branch)
        # print(np.shape(tree_tensor_1))

        p_count = 0
        for i in range(1, perm_num + 1):  # reading the permuted docs
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file + ".EGrid" + "-" + str(i))]

            neg_tensor = zero_padding
            for idx, p_line in enumerate(permuted_lines):
                role_matrix = get_eTrans_by_Depth_Tensor(e_trans=p_line, depths=depths,
                                                         vocab_idmap=vocab_idmap, max_depth=max_depth,
                                                         max_branch=max_branch)  # merge the grid of positive document
                if role_matrix is not None:
                    neg_tensor = np.vstack((neg_tensor, role_matrix, zero_padding))
                    # neg_tensor.append(role_matrix)

            # print(np.shape(neg_tensor))
            # padding entity here
            tree_tensor_0 = padding_entity(neg_tensor, maxlen, max_branch)
            # tree_tensor_0 = padding_entity(neg_tensor, max_entity, max_depth, max_branch)

            # if neg_tensor is not pos_tensor: #check the duplication stupid
            if np.array_equal(neg_tensor, pos_tensor) == False:
                # print("Hello")
                p_count = p_count + 1
                tensor_0.append(tree_tensor_0)

        for i in range(0, p_count):  # stupid code
            tensor_1.append(tree_tensor_1)

    assert len(tensor_0) == len(tensor_1)

    vocab_info = {"vocab_x": vocab_x,
                  "vocab_idmap": vocab_idmap,
                  "idmap_vocab": idmap_vocab}


    #tensor_0 = np.asarray(tensor_0)
    #tensor_1 = np.asarray(tensor_1)

    return tensor_1, tensor_0, vocab_info


def get_eTrans_by_Depth_Tensor(e_trans="", depths=None, vocab_idmap=None, max_depth=20, max_branch=5):
    # get a matrix for a entity

    x = e_trans.split()
    entity = x[0]
    if entity + "_S" not in vocab_idmap:
        entity = "oov"

    role_list = x[1:]  # remove the first as entity
    length = len(role_list)
    e_occur = role_list.count('X') + role_list.count('S') + role_list.count(
        'O')  # counting the number of occurrence of entities
    if length > 80:
        if e_occur < 3:
            return None
    elif length > 20:
        if e_occur < 2:
            return None

    maxDepth = max(depths)
    role_matrix = []

    for lv in range(0, maxDepth + 1):
        idxs = [i for i, level in enumerate(depths) if level == lv]
        # get the the role for each depth
        tmp = []
        for idx in idxs:
            role = role_list[idx]
            if role == "-":
                tmp.append(vocab_idmap[role])
                # tmp.append(role)
            else:
                tmp.append(vocab_idmap[entity + "_" + role])
                # tmp.append(entity + "_" + role)
        role_matrix.append(tmp)

    # padding for branch like keras sequence padding
    role_matrix = new_adjust_index(role_matrix,
                                   maxlen=max_branch)  # make branch length to max_branch if branch length is too long
    role_matrix = sequence.pad_sequences(role_matrix, max_branch)

    # padding for depth
    # role_matrix = padding_depth(role_matrix, max_depth, max_branch)
    # print(np.shape(role_matrix))
    role_matrix = np.fliplr(role_matrix)
    return role_matrix


def new_adjust_index(X, maxlen=None):
    if maxlen:  # exclude tweets that are larger than maxlen
        new_X = []
        for x in X:
            if len(x) > maxlen:
                tmp = x[0:maxlen]
                new_X.append(tmp)
            else:
                new_X.append(x)
        X = new_X
    return X


def padding_depth(role_matrix, max_depth, max_branch):
    # depth padding
    if len(role_matrix) > max_depth:
        # cut off role matrix
        return role_matrix[0:max_depth, 0:max_branch]

    offset = max_depth - len(role_matrix)
    result = np.zeros((max_depth, max_branch), dtype=np.int)
    result[offset:max_depth, 0: max_branch] = role_matrix

    return result

def padding_entity(tree_tensor, maxlen, max_branch):
    #depth padding
    if len(tree_tensor) > maxlen:
        #cut off entities
        return tree_tensor[0:maxlen, :]

    result = np.zeros((maxlen, max_branch), dtype=np.int)
    offset = len(tree_tensor)
    result[0:offset, :] = tree_tensor

    return result

########################################################################
##*********************************************
##******** For Tree level 4D *********************

def load_4D_tree_tensor(filelist="list_of_grid.txt", perm_num=20, w_size=3, vocabs=None,
                        max_entity=20, max_depth=10, max_branch=5):
    # loading entiry-grid data from list of pos document and list of neg document
    # Max Entity = dim1 = 200
    # Max Level  = dim2 = 300
    # Max Branch = dim3 = 20

    vocab_x = []
    for ent in vocabs:
        vocab_x.append(ent + "_S")
        vocab_x.append(ent + "_O")
        vocab_x.append(ent + "_X")

    vocab_x.append("-")
    vocab_x.append("0")

    vocab_idmap = {}
    idmap_vocab = {}
    for i in range(len(vocab_x)):
        vocab_idmap[vocab_x[i]] = i
        idmap_vocab[i] = vocab_x[i]

    list_of_files = [line.rstrip('\n') for line in open(filelist)]

    # process postive gird, convert each file to be a sentence
    tensor_1 = []
    tensor_0 = []

    for file_id, file in enumerate(list_of_files):
        # print(file)
        depths = [int(line) for line in open(file + ".depth")]
        egrids = [line.rstrip('\n') for line in open(file + ".EGrid")]

        pos_tensor = []  # np.zeros((max_depth, max_branch), dtype=np.int)
        for idx, line in enumerate(egrids):
            role_matrix = get_2D_eTrans_Tensor(e_trans=line, depths=depths,
                                               vocab_idmap=vocab_idmap, max_depth=max_depth,
                                               max_branch=max_branch)  # merge the grid of positive document
            if role_matrix is not None:
                # pos_tensor = np.vstack((pos_tensor, role_matrix))
                pos_tensor.append(role_matrix)

        # padding entity here
        tree_tensor_1 = padding_entity_new(pos_tensor, max_entity, max_depth, max_branch)

        p_count = 0
        for i in range(1, perm_num + 1):  # reading the permuted docs
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file + ".EGrid" + "-" + str(i))]

            neg_tensor = []
            for idx, p_line in enumerate(permuted_lines):
                role_matrix = get_2D_eTrans_Tensor(e_trans=p_line, depths=depths,
                                                   vocab_idmap=vocab_idmap, max_depth=max_depth,
                                                   max_branch=max_branch)  # merge the grid of positive document
                if role_matrix is not None:
                    neg_tensor.append(role_matrix)

            # padding entity here
            tree_tensor_0 = padding_entity_new(neg_tensor, max_entity, max_depth, max_branch)

            # if neg_tensor is not pos_tensor: #check the duplication
            if np.array_equal(neg_tensor, pos_tensor) == False:
                # print("Hello")
                p_count = p_count + 1
                tensor_0.append(tree_tensor_0)

        for i in range(0, p_count):  # stupid code
            tensor_1.append(tree_tensor_1)

    assert len(tensor_0) == len(tensor_1)

    vocab_info = {"vocab_x": vocab_x,
                  "vocab_idmap": vocab_idmap,
                  "idmap_vocab": idmap_vocab}

    #tensor_0 = np.asarray(tensor_0)
    #tensor_1 = np.asarray(tensor_1)

    # print(np.shape(tensor_0))


    return tensor_1, tensor_0, vocab_info



def get_2D_eTrans_Tensor(e_trans="", depths=None, vocab_idmap=None, max_depth=20, max_branch=5):
    # get a matrix for a entity

    x = e_trans.split()
    entity = x[0]
    if entity + "_S" not in vocab_idmap:
        entity = "oov"

    role_list = x[1:]  # remove the first as entity
    length = len(role_list)
    e_occur = role_list.count('X') + role_list.count('S') + role_list.count(
        'O')  # counting the number of occurrence of entities
    if length > 80:
        if e_occur < 3:
            return None
    elif length > 20:
        if e_occur < 2:
            return None

    maxDepth = max(depths)
    role_matrix = []

    for lv in range(0, maxDepth + 1):
        idxs = [i for i, level in enumerate(depths) if level == lv]
        # get the the role for each depth
        tmp = []
        for idx in idxs:
            role = role_list[idx]
            if role == "-":
                tmp.append(vocab_idmap[role])
                # tmp.append(role)
            else:
                tmp.append(vocab_idmap[entity + "_" + role])
                # tmp.append(entity + "_" + role)
        role_matrix.append(tmp)

    # padding for branch like keras sequence padding
    role_matrix = new_adjust_index(role_matrix,
                                   maxlen=max_branch)  # make branch length to max_branch if branch length is too long
    role_matrix = sequence.pad_sequences(role_matrix, max_branch)

    # padding for depth
    role_matrix = padding_depth(role_matrix, max_depth, max_branch)
    #print(role_matrix)
    role_matrix = np.fliplr(role_matrix)

    return role_matrix


def padding_entity_new(tree_tensor, max_entity, max_depth, max_branch):
    #depth padding
    if len(tree_tensor) > max_entity:
        # cut off entities
        result = np.zeros((max_entity, max_depth, max_branch), dtype=np.int)
        result[0:max_entity, 0:max_depth, 0:max_branch] = tree_tensor[0:max_entity]
        # return tree_tensor[0:max_entity]#, 0:max_depth, 0:max_branch]
        return result

    result = np.zeros((max_entity, max_depth, max_branch), dtype=np.int)
    offset = max_entity - len(tree_tensor)
    result[offset:max_entity, 0:max_depth, 0:max_branch] = tree_tensor

    return result


########################################################################
##*********************************************
##******** Pathlevel Extended *********************

def load_data_by_branch_word_entity_extended(filelist="list_of_grid.txt", perm_num=20, maxlen=2500, maxbranch=15,
                                             w_size=3, vocabs=None, emb_size=300, fn=None):
    vocab_x = []
    vocab_x.append("0")
    vocab_x.append("-")
    for ent in vocabs:
        vocab_x.append(ent + "_S")
        vocab_x.append(ent + "_O")
        vocab_x.append(ent + "_X")

    vocab_idmap = {}
    idmap_vocab = {}
    for i in range(len(vocab_x)):
        vocab_idmap[vocab_x[i]] = i
        idmap_vocab[i] = vocab_x[i]

    # loading entiry-grid data from list of pos document and list of neg document
    list_of_files = [line.rstrip('\n') for line in open(filelist)]

    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []
    f_track = []  # tracking branch
    pair_id = 0

    zero_padding = np.zeros((w_size), dtype=np.int)

    for file_id, file in enumerate(list_of_files):
        # print(file)

        # loading commentIDs
        cmtIDs = [line.rstrip('\n') for line in open(file + ".commentIDs")]
        cmtIDs = [int(i) for i in cmtIDs]

        org_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]
        egrids = [line.rstrip('\n') for line in open(file + ".EGrid")]

        f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]
        f_list = {}
        for line in f_lines:
            x = line.split()
            entity = x[0]
            x = x[1:]  # remnove the entity name
            f_list[entity] = x

        # f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]
        max_path_len = 0
        pos_path = []

        for branch in org_tree:  #

            branch = [int(id) for id in branch.split('.')]  # convert string to integer, branch
            idxs = [idx for idx, cmtID in enumerate(cmtIDs) if cmtID in branch]

            grid_1 = zero_padding
            for idx, line in enumerate(egrids):
                e_trans = get_eTrans_by_Index_word_entity_extended(e_trans=line, idxs=idxs, vocab_idmap=vocab_idmap,
                                                                   f_list=f_list,
                                                                   feats=fn)  # merge the grid of positive document
                if len(e_trans) != 0:
                    # print e_trans
                    grid_1 = np.concatenate((grid_1, e_trans, zero_padding), axis=0)

                    # print(np.shape(grid_1))
            pos_path.append(padding_len_extended(grid_1, maxlen, w_size))
            # print(np.shape(pos_path[-1]))

        pos_doc = []
        if (len(pos_path) > maxbranch):
            for i in range(maxbranch):
                pos_doc.append(pos_path[i])
        else:
            offset = maxbranch - len(pos_path)
            for i in range(len(pos_path)):
                pos_doc.append(pos_path[i])
            for i in range(offset):
                pos_doc.append(np.zeros((maxlen), dtype=np.int))

        # pos_doc = np.vstack(pos_doc)
        pos_doc = np.column_stack(pos_doc)
        # print(np.shape(pos_doc))

        p_count = 0
        for i in range(1, perm_num + 1):  # reading the permuted docs
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file + ".EGrid" + "-" + str(i))]
            neg_path = []

            for branch in org_tree:

                branch = [int(id) for id in branch.split('.')]  # convert string to integer, branch
                idxs = [idx for idx, cmtID in enumerate(cmtIDs) if cmtID in branch]

                grid_0 = zero_padding
                for idx, p_line in enumerate(permuted_lines):
                    e_trans_0 = get_eTrans_by_Index_word_entity_extended(e_trans=p_line, idxs=idxs,
                                                                         vocab_idmap=vocab_idmap,
                                                                         f_list=f_list,
                                                                         feats=fn)  # merge the grid of positive document
                    if len(e_trans_0) != 0:
                        # print e_trans
                        grid_0 = np.concatenate((grid_0, e_trans_0, zero_padding), axis=0)

                        # print(np.shape(grid_0))
                neg_path.append(padding_len_extended(grid_0, maxlen, w_size))
                # print(np.shape(neg_path[-1]))

            neg_doc = []
            if (len(neg_path) > maxbranch):
                for i in range(maxbranch):
                    neg_doc.append(neg_path[i])
            else:
                offset = maxbranch - len(neg_path)
                for i in range(len(neg_path)):
                    neg_doc.append(neg_path[i])
                for i in range(offset):
                    neg_doc.append(np.zeros((maxlen), dtype=np.int))

            # neg_doc = np.vstack(neg_doc)
            neg_doc = np.column_stack(neg_doc)
            # print(np.shape(neg_doc))
            if np.array_equal(neg_doc, pos_doc) == False:
                p_count = p_count + 1
                sentences_0.append(neg_doc)

        for i in range(0, p_count):
            sentences_1.append(pos_doc)

        # sentences_1, sentences_0 = [],[]
        # print(p_count)

    vocab_info = {"vocab_x": vocab_x,
                  "vocab_idmap": vocab_idmap,
                  "idmap_vocab": idmap_vocab}

    return sentences_1, sentences_0, vocab_info


def get_eTrans_by_Index_word_entity_extended(e_trans="", idxs=None, vocab_idmap=None, f_list={}, feats=None):
    x = e_trans.split()
    entity = x[0]
    x = x[1:]  # remove the first
    if entity + "_S" not in vocab_idmap:
        entity = "oov"

    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O')  # counting the number of occurrence of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""

    final_sent = []

    for idx in idxs:
        role = x[idx]  # id in file starts at 1
        if role == "-":
            final_sent.append(vocab_idmap[role])
        else:
            final_sent.append(vocab_idmap[entity + "_" + role])

    if feats == None:  # coherence model without features
        return np.asarray(final_sent)



def padding_len_extended(pos_path, maxlen, w_size):
    result = np.zeros((maxlen), dtype=np.int)
    if(len(pos_path)> maxlen):
        offset = maxlen - w_size
        result[0:offset] = pos_path[0:offset]
    else:
        offset = len(pos_path)
        result[0:offset] = pos_path
    return result


##*********************************************
##******** Pathlevel Extended gridCNN *********************

def load_data_by_branch_extended(filelist="list_of_grid.txt", perm_num=20, maxlen=2500, maxbranch=15,
                                             w_size=3, vocabs=None, emb_size=300, fn=None):


    vocab_idmap = {}
    idmap_vocab = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i
        idmap_vocab[i] = vocabs[i]

    # loading entiry-grid data from list of pos document and list of neg document
    list_of_files = [line.rstrip('\n') for line in open(filelist)]

    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []


    zero_padding = np.zeros((w_size), dtype=np.int)

    for file_id, file in enumerate(list_of_files):
        # print(file)

        # loading commentIDs
        cmtIDs = [line.rstrip('\n') for line in open(file + ".commentIDs")]
        cmtIDs = [int(i) for i in cmtIDs]

        org_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]
        egrids = [line.rstrip('\n') for line in open(file + ".EGrid")]

        f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]
        f_list = {}
        for line in f_lines:
            x = line.split()
            entity = x[0]
            x = x[1:]  # remove the entity name
            f_list[entity] = x

        # f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]
        max_path_len = 0
        pos_path = []

        for branch in org_tree:  #

            branch = [int(id) for id in branch.split('.')]  # convert string to integer, branch
            idxs = [idx for idx, cmtID in enumerate(cmtIDs) if cmtID in branch]

            grid_1 = zero_padding
            for idx, line in enumerate(egrids):
                e_trans = get_eTrans_by_Index_extended(e_trans=line, idxs=idxs, vocab_idmap=vocab_idmap,
                                                                   f_list=f_list,
                                                                   feats=fn)  # merge the grid of positive document
                if len(e_trans) != 0:
                    # print e_trans
                    grid_1 = np.concatenate((grid_1, e_trans, zero_padding), axis=0)

                    # print(np.shape(grid_1))
            pos_path.append(padding_len_extended(grid_1, maxlen, w_size))
            # print(np.shape(pos_path[-1]))

        pos_doc = []
        if (len(pos_path) > maxbranch):
            for i in range(maxbranch):
                pos_doc.append(pos_path[i])
        else:
            offset = maxbranch - len(pos_path)
            for i in range(len(pos_path)):
                pos_doc.append(pos_path[i])
            for i in range(offset):
                pos_doc.append(np.zeros((maxlen), dtype=np.int))

        # pos_doc = np.vstack(pos_doc)
        pos_doc = np.column_stack(pos_doc)
        # print(np.shape(pos_doc))

        p_count = 0
        for i in range(1, perm_num + 1):  # reading the permuted docs
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file + ".EGrid" + "-" + str(i))]
            neg_path = []

            for branch in org_tree:

                branch = [int(id) for id in branch.split('.')]  # convert string to integer, branch
                idxs = [idx for idx, cmtID in enumerate(cmtIDs) if cmtID in branch]

                grid_0 = zero_padding
                for idx, p_line in enumerate(permuted_lines):
                    e_trans_0 = get_eTrans_by_Index_extended(e_trans=p_line, idxs=idxs,
                                                                         vocab_idmap=vocab_idmap,
                                                                         f_list=f_list,
                                                                         feats=fn)  # merge the grid of positive document
                    if len(e_trans_0) != 0:
                        # print e_trans
                        grid_0 = np.concatenate((grid_0, e_trans_0, zero_padding), axis=0)

                        # print(np.shape(grid_0))
                neg_path.append(padding_len_extended(grid_0, maxlen, w_size))
                # print(np.shape(neg_path[-1]))

            neg_doc = []
            if (len(neg_path) > maxbranch):
                for i in range(maxbranch):
                    neg_doc.append(neg_path[i])
            else:
                offset = maxbranch - len(neg_path)
                for i in range(len(neg_path)):
                    neg_doc.append(neg_path[i])
                for i in range(offset):
                    neg_doc.append(np.zeros((maxlen), dtype=np.int))

            # neg_doc = np.vstack(neg_doc)
            neg_doc = np.column_stack(neg_doc)
            # print(np.shape(neg_doc))
            if np.array_equal(neg_doc, pos_doc) == False:
                p_count = p_count + 1
                sentences_0.append(neg_doc)

        for i in range(0, p_count):
            sentences_1.append(pos_doc)

        # sentences_1, sentences_0 = [],[]
        # print(p_count)

    vocab_info = {"vocab_idmap": vocab_idmap,
                  "idmap_vocab": idmap_vocab}

    return sentences_1, sentences_0, vocab_info


def get_eTrans_by_Index_extended(e_trans="", idxs=None, vocab_idmap=None, f_list={}, feats=None):
    x = e_trans.split()
    entity = x[0]
    x = x[1:]  # remove the first

    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O')  # counting the number of occurrence of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""

    final_sent = []

    for idx in idxs:
        role = x[idx]  # id in file starts at 1
        final_sent.append(vocab_idmap[role])

    if feats == None:  # coherence model without features
        return np.asarray(final_sent)




# ===============================================
# for thread reconstruction 02
# based on load_data_by_branch_word_entity_extended

def load_reconstruction_task_02(filelist="list_of_grid.txt", perm_num=20, maxlen=2500, maxbranch=4,
                                w_size=3, vocabs=None):
    vocab_x = []
    vocab_x.append("0")
    vocab_x.append("-")
    for ent in vocabs:
        vocab_x.append(ent + "_S")
        vocab_x.append(ent + "_O")
        vocab_x.append(ent + "_X")
    vocab_idmap = {}
    idmap_vocab = {}
    for i in range(len(vocab_x)):
        vocab_idmap[vocab_x[i]] = i
        idmap_vocab[i] = vocab_x[i]

    # loading entiry-grid data from list of pos document and list of neg document
    list_of_files = [line.rstrip('\n') for line in open(filelist)]

    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []
    f_track = []  # tracking branch
    pair_id = 0

    zero_padding = np.zeros((w_size), dtype=np.int)

    for file_id, file in enumerate(list_of_files):
        # print(file)

        # loading commentIDs
        cmtIDs = [int(line.rstrip('\n')) for line in open(file + ".commentIDs")]
        org_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]
        egrids = [line.rstrip('\n') for line in open(file + ".EGrid")]

        max_path_len = 0
        pos_path = []

        # print(org_tree)
        for branch in org_tree:  #

            branch = [int(id) for id in branch.split('.')]  # convert string to integer, branch
            idxs = [idx for idx, cmtID in enumerate(cmtIDs) if cmtID in branch]

            grid_1 = zero_padding
            for idx, line in enumerate(egrids):
                e_trans = get_eTrans_by_Index_word_entity_extended(e_trans=line, idxs=idxs, vocab_idmap=vocab_idmap)
                if len(e_trans) != 0:
                    # print e_trans
                    grid_1 = np.concatenate((grid_1, e_trans, zero_padding), axis=0)
                    # print(np.shape(grid_1))
            pos_path.append(padding_len_extended(grid_1, maxlen, w_size))
            # print(np.shape(pos_path[-1]))

        pos_doc = []
        if (len(pos_path) > maxbranch):
            for i in range(maxbranch):
                pos_doc.append(pos_path[i])
        else:
            offset = maxbranch - len(pos_path)
            for i in range(len(pos_path)):
                pos_doc.append(pos_path[i])
            for i in range(offset):
                pos_doc.append(np.zeros((maxlen), dtype=np.int))

        # pos_doc = np.vstack(pos_doc)
        pos_doc = np.column_stack(pos_doc)

        p_trees = gen_trees.gen_tree_branches(n=max(cmtIDs))
        p_count = 0
        # print(len(p_trees))

        for p_tree in p_trees:
            # print(p_tree)

            neg_path = []
            for branch in p_tree:
                branch = [int(id) for id in branch]  # convert string to integer, branch
                idxs = [idx for idx, cmtID in enumerate(cmtIDs) if cmtID in branch]

                grid_0 = zero_padding
                for idx, line in enumerate(egrids):
                    e_trans_0 = get_eTrans_by_Index_word_entity_extended(e_trans=line, idxs=idxs,
                                                                         vocab_idmap=vocab_idmap)
                    if len(e_trans_0) != 0:
                        # print e_trans
                        grid_0 = np.concatenate((grid_0, e_trans_0, zero_padding), axis=0)
                        # print(np.shape(grid_0))
                neg_path.append(padding_len_extended(grid_0, maxlen, w_size))
                # print(np.shape(neg_path[-1]))

            neg_doc = []
            if (len(neg_path) > maxbranch):
                for i in range(maxbranch):
                    neg_doc.append(neg_path[i])
            else:
                offset = maxbranch - len(neg_path)
                for i in range(len(neg_path)):
                    neg_doc.append(neg_path[i])
                for i in range(offset):
                    neg_doc.append(np.zeros((maxlen), dtype=np.int))

            # neg_doc = np.vstack(neg_doc)
            neg_doc = np.column_stack(neg_doc)
            # print(np.shape(neg_doc))
            if np.array_equal(neg_doc, pos_doc) == False:
                p_count = p_count + 1
                sentences_0.append(neg_doc)

        for i in range(0, p_count):
            sentences_1.append(pos_doc)

    vocab_info = {"vocab_x": vocab_x,
                  "vocab_idmap": vocab_idmap,
                  "idmap_vocab": idmap_vocab}
    print(np.shape(sentences_1))

    return sentences_1, sentences_0, vocab_info




