import pickle
import re
import sys
class Node(object):
    def __init__(self, featureId, threshold, sign=0):
        self.id = None
        self.featureId = featureId
        self.threshold = threshold
        self.left = None
        self.right = None
        self.sign = sign

class Tree(object):
    def __init__(self,tree_str):
        #按照前序遍历存储二叉树
        self.root = None
        self.leaf = dict()
        tree_queue = []
        nodes = re.split(r"  +", tree_str)
        for node in nodes:
            if node == '':
                continue
            if tree_queue != []:
                present_node = tree_queue.pop()
                if present_node.left and present_node.right:
                    continue

            if 'Predict' in node:
                present_node.sign = present_node.sign + 1
                leaf_node = Node(featureId=None,threshold=None, sign=3)
                if present_node.left == None:
                    present_node.left = leaf_node
                    tree_queue.append(present_node)
                else:
                    present_node.right = leaf_node
            else:
                featureId = int(re.findall(r'feature ([0-9]*) ', node)[0])
                threshold = float(re.findall('[<>=] (.*)\)', node)[0])
                if 'If' in node:
                    if self.root == None:
                        present_node = Node(featureId, threshold, -1)
                        tree_queue.append(present_node)
                        self.root = present_node
                    else:
                        noding = Node(featureId, threshold)
                        while True:
                            if present_node.left == None:
                                present_node.left = noding
                                break
                            elif present_node.right == None:
                                present_node.right = noding
                                break
                            else:
                                present_node = tree_queue.pop()
                        tree_queue.append(present_node)
                        tree_queue.append(noding)
                elif 'Else' in node:
                    tree_queue.append(present_node)
                else:
                    continue

        self.gra_order(self.root)
        return

    def pre_order(self, root):
        if root == None:
            return ''
        print(root.featureId, root.threshold, root.sign, root.id)
        if root.left:
            self.pre_order(root.left)
        if root.right:
            self.pre_order(root.right)

    #  层次遍历
    def gra_order(self, root):
        if root == None:
            return ''
        queue = [root]
        id = 0
        while queue:
            res = []
            for item in queue:
                #print(item.featureId)
                item.id = id
                id = id+1
                if item.left:
                    res.append(item.left)
                if item.right:
                    res.append(item.right)
            queue = res


    def get_leaf(self):
        leaf_map = dict()
        return leaf_map



class GbdtModelTrees(object):
    def __init__(self, tree_string):
        tree_str_list = re.split(r'Tree [0-9]*:', tree_string.replace("\n",""))[1:]
        self.trees = self.createTrees(tree_str_list)

    def createTrees(self,tree_str_list):
        trees = []
        for tree_str in tree_str_list:
            tree = Tree(tree_str)
            trees.append(tree)
        return trees

    def getTree(self,th_tree):
        return self.trees[th_tree].root


if __name__ == "__main__":
    tree_string = pickle.load(open('tree.pk','rb'))
    gbdt_model_trees = GbdtModelTrees(tree_string)
    print(tree_string)
    a = gbdt_model_trees.getTree(29)
    a.pre_order(a.root)
    sys.exit(-1)

    tree_list = re.split(r'Tree [0-9]*:', tree_string.replace("\n",""))[1:2]
    for i in tree_list:
        print(i)
        aa = re.split(r"  +", i)
        for j in aa:
            print([j])
            featureId = re.findall(r'feature ([0-9]*) ', j)
            threshold = re.findall('[<>=] (.*)\)', j)
            pp = re.findall(r'Predict: (.*)', j)
            #print(f'featureId:{featureId}, threshold:{threshold}, pre:{pp}')
        print("------------------------------------------------")
    print(len(tree_list))


