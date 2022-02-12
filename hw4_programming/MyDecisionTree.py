import numpy as np
import math

class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """
    def __init__(self,):
        self.is_leaf = False # whether or not the current node is a leaf node
        self.feature = None # index of the selected feature (for non-leaf node)
        self.label = -1 # class label (for leaf node)
        self.left_child = None # left child node
        self.right_child = None # right child node

class Decision_tree:
    """
    Decision tree with binary features
    """
    def __init__(self,min_entropy):
        self.min_entropy = min_entropy
        self.root = None

    def fit(self,train_x,train_y):
        # construct the decision-tree with recursion
        self.root = self.generate_tree(train_x,train_y)

    def predict(self,test_x):
        # iterate through all samples
        
        prediction = np.zeros([len(test_x),]).astype('int') # placeholder
        temp = self.root
        for i in range(len(test_x)):
            # traverse the decision-tree based on the features of the current sample
            while self.root.is_leaf == False:
                if test_x[i][self.root.feature] == 0:
                    self.root = self.root.left_child
                else: # 1
                    self.root = self.root.right_child
            prediction[i] = self.root.label
            self.root = temp

        return prediction

    def generate_tree(self,data,label):
        # initialize the current tree node
        cur_node = Tree_node()
        
        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)
        
        # determine if the current node is a leaf node
        if node_entropy < self.min_entropy:
            # determine the class label for leaf node
            counter = [0,0,0,0,0,0,0,0,0,0]
            for i in range(len(label)):
                if(label[i]==0):
                    counter[0] +=1
                elif(label[i]==1):
                    counter[1] +=1
                elif(label[i]==2):
                    counter[2] +=1
                elif(label[i]==3):
                    counter[3] +=1
                elif(label[i]==4):
                    counter[4] +=1
                elif(label[i]==5):
                    counter[5] +=1
                elif(label[i]==6):
                    counter[6] +=1
                elif(label[i]==7):
                    counter[7] +=1
                elif(label[i]==8):
                    counter[8] +=1
                elif(label[i]==9):
                    counter[9] +=1                                                                                                                    
            cur_node.label = counter.index(max(counter))
            cur_node.is_leaf = True
            return cur_node

        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data,label)
        cur_node.feature = selected_feature

        # split the data based on the selected feature and start the next level of recursion
        new_left_data = [] # 0
        new_right_data = [] # 1
        new_left_label = []
        new_right_label = []
        for i in range(len(data)):
            if data[i][cur_node.feature] == 0:
                new_left_data.append(data[i])
                new_left_label.append(label[i])
            else:
                new_right_data.append(data[i])
                new_right_label.append(label[i])
        cur_node.left_child = self.generate_tree(new_left_data,new_left_label)
        cur_node.right_child = self.generate_tree(new_right_data,new_right_label)

        return cur_node

    def select_feature(self,data,label):
        # iterate through all features and compute their corresponding entropy
        best_feat = 0
        temp = 0
        for i in range(len(data[0])):
            left_child = []
            right_child = []
            for j in range(len(data)):
                if data[j][i] == 0:
                    left_child.append(label[j])
                else:
                    right_child.append(label[j])
            # select the feature with minimum entropy
            ans = Decision_tree.compute_split_entropy(self, left_child, right_child)
            if i == 0:
                temp = ans
            else:
                if ans < temp:
                    temp = ans
                    best_feat = i
            
        return best_feat

    def compute_split_entropy(self,left_y,right_y):
        # compute the entropy of a potential split, left_y and right_y are labels for the two branches
        left_0 = 0
        left_1 = 0
        left_2 = 0
        left_3 = 0
        left_4 = 0
        left_5 = 0
        left_6 = 0
        left_7 = 0
        left_8 = 0
        left_9 = 0

        right_0 = 0
        right_1 = 0
        right_2 = 0
        right_3 = 0
        right_4 = 0
        right_5 = 0
        right_6 = 0
        right_7 = 0
        right_8 = 0
        right_9 = 0

        for i in range(len(left_y)):
            if left_y[i] == 0:
                left_0 += 1
            elif left_y[i] == 1:
                left_1 += 1
            elif left_y[i] == 2:
                left_2 += 1
            elif left_y[i] == 3:
                left_3 += 1
            elif left_y[i] == 4:
                left_4 += 1
            elif left_y[i] == 5:
                left_5 += 1
            elif left_y[i] == 6:
                left_6 += 1
            elif left_y[i] == 7:
                left_7 += 1
            elif left_y[i] == 8:
                left_8 += 1
            elif left_y[i] == 9:
                left_9 += 1

        for i in range(len(right_y)):
            if right_y[i] == 0:
                right_0 += 1
            elif right_y[i] == 1:
                right_1 += 1
            elif right_y[i] == 2:
                right_2 += 1
            elif right_y[i] == 3:
                right_3 += 1
            elif right_y[i] == 4:
                right_4 += 1
            elif right_y[i] == 5:
                right_5 += 1
            elif right_y[i] == 6:
                right_6 += 1
            elif right_y[i] == 7:
                right_7 += 1
            elif right_y[i] == 8:
                right_8 += 1
            elif right_y[i] == 9:
                right_9 += 1

        if len(left_y) == 0:
            left_entropy = 0
            left_weight = 0
        else:
            left_entropy = -(left_0/len(left_y)) * math.log2((left_0/len(left_y)) + 1e-15) - (left_1/len(left_y)) * math.log2((left_1/len(left_y)) + 1e-15) - (left_2/len(left_y)) * math.log2((left_2/len(left_y)) + 1e-15) - (left_3/len(left_y)) * math.log2((left_3/len(left_y)) + 1e-15) - (left_4/len(left_y)) * math.log2((left_4/len(left_y)) + 1e-15) - (left_5/len(left_y)) * math.log2((left_5/len(left_y)) + 1e-15) - (left_6/len(left_y)) * math.log2((left_6/len(left_y)) + 1e-15) - (left_7/len(left_y)) * math.log2((left_7/len(left_y)) + 1e-15) - (left_8/len(left_y)) * math.log2((left_8/len(left_y)) + 1e-15) - (left_9/len(left_y)) * math.log2((left_9/len(left_y)) + 1e-15)
            left_weight = len(left_y)/(len(left_y)+len(right_y))

        if len(right_y) == 0:
            right_entropy = 0
            right_weight = 0
        else:
            right_entropy = -(right_0/len(right_y)) * math.log2((right_0/len(right_y)) + 1e-15) - (right_1/len(right_y)) * math.log2((right_1/len(right_y)) + 1e-15) - (right_2/len(right_y)) * math.log2((right_2/len(right_y)) + 1e-15) - (right_3/len(right_y)) * math.log2((right_3/len(right_y)) + 1e-15) - (right_4/len(right_y)) * math.log2((right_4/len(right_y)) + 1e-15) - (right_5/len(right_y)) * math.log2((right_5/len(right_y)) + 1e-15) - (right_6/len(right_y)) * math.log2((right_6/len(right_y)) + 1e-15) - (right_7/len(right_y)) * math.log2((right_7/len(right_y)) + 1e-15) - (right_8/len(right_y)) * math.log2((right_8/len(right_y)) + 1e-15) - (right_9/len(right_y)) * math.log2((right_9/len(right_y)) + 1e-15)
            right_weight = len(right_y)/(len(left_y)+len(right_y))

        split_entropy = left_weight * left_entropy + right_weight * right_entropy

        return split_entropy

    def compute_node_entropy(self,label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)        
        count_0 = 0
        count_1 = 0
        count_2 = 0
        count_3 = 0
        count_4 = 0
        count_5 = 0
        count_6 = 0
        count_7 = 0
        count_8 = 0
        count_9 = 0

        for i in range(len(label)):
            if label[i] == 0:
                count_0 += 1
            elif label[i] == 1:
                count_1 += 1
            elif label[i] == 2:
                count_2 += 1
            elif label[i] == 3:
                count_3 += 1
            elif label[i] == 4:
                count_4 += 1
            elif label[i] == 5:
                count_5 += 1
            elif label[i] == 6:
                count_6 += 1
            elif label[i] == 7:
                count_7 += 1
            elif label[i] == 8:
                count_8 += 1
            elif label[i] == 9:
                count_9 += 1

        node_entropy = -(count_0/len(label)) * math.log2((count_0/len(label)) + 1e-15) - (count_1/len(label)) * math.log2((count_1/len(label)) + 1e-15) - (count_2/len(label)) * math.log2((count_2/len(label)) + 1e-15) - (count_3/len(label)) * math.log2((count_3/len(label)) + 1e-15) - (count_4/len(label)) * math.log2((count_4/len(label)) + 1e-15) - (count_5/len(label)) * math.log2((count_5/len(label)) + 1e-15) - (count_6/len(label)) * math.log2((count_6/len(label)) + 1e-15) - (count_7/len(label)) * math.log2((count_7/len(label)) + 1e-15) - (count_8/len(label)) * math.log2((count_8/len(label)) + 1e-15) - (count_9/len(label)) * math.log2((count_9/len(label)) + 1e-15)
        
        return node_entropy
