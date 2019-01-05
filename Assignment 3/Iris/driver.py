from DecisionTree import *
import pandas as pd
from sklearn import model_selection
import itertools
from sklearn import datasets as d_sets
import numpy as np



def powerset(x):
    lst = list(x)
    return itertools.chain.from_iterable(itertools.combinations(lst, r) for r in range(len(lst) + 1))


def compute_prune(nodes, t, test):
    best_prune_nodes = 0
    best_pruning_accuracy = 0
    for item in powerset(nodes):
        print("\n**************************************\n")
        print("Pruning Nodes: ", item , "\n")
        t_pruned = prune_tree(t, item)
        print("\n*************Tree after pruning*******\n")
        print_tree(t_pruned)

        acc = computeAccuracy(test, t)
        print("Accuracy on test = ", str(acc), "\n")
        if acc > best_pruning_accuracy:
            best_prune_nodes = item
            best_pruning_accuracy = acc

    print("Maximum post pruning accuracy:", best_pruning_accuracy, "obtained on nodes:", best_prune_nodes)

def main():
    '''
    dataset = d_sets.load_iris()
    df = pd.DataFrame(data = np.c_[dataset['data'], dataset['target']], columns = dataset['feature_names'] + ['class'])
    #df = pd.read_csv(dataset.data, header=None, names=names)
    lst = df.values.tolist()
    header = dataset['feature_names'] + ['class']
    '''

    header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
    lst = df.values.tolist()


    t = build_tree(lst, header)
    print_tree(t)

    print("\n**************** Leaf nodes ******************\n")
    leaves = getLeafNodes(t)
    leaf_nodes = []
    for leaf in leaves:
        leaf_nodes.append(leaf.id)
        print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))

    print("\n*************** Non-leaf nodes ****************\n")
    innerNodes = getInnerNodes(t)
    inner_nodes_list = []
    for inner in innerNodes:
        inner_nodes_list.append(inner.id)
        print("id = " + str(inner.id) + " depth =" + str(inner.depth))

    trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
    train = trainDF.values.tolist()
    test = testDF.values.tolist()

    t = build_tree(train, header)
    print("\n*************Tree before pruning*******\n")

    print_tree(t)
    acc = computeAccuracy(test, t)
    print("Accuracy on test = " + str(acc))
    print ("\n******************************************\n")


    ## TODO: You have to decide on a pruning strategy
    #t_pruned = prune_tree(t, [26, 11, 5])

    parent_nodes = []

    #Prune nodes 1 level above the leaf
    print("\n********Pruning Nodes on Level N-1********\n")
    for index in range(len(leaf_nodes)):
        temp = leaf_nodes[index]
        parent_nodes.append(get_parent(temp))

    compute_prune(parent_nodes, t, test)



if __name__ == '__main__':
    main()