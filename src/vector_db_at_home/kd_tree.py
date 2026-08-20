import numpy as np

# I found something that says this doesn't work
# above about 20 dimensions


# one super simple algorithm I found
# split_dim = depth mod k
# split_val = median of that dimension
#
# I think the proper algorithm is to split
# on the dimension that has the largest range
# like how decision trees split on the dimension
# with the largest information gain
#
# that doesn't really make sense to me because
# I'd have to know all the points ahead of time
# I guess I could have a rebalancing operation or something?
# idk yeah it seems like k-d trees are built "offline"
#
# another similar structure is a ball tree but I don't think
# it will work much better than the k-d tree
# https://steveomohundro.com/wp-content/uploads/2009/03/omohundro89_five_balltree_construction_algorithms.pdf
# this paper lists 2 "online" algorithms for constructing the tree 1 node at a time
# that paper is actually really good and helpful
#
# the other problem is that storing trees in relational databases is famously difficult
# maybe I just pickle it and store the bytes
# or dump it to JSON and store those bytes
# something like that
# then I have to rewrite that blob every time I change anything
# which is probably fine I guess
# other option for this data is probably a parent/child table
# as in 2 columns, parent and chile, with node ids
# and then
#
# wikipedia is hyping up something called an M-tree
# it says it has to use a distance function that satisfies the triangle inequality
# which euclidean distance does
# but other things might not. like cosine similarity probably doesn't
# https://en.wikipedia.org/wiki/M-tree
#
#
#
# another quote about this not really working
# https://en.wikipedia.org/wiki/K-d_tree#Degradation_in_performance_with_high-dimensional_data
# In high-dimensional spaces, the curse of dimensionality causes the algorithm to need to visit many more branches than in lower-dimensional spaces. In particular, when the number of points is only slightly higher than the number of dimensions, the algorithm is only slightly better than a linear search of all of the points. As a general rule, if the dimensionality is k, the number of points in the data, n, should be n ≫ 2 k {\displaystyle n\gg 2^{k}}. Otherwise, when k-d trees are used with high-dimensional data, most of the points in the tree will be evaluated and the efficiency is no better than exhaustive search,[14] and, if a good-enough fast answer is required, approximate nearest-neighbour methods should be used instead.
#
# also yeah I guess this isn't even approximate nearest neighbors
# we're still going to find the correct answer
# unless we make the tree really shallow and only return results from the leaf node we land in
# and don't do any of the backtracking to search neighboring nodes


class KDTreeNode:
    def __init__(self, split_dim: int | None, split_val: int | None, points: list):
        self.split_dim = split_dim
        self.split_val = split_val
        self.points = points
        self.l: KDTreeNode = None
        self.r: KDTreeNode = None


class KDTree:
    def __init__(self, k: int, max_points_leaf: int = 10):
        self.k = k
        self.max_points_leaf = max_points_leaf
        self.root = KDTreeNode(None, None, [])

    def insert_rec(self, root: KDTreeNode, point: np.ndarray):
        # if the node doesn't have split_dim/split_val or points,
        # this is the first point we're inserting
        if root.split_dim is None and root.split_val is None and len(root.points) == 0:
            return KDTreeNode(None, None, [point])

        # if we've hit something that has points,
        # check whether inserting would increase us above max_points_leaf
        # and then split on the median of the dimension with the largest range
        # or idk maybe split on max - (max - min)/2
        # and replace the current node with a splitting node
        # and two leaf nodes
        if len(root.points) > 0:
            if len(root.points) < self.max_points_leaf:
                root.points.append(point)
                return root

            # TODO: split and add two leaf nodes
            return

        # if the root has no points,
        # search down based on split_dim and split_val
        val = point[root.split_dim]
        if val <= root.split_val:
            return self.insert_rec(root.l, point)
        return self.insert_rec(root.r, point)

    def insert(self, point: np.ndarray):
        self.root = self.insert_rec(self.root, point)
