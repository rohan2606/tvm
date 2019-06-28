'''
    Generic Node Class
'''
class Node(object):

    def __init__(self, val, shape, parents=[]):
        self.val = val
        self.data_size = reduce((lambda x, y: x * y), shape) # multiply all elements in list
        # self.data_size is in bytes
        self.parents = parents
        return


    def make_parent_link(self, new_parent_node):
        self.parents.append(new_parent_node)
        return
