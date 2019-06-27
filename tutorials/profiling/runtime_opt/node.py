'''
    Generic Node Class
'''
class Node(object):

    def __init__(self, val, children=[], parents=[]):
        self.val = val
        self.children = children
        return

    def make_child_link(self, new_child_node):
        self.children.append(new_child_node)
        return
