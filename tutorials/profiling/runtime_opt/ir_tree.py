# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _graph_annotation_pass:

====================
**Author**: `Rohan Mukherjee <https://github.com/rohan2606>`_,
             Amazon, AWS <mukrohan@amazon.com>


"""


import json
from node import Node
from functools import reduce

from annotation_pass import DEBUG


'''
Reads a graph compiled by relay and constructs a tree data structure out of it. Return the parent nodes.
'''
class ir_tree(Object):


    '''
    for debug mode, start reading graph directly from './debug_graph.json'
    for deployment, graph is passed as a graph JSON object

    finally convert that into Tree
    '''
    def __init__(self, graph=None):
        if graph is None:
            assert(DEBUG==True), 'reading from json disabled in non-debug mode'
            print('Running in Debug Mode, reading graph from ./debug_graph.json')
            graph = self.read_json('debug_graph.json')
        else:
            assert(DEBUG==False), 'debug mode not active in deployment mode'

        assert type(graph)==dict, 'graph format mismatch'

        # initialize empty containers repreenting the tree
        self._nodes = set()
        self._edges = set()
        self.symbolic_edges = set()
        self._leaf_nodes = set()
        self._top_nodes = set()

        # parse the graph to construvt the tree
        self.convert2Tree(graph)
        return

    '''
    simple json reader, function bypassed if not used in debug mode
    '''
    def read_json(self, file_name):
        try:
            with open() as f:
                js = json.load(f)
            return js
        except:
            print('No such file as debug_graph.json detected. Turn debug mode off if reqd')



    '''
    reads json graph from relay, converts to tree
    '''
    def convert2Tree(self, graph):

        # node id from relay is equal to its position in the list, reused to parse child-parent relationship
        # node_id_map maps from input node id to node name
        node_map = dict()

        # top level nodes mean the nodes that have no parent, it can be more than one
        self._leaf_nodes = set()

        if DEBUG:
            print('Number of nodes in graph :: ' + str(len(graph['nodes'])))

        for node_id, (node_vals, shape) in enumerate(zip(graph['nodes'], graph['shape'][1])):
            # get the node vals
            # node_type is either a tvm_op or null
            # node_name is the unq key
            node_type = node_vals['op']
            assert node_type == None or node_type == 'tvm_op', 'wrong type detected'
            node_name = node_vals['name']
            input_ids = node_vals['inputs']

            _node = Node(node_name, shape)
            # add Node to the mapping dictionary
            node_map[node_id] = _node

            # add Node to the leaf nodes temporarily, will be pruned if needed
            self._leaf_nodes.add(node_id)

            # add Node to top_nodes if they have no inputs
            if len(input_ids) == 0:
                self._top_nodes.add(node_id)

            # add to global list of nodes
            self._nodes.add(node_id)


            #make symbolic links to parent
            for parent_id in input_ids:
                parent_id = parent_id[0] # some pecularity in relay graph format, remaining items are always 0
                parent = node_map[parent_id] # get the parent node
                _node.make_parent_link(parent)
                # self._edges.add((parent, child))
                self._edges.add((parent_id, node_id)) #parent, child))
                if parent in self._leaf_nodes:
                    self._leaf_nodes.remove(parent_id)


        return





    '''
        Traverse the graph following the edges starting from the top level nodes.
    '''
    def bottom_up_edge_traversal(self, leaf_nodes=self._leaf_nodes, visited=set(), traversal=[]):

        new_leaves = set()

        for leaf in leaf_nodes:
            if leaf.val not in visited:
                visited.add(leaf.val)
                for parent in leaf.parents:
                    traversal.append((parent.val , leaf.val))
                    if parent.val not in visited:
                        visited.add(parent.val)
                        new_leaves.add(parent)

        if len(new_leaves) == 0:
            return traversal
        else:
            return self.bottom_up_edge_traversal(new_leaves, visited, traversal)



    '''
        Visualize the graph by plotting edge traversal with a dot script
    '''
    def view_graph(self):
        from graphviz import Digraph
        g = Digraph('G', filename='op_graph.gv')
        edges = self.bottom_up_edge_traversal()
        for edge in edges:
            parent = edge[0]
            child = edge[1]
            g.edge(parent, child)

        g.view()
        return
