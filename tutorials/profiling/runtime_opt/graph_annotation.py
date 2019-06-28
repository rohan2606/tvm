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

This is a simple optimization pass to run simulated annealing on Relay IR graphs.

A graph is considered composed of a bag of OpNodes

"""
from ir_tree import IrTree
from op_runtime_database import OpRuntimeDatabase


global DEBUG
DEBUG = True


'''
==================================== DEVICE to ID map =============================

CPU : 0
GPU : 1
.
.
more accelerators

'''


class GraphAnnotation(object):

    def __init__(self, graph=None):

        # load the database
        self.devices = ['CPU', 'GPU']
        self.op_runtimes = [None] * len(self.devices)
        for k, device in enumerate(self.devices):
            self.op_runtimes[k] = OpRuntimeDatabase('../logs/', device, 'unfused', 'op_runtimes.json')
            self.op_runtimes[k].create_database()

        # op_nodes are entirely treated as a bag of edges, where the edge information is used for communication time
        self.ir = IrTree(graph)

        if DEBUG:
            for k in range(len(self.devices)):
                self.op_runtimes[k].assert_all_nodes_exist(self.ir._nodeId_to_node_map)

        # annotations start at all CPU, for device mapping id check top of the file
        num_nodes = len(self.ir._nodes)
        self.annotations_ = [0 for _ in range(num_nodes)]
        return



    # '''
    #     given a node name (guaranteed to be unq) get its current allocation/annotation
    # '''
    # def get_device_in_annotation(self, node):
    #     return self.annotations_[node.val]
    #


    '''
        Final time is sum of op runtimes and communication overhead
    '''
    def get_cost(self, annotation):
        return self.get_op_cost(annotation) + self.get_comm_cost(annotation)



    '''
        Op Cost is independent sum of op runtimes
    '''
    def get_op_cost(self, annotation):

        run_time = 0.
        # General runtime O(Nodes)
        for node_id in range(len(self.ir._nodes)):
            device = annotation[node_id]  # self.get_device_in_annotation(node)
            node = self.ir._nodeId_to_node_map[node_id]
            run_time += self.op_runtimes[device].get_op_runtime(node)

        return run_time

    '''
        Communication time is defined as the time required to transfer data in case of a device switch, 
        plus two corner cases for start and end.
    '''
    def get_comm_cost(self, annotation):

        comm_time = 0.
        # General runtime O(Edge)
        for edge in self.ir._edges:
            parent_id, child_id = edge[0], edge[1]
            parent_node = self.ir._nodeId_to_node_map[parent_id]
            parent_device, child_device = annotation[parent_id], annotation[child_id]
            # self.get_device_in_annotation(parent), self.get_device_in_annotation(child)
            # comm_time depends on what your parent device is
            comm_time += self.op_runtimes[parent_device].get_communication_time(parent_device, parent_node.data_size, child_device)

        # First Corner Case, if initial op in GPU
        for node_id in self.ir._top_nodes:
            device = annotation[node_id] # self.get_device_in_annotation(node)
            node = self.ir._nodeId_to_node_map[node_id]
            comm_time += self.op_runtimes[0].get_communication_time(0, node.data_size, device)

        # Second Corner Case, if final op in GPU
        for node_id in self.ir._top_nodes:
            device = annotation[node_id] # self.get_device_in_annotation(node)
            node = self.ir._nodeId_to_node_map[node_id]
            comm_time += self.op_runtimes[device].get_communication_time(device, node.data_size, 0)

        return comm_time


        
