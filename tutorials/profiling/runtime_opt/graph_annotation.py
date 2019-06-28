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

DEBUG = True

from ir_tree import ir_tree
from op_timing_database import op_timing_database
from scipy import optimize
from hill_climbing import ir_graph_tuner

'''
==================================== DEVICE to ID map =============================

CPU : 0
GPU : 1
.
.
more accelerators

'''


class graph_annotation(Object):


    def __init__(self, graph=None):

        # load the database
        self.devices = ['CPU', 'GPU']
        self.op_runtimes = [None] * len(self.devices)
        for k, device in enumerate(len(self.devices)):
            self.op_runtimes[k] = op_timing_database('../logs/', device, 'unfused')
            self.op_runtimes[k].create_database()

        # op_nodes are entirely treated as a bag of edges, where the edge information is used for communication time
        self.ir = ir_tree(graph)

        if DEBUG:
            for k in range(len(self.devices)):
                self.op_runtimes[k].assert_all_nodes_exist(_nodes)

        # annotations start at all CPU, for device mapping id check top of the file
        num_nodes = len(ir._nodes)
        self.annotations_ = [0 for i in range(num_nodes)]
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
        for node in enumerate(self.ir._nodes):
            device = self.annotations_[i] #self.get_device_in_annotation(node)
            run_time += self.op_runtimes[device].get_op_runtime(node.name)

        return run_time

    '''
        Communication time is defined as the time required to transfer data in case of a device switch, plus two corner cases
        start and end.
    '''
    def get_comm_cost(self, annotation):

        comm_time = 0.

        # General runtime O(Edge)
        for edge in self.ir._edges:
            parent, child = edge[0] , edge[1]
            parent_device, child_device  = annotation[parent], annotation[child]
            #self.get_device_in_annotation(parent), self.get_device_in_annotation(child)
            ## comm_time depends on what your parent device is
            comm_time += self.op_runtimes[parent_device].get_communication_time(parent_device, parent.data_size, child_device)

        # First Corner Case, if initial op in GPU
        for node in self.ir._top_nodes:
            device = self.get_device_in_annotation(node)
            comm_time += self.op_runtimes[0].get_communication_time(0, node.data_size, device)

        # Second Corner Case, if final op in GPU
        for node in self.ir._top_nodes:
            device = self.get_device_in_annotation(node)
            comm_time += self.op_runtimes[device].get_communication_time(device, node.data_size, 0)

        return comm_time

    def optimize(self):
        print(f"Starting at {self.annotations_} with function value {self.get_cost(self.annotations_)}")
        opt = ir_graph_tuner(self.annotations_)
        opt.steps = 100000
        # since our state is just a list, slice is the fastest way to copy
        opt.copy_strategy = "slice"
        self.annotations_, cost = opt.anneal()
        print(f"Minima found at {self.annotations_} with function value {cost}")
        
