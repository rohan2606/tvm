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

    Generic Node Class

"""
from functools import reduce


class Node(object):

    def __init__(self, name, _type, shape, parents=None):

        if parents is None:
            parents = list()

        self.name = self.parse_name(name)
        self.shape = shape
        self.data_size = reduce((lambda x, y: x * y), shape)  # multiply all elements in list


        self.type = _type

        # self.data_size is in bytes
        self.parents = parents
        return

    def make_parent_link(self, new_parent_node):
        self.parents.append(new_parent_node)
        return

    def parse_name(self, name):
        op_list = name.strip().split('_')
        if op_list[-1].isdigit():
            op_list = op_list[:-1]

        return '_'.join(op_list)

    def parse_type(self, type, name):
        if type == 'tvm_op':
            return type
        elif type == 'null' and name == "data":
            return "data"
        elif type[0] == 'p' and type[1:].digit():
            return "param"
        else:
            raise AttributeError


    def update_name_with_shapes(self):
        # Please make sure you have all your parent and shape information parsed already

        input_shapes = []
        for parent in self.parents:
            input_shapes.append(parent.shape)

        if len(input_shapes) > 0:
            self.name += '_' + str(input_shapes)

        output_shape = self.shape
        self.name += '_' + str(output_shape)
        return





