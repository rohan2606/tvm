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
**Author**: `Rohan Mukherjee <https://github.com/rohan2606>`_, \
            ``_

This is a simple optimization pass to run simulated annealing on Relay IR graphs.

A graph is considered composed of a bag of OpNodes

"""

from IRTree import IRTree

class graph_annotation(object):


    def __init__(self, graph=None):

        IRTree(graph).IRTree
        self.annotations_ = None


        return
