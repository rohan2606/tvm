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


from graph_annotation import GraphAnnotation
from simulated_annealing import simulated_annealing


def optimize():

    # Initialize the graph
    _graph = GraphAnnotation()
    print(f"Starting at {_graph.annotations_} with function value {_graph.get_cost(_graph.annotations_)}")

    base_case_all_gpu = [ (1+0*(annotation)) for annotation in _graph.annotations_]
    print(f"Base Case of all GPUs at {base_case_all_gpu} with function value {_graph.get_cost(base_case_all_gpu)}")

    # Setup the optimization
    opt = simulated_annealing(_graph.annotations_)
    opt.steps = 100000
    # Unsure :: # since our state is just a list, slice is the fastest way to copy
    opt.copy_strategy = "slice"

    # Run the annealing
    final_annotations_, cost = opt.anneal()
    print(f"Minima found at {final_annotations_} with function value {cost}")
    return final_annotations_


if __name__ == '__main__':
    annotation = optimize()
