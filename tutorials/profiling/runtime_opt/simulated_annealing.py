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

from simanneal import Annealer
import random
from graph_annotation import GraphAnnotation


class simulated_annealing(Annealer):
    """ Optimizes graph annotation with simulated annealing"""

    # the state is equal to our annotation
    def __init__(self, state):
        self.annotation = GraphAnnotation()
        super(simulated_annealing, self).__init__(state)  # important!

    def move(self):
        # select a random node
        n = random.randint(0, len(self.state) - 1)

        # its current_state or annotation
        # curr = self.state[n]

        # TODO : add type based contraints like data being bound to CPU only
        # new annotation
        self.state[n] = random.randint(0, len(self.annotation.devices) - 1)

    def energy(self):
        return self.annotation.get_cost(self.state)



