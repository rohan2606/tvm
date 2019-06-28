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


'''
Loads data from a previously collected database of op runtimes for a particular device in a particular hardware

## # BUG: the current code loads op runtime database everytime, plus does it serially via JSON
'''
class op_timing_database(Object):

    def __init__(self, logdir, accelerator, fusion, db_json):
        self.op_runtime_db = dict()
        self.json_db_file = logdir + accelerator + '_' + fusion + '/' + db_json

        ## bandwidth is assumed to be commutative and unq irrespective of target, source
        self.bandwidth = 2. # GBps or Bp(us)
        return


    '''
    load database from JSON file
    '''
    def create_database(self):
        with open(self.json_db_file) as f:
            js = json.load(f)

        for item, val in js['op_runtimes'].items():
            # if item in op_nodes:
            self.op_runtime_db[item] = val['mean']

        return

    def get_op_runtime(self, op_name):
        return self.op_runtime_db[op_name]


    '''
    NOTE : communication time is commutative
    '''
    def get_communication_time(self, my_device, data_size, child_device):

        if my_device == child_device:
            return 0.0
        else:
            return data_size/self.bandwidth


    def assert_all_nodes_exist(self, _nodes):
        for node in _nodes:
            assert(node in self.op_runtime_db)
