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


class OpRuntimeDatabase(object):

    def __init__(self, log_dir, accelerator, fusion, db_json):
        self.op_runtime_db = dict()
        self.json_db_file = log_dir + accelerator + '_' + fusion + '/' + db_json

        # bandwidth is assumed to be commutative and unq irrespective of target, source
        self.bandwidth = 2000000000.  # practically 2 # GBps or Bp(us)
        return

    '''load database from JSON file'''
    def create_database(self):
        with open(self.json_db_file) as f:
            js = json.load(f)

        for item, val in js['op_runtimes'].items():
            # if item in op_nodes:
            self.op_runtime_db[item] = float(val['mean'])

        return

    def get_op_runtime(self, op):
        if op.type == 'tvm_op':
            return self.op_runtime_db[op.name]
        else:
            return 0.

    ''' NOTE : communication time is commutative '''
    def get_communication_time(self, my_device, data_size, child_device):
        if my_device == child_device:
            return 0.
        else:
            return data_size * 64 / self.bandwidth

    def assert_all_nodes_exist(self, _nodes_map):
        for _, node in _nodes_map.items():
            if node.type == "tvm_op":
                assert (node.name in self.op_runtime_db), node.name
            else:
                assert  (node.type == "null")
