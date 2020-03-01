#!/usr/bin/env python2

import sys
import os
import argparse

HERE=os.path.join(os.path.dirname(__file__)
sys.path.insert(0, HERE, ".."))
import paella

#----------------------------------------------------------------------------------------------

class SystemSetup(paella.Setup):
    def __init__(self, nop=False):
        paella.Setup.__init__(self, nop)

    def common_first(self):
        # self.install("")
        # self.group_install("")
        # self.setup_pip()
        # self.pip_install("")
        pass

    def debian_compat(self):
        pass

    def redhat_compat(self):
        pass

    def fedora(self):
        pass

    def macosx(self):
        pass

    def common_last(self):
        pass

#----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Set up system for build.')
parser.add_argument('-n', '--nop', action="store_true", help='no operation')
# parser.add_argument('--bool', action="store_true", help="flag")
# parser.add_argument('--int', type=int, default=1, help='number')
# parser.add_argument('--str', type=str, default='str', help='number')
args = parser.parse_args()

SystemSetup(nop = args.nop).setup()
