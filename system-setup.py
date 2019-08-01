#!/usr/bin/env python2

import sys
import os
import popen2
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "deps/readies"))
import paella

#----------------------------------------------------------------------------------------------

class RedisAISetup(paella.Setup):
    def __init__(self, nop=False):
        paella.Setup.__init__(self, nop)

    def common_first(self):
        self.setup_pip()
        self.pip_install("wheel")
        self.pip_install("setuptools --upgrade")
        
        self.install("git cmake ca-certificates curl unzip wget patchelf awscli")
        self.install("python3 python3-pip python3-venv python3-psutil python3-networkx")

    def debian_compat(self):
        self.run("curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash")
        self.install("git-lfs")
        # self.run("git lfs install")

        self.install("build-essential")
        # self.install("python-psutil")

    def redhat_compat(self):
        self.group_install("'Development Tools'")
        self.install("redhat-lsb-core")

        # uninstall and install psutil (order is important), otherwise RLTest fails
        self.run("pip uninstall -y psutil")
        self.install("python2-psutil")

    def fedora(self):
        self.group_install("'Development Tools'")

    def macosx(self):
        r, w, e = popen2.popen3('xcode-select -p')
        if r.readlines() == []:
            fatal("Xcode tools are not installed. Please run xcode-select --install.")

    def common_last(self):
        if not self.has_command("ramp"):
            self.pip3_install("git+https://github.com/RedisLabs/RAMP --upgrade")
        if not self.has_command("RLTest"):
            self.pip3_install("git+https://github.com/RedisLabsModules/RLTest.git@master")
        self.pip3_install("-r test/test_requirements.txt")

#----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Set up system for build.')
parser.add_argument('-n', '--nop', action="store_true", help='no operation')
args = parser.parse_args()

RedisAISetup(nop = args.nop).setup()
