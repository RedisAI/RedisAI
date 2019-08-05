#!/usr/bin/env python3

import sys
import os
from subprocess import Popen, PIPE
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "deps/readies"))
import paella

#----------------------------------------------------------------------------------------------

class RedisAISetup(paella.Setup):
    def __init__(self, nop=False):
        paella.Setup.__init__(self, nop)

    def common_first(self):
        self.setup_pip()
        self.pip3_install("wheel")
        self.pip3_install("setuptools --upgrade")
        
        self.install("git cmake ca-certificates curl unzip wget patchelf awscli")

    def debian_compat(self):
        self.run("curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash")
        self.install("git-lfs")
        self.run("git lfs install")

        self.install("build-essential")
        self.install("python3 python3-pip python3-venv python3-psutil python3-networkx")

    def redhat_compat(self):
        self.group_install("'Development Tools'")
        self.install("redhat-lsb-core")

        self.install("epel-release")
        self.install("python36 python36-pip")
        self.install("python36-psutil")

    def fedora(self):
        self.group_install("'Development Tools'")

    def macosx(self):
        p = Popen('xcode-select -p', stdout=PIPE, close_fds=True, shell=True)
        if p.stdout.readlines() == []:
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
