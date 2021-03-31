#!/usr/bin/env python3

import sys
import os
import argparse

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "readies"))
READIES = os.path.join(ROOT, "opt/readies")
sys.path.insert(0, READIES)
import paella

#----------------------------------------------------------------------------------------------

class RedisAISetup(paella.Setup):
    def __init__(self, nop=False):
        paella.Setup.__init__(self, nop)

    def common_first(self):
        self.install_downloaders()
        self.pip_install("wheel")

        self.install("git unzip patchelf")
        if self.osnick != 'centos8':
            self.install("coreutils") # for realpath

    def debian_compat(self):
        self.run("%s/bin/enable-utf8" % READIES)
        self.run("%s/bin/getgcc" % READIES)
        self.install("gawk")
        self.install("libssl-dev")
        self.install("python3-regex")
        self.install("python3-networkx python3-numpy")
        if self.platform.is_arm():
            self.install("python3-dev") # python3-skimage
        self.install("libmpich-dev libopenblas-dev") # for libtorch
        self.install_git_lfs_on_linux()

    def redhat_compat(self):
        self.run("%s/bin/enable-utf8" % READIES)
        self.run("%s/bin/getepel" % READIES)
        self.install("redhat-lsb-core")

        self.run("%s/bin/getgcc --modern" % READIES)

        self.run("""
            dir=$(mktemp -d /tmp/tar.XXXXXX)
            (cd $dir; wget -q -O tar.tgz http://redismodules.s3.amazonaws.com/gnu/gnu-tar-1.32-x64-centos7.tgz; tar -xzf tar.tgz -C /; )
            rm -rf $dir
            """)

        if not self.dist == "amzn":
            self.install("epel-release")
            self.install("python3-devel libaec-devel")
        else:
            self.run("amazon-linux-extras install epel", output_on_error=True)
            self.install("python3-devel")

        self.install_git_lfs_on_linux()

    def fedora(self):
        self.run("%s/bin/getepel" % READIES)
        self.install("python3-networkx")
        self.install_git_lfs_on_linux()

    def macosx(self):
        if sh('xcode-select -p') == '':
            fatal("Xcode tools are not installed. Please run xcode-select --install.")

        self.install_gnu_utils()
        self.install("cmake")
        self.install("git-lfs")
        self.install("redis")

    def common_last(self):
        self.run("python3 -m pip uninstall -y ramp-packer RLTest || true")
        # redis-py-cluster should be installed from git due to redis-py dependency
        self.pip_install("--no-cache-dir git+https://github.com/Grokzen/redis-py-cluster.git@master")
        self.pip_install("--no-cache-dir git+https://github.com/RedisLabsModules/RLTest.git@master")
        self.pip_install("--no-cache-dir git+https://github.com/RedisLabs/RAMP@master")

        self.pip_install("-r %s/readies/paella/requirements.txt" % HERE)
        self.pip_install("-r %s/test/test_requirements.txt" % ROOT)

        self.pip_install("awscli")
        self.pip_install("mkdocs mkdocs-material mkdocs-extensions")

#----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Set up system for build.')
parser.add_argument('-n', '--nop', action="store_true", help='no operation')
args = parser.parse_args()

RedisAISetup(nop=args.nop).setup()
