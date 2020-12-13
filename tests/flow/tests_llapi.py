import redis

from includes import *
import os

'''
python -m RLTest --test tests_llapi.py --module path/to/redisai.so
'''
goal_dir = os.path.join(os.getcwd(), "../module/LLAPI.so")
TEST_MODULE_PATH = os.path.abspath(goal_dir)

def test_basic_check(env):

    con = env.getConnection()
    ret = con.execute_command("MODULE", "LOAD", TEST_MODULE_PATH)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command("RAI_llapi.basic_check")
    env.assertEqual(ret, b'OK')
