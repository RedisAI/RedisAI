import subprocess
from RLTest import Env
from includes import *
import shutil
import argparse
import signal
from redis import RedisError

terminate_flag = 0
parent_pid = os.getpid()


# this should capture user SIGINT signals (such as keyboard ctrl-c). Since we are using multi-processing,
# this handler will be inherited by all the running processes. Note that every process will get the signal,
# as all of them are at the same group.
def handler(signum, frame):
    global terminate_flag
    terminate_flag = 1
    global parent_pid
    if os.getpid() == parent_pid:  # print it only once
        print("\nReceived user interrupt. Shutting down...")


def _exit():
    # remove the logs that were auto generated by redis
    shutil.rmtree('logs', ignore_errors=True)
    sys.exit(1)


def run_benchmark(env, num_runs_mnist, num_runs_inception, num_runs_bert, num_parallel_clients):
    global terminate_flag
    con = get_connection(env, '{1}')

    print("Loading ONNX models...")
    model_pb = load_file_content('mnist.onnx')
    sample_raw = load_file_content('one.raw')
    inception_pb = load_file_content('inception-v2-9.onnx')
    _, _, _, _, img = load_mobilenet_v2_test_data()
    bert_pb = load_file_content('bert-base-cased.onnx')
    bert_in_data = np.random.randint(-2, 1, size=(10, 100), dtype=np.int64)

    for i in range(50):
        if terminate_flag == 1:
            _exit()
        ret = con.execute_command('AI.MODELSTORE', 'mnist{1}'+str(i), 'ONNX', DEVICE, 'BLOB', model_pb)
        env.assertEqual(ret, b'OK')
    con.execute_command('AI.TENSORSET', 'mnist_in{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)

    for i in range(20):
        if terminate_flag == 1:
            _exit()
        ret = con.execute_command('AI.MODELSTORE', 'inception{1}'+str(i), 'ONNX', DEVICE, 'BLOB', inception_pb)
        env.assertEqual(ret, b'OK')

    backends_info = get_info_section(con, 'backends_info')
    print(f'Done. ONNX memory consumption is: {backends_info["ai_onnxruntime_memory"]} bytes')

    ret = con.execute_command('AI.TENSORSET', 'inception_in{1}', 'FLOAT', 1, 3, 224, 224, 'BLOB', img.tobytes())
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.MODELSTORE', 'bert{1}', 'ONNX', DEVICE, 'BLOB', bert_pb)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'bert_in{1}', 'INT64', 10, 100, 'BLOB', bert_in_data.tobytes())
    env.assertEqual(ret, b'OK')

    def run_parallel_onnx_sessions(con, i, model, input, num_runs):
        for _ in range(num_runs):
            if terminate_flag == 1:
                return
            # If the user is terminating the benchmark, redis-server will receive a termination signal as well, and
            # RedisError exception will thrown (and caught)
            try:
                if model == 'bert{1}':
                    ret = con.execute_command('AI.MODELEXECUTE', model, 'INPUTS', 3, input, input, input,
                                              'OUTPUTS', 2, 'res{1}', 'res2{1}')
                else:
                    ret = con.execute_command('AI.MODELEXECUTE', model, 'INPUTS', 1, input, 'OUTPUTS', 1, 'res{1}')
                env.assertEqual(ret, b'OK')
            except RedisError:
                return

    def run_mnist():
        run_test_multiproc(env, '{1}', num_parallel_clients, run_parallel_onnx_sessions,
                           ('mnist{1}0', 'mnist_in{1}', num_runs_mnist))

    def run_bert():
        run_test_multiproc(env, '{1}', num_parallel_clients, run_parallel_onnx_sessions,
                           ('bert{1}', 'bert_in{1}', num_runs_bert))

    # run only mnist
    mnist_total_requests_count = num_runs_mnist*num_parallel_clients
    print(f'\nRunning {num_runs_mnist} consecutive executions of mnist from {num_parallel_clients} parallel clients...')
    start_time = time.time()
    run_test_multiproc(env, '{1}', num_parallel_clients, run_parallel_onnx_sessions,
                       ('mnist{1}0', 'mnist_in{1}', num_runs_mnist))
    if terminate_flag == 1:
        _exit()
    print(f'Done. Total execution time for {mnist_total_requests_count} requests: {time.time()-start_time} seconds')
    mnist_time = con.execute_command('AI.INFO', 'mnist{1}0')[11]
    print("Average serving time per mnist run session is: {} seconds"
          .format(float(mnist_time)/1000000/mnist_total_requests_count))

    # run only inception
    inception_total_requests_count = num_runs_inception*num_parallel_clients
    print(f'\nRunning {num_runs_inception} consecutive executions of inception from {num_parallel_clients} parallel clients...')
    start_time = time.time()
    run_test_multiproc(env, '{1}', num_parallel_clients, run_parallel_onnx_sessions,
                       ('inception{1}0', 'inception_in{1}', num_runs_inception))
    if terminate_flag == 1:
        _exit()
    print(f'Done. Total execution time for {inception_total_requests_count} requests: {time.time()-start_time} seconds')
    inception_time = con.execute_command('AI.INFO', 'inception{1}0')[11]
    print("Average serving time per inception run session is: {} seconds"
          .format(float(inception_time)/1000000/inception_total_requests_count))

    # run only bert
    bert_total_requests_count = num_runs_bert*num_parallel_clients
    print(f'\nRunning {num_runs_bert} consecutive executions of bert from {num_parallel_clients} parallel clients...')
    start_time = time.time()
    run_test_multiproc(env, '{1}', num_parallel_clients, run_parallel_onnx_sessions, ('bert{1}', 'bert_in{1}', num_runs_bert))
    if terminate_flag == 1:
        _exit()
    print(f'Done. Total execution time for {bert_total_requests_count} requests: {time.time()-start_time} seconds')
    bert_time = con.execute_command('AI.INFO', 'bert{1}')[11]
    print("Average server time per bert run session is: {} seconds"
          .format(float(bert_time)/1000000/bert_total_requests_count))

    con.execute_command('AI.INFO', 'mnist{1}0', 'RESETSTAT')
    con.execute_command('AI.INFO', 'inception{1}0', 'RESETSTAT')
    con.execute_command('AI.INFO', 'bert{1}', 'RESETSTAT')

    # run all 3 models in parallel
    total_requests_count = mnist_total_requests_count+inception_total_requests_count+bert_total_requests_count
    print(f'\nRunning requests for all 3 models from {3*num_parallel_clients} parallel clients...')
    start_time = time.time()
    t = threading.Thread(target=run_mnist)
    t.start()
    t2 = threading.Thread(target=run_bert)
    t2.start()
    run_test_multiproc(env, '{1}', num_parallel_clients, run_parallel_onnx_sessions,
                       ('inception{1}0', 'inception_in{1}', num_runs_inception))
    t.join()
    t2.join()
    if terminate_flag == 1:
        _exit()
    print(f'Done. Total execution time for {total_requests_count} requests: {time.time()-start_time} seconds')
    mnist_info = con.execute_command('AI.INFO', 'mnist{1}0')[11]
    inception_info = con.execute_command('AI.INFO', 'inception{1}0')[11]
    bert_info = con.execute_command('AI.INFO', 'bert{1}')[11]
    total_time = mnist_info+inception_info+bert_info
    print("Average serving time per run session is: {} seconds"
          .format(float(total_time)/1000000/total_requests_count))


if __name__ == '__main__':

    # set a handler for user interrupt signal
    signal.signal(signal.SIGINT, handler)

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", default='1',
                        help='The number of RedisAI working threads that can execute sessions in parallel')
    parser.add_argument("--num_runs_mnist", type=int, default=500,
                        help='The number of requests per client that is running mnist run sessions')
    parser.add_argument("--num_runs_inception", type=int, default=50,
                        help='The number of requests per client that is running inception run sessions')
    parser.add_argument("--num_runs_bert", type=int, default=5,
                        help='The number of requests per client that is running bert run sessions')
    parser.add_argument("--num_parallel_clients", type=int, default=20,
                        help='The number of parallel clients that send consecutive run requests per model')
    args = parser.parse_args()

    if not os.path.isdir('tests/flow/test_data'):
        print("Downloading tests data from s3...")
        with open('tests/flow/test_data_files.txt') as f:
            for resource in f.readlines():
                subprocess.call(['wget', '-q', '-x', '-nH', '--cut-dirs=2', resource[:-1], '-P', 'tests/flow/test_data'])
        print("Done")

    terminate_flag = 0
    print(f'Running ONNX benchmark on RedisAI, using {args.num_threads} working threads')
    env = Env(module='install-cpu/redisai.so',
              moduleArgs='MODEL_EXECUTION_TIMEOUT 50000 THREADS_PER_QUEUE '+args.num_threads, logDir='logs')

    # If the user is terminating the benchmark, redis-server will receive a termination signal as well, and
    # RedisError exception will thrown (and caught)
    try:
        run_benchmark(env, num_runs_mnist=args.num_runs_mnist, num_runs_inception=args.num_runs_inception,
                    num_runs_bert=args.num_runs_bert, num_parallel_clients=args.num_parallel_clients)
        env.stop()
    except RedisError as e:
        pass
    finally:
        # remove the logs that were auto generated by redis
        shutil.rmtree('logs', ignore_errors=True)
