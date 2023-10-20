import os
import re
import argparse
import subprocess
import time
import psutil
import pandas as pd

from pathlib import Path
from datasets import load_dataset

import json

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_test_data_name', default='code_test_data_palm.jsonl', type=str) # will read the submission dataset that is to be evaluated from './dataset/{code_test_data_name}'
    parser.add_argument('--codes_dir_name', default='palm_codes', type=str) # same as --codes_dir_name in save_codes. source codes files to be evaluated should be stored at './codes/{args.codes_dir_name}/{lang_cluster}/{code_uid}
    parser.add_argument('--temp_save_name', default='palm_data', type=str) # the converted dataset(add the is_passed and performance metrics for each eubmission data) will be placed at './dataset/{temp_save_name}'
    args = parser.parse_args()

    return args


def count_memory_and_time(command, input=None):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, shell=True)
    # print(process)
    process.stdin.write(input)
    process.stdin.flush()

    cpu_time = 0
    peak_memory = 0
    timeout_flag = False
    # start_time = time.time()
    timeout_cnt = 0
    while process.poll() is None:
        try:
            p = psutil.Process(process.pid)
            current_memory = p.memory_full_info().uss / 1024
            cpu_time = float(sum(p.cpu_times()[:4]))
            # print(p.pid)
        except:
            # print('进程已经结束了')
            current_memory = 0
        # print(f'当前内存: {current_memory}KB')
        if current_memory > peak_memory:
            peak_memory = current_memory
        time.sleep(0.0002)
        # current_time = time.time()
        # elapsed_time = current_time - start_time
        # if elapsed_time > 7:
        timeout_cnt += 1
        if timeout_cnt > 25000: # 5s
            print("Time limit exceeded!")
            process.kill()
            timeout_flag = True
            break
    # stdout = process.stdout.readline()
    # stderr = process.stderr.readline()
    # print('程序预期输出:', stdout)
    # print('程序运行错误:', stderr)
    # print(process)
    # process.wait(timeout=20)

    # print(f'程序占用的内存峰值: {peak_memory}KB, CPU耗时: {cpu_time}s')

    return peak_memory, cpu_time, timeout_flag


def execute_command(command, input=None):
    if input is not None:
        input = input.replace('\r\n', '\n')
    try:
        # References: https://stackoverflow.com/questions/66480855/python-subprocess-run-timeout-behaving-differently-on-windows-vs-linux
        outcome = subprocess.run(command, input=input, capture_output=True, text=True, timeout=20, shell=True)
    except Exception as e:
        print('Error occurred while executing command:', e)
        outcome = subprocess.CompletedProcess(args=command, returncode=-1, stdout='', stderr=str(e))
    return outcome


def add_passrate_perfmetrcs(example):
    src_uid = example['src_uid']
    lang = example['lang']
    lang_cluster = example['lang_cluster']
    code_uid = example['code_uid']
    source_code = example['source_code']
    hidden_unit_tests = eval(example['hidden_unit_tests'])# example['unittests']
    num_hidden_unit_tests = len(hidden_unit_tests)
    testcases_perf = []
    # LLM failed to generate hidden unit tests
    # if code_uid == '43c115a0548a290a0fa114a2f66e4ab0':
    #     example['pass_rate'] = 0
    #     example['mean_cpu_time'] = 0
    #     example['mean_peak_mem'] = 0
    #     return example

    if num_hidden_unit_tests == 0:
        print('Failed to generate hidden unit tests:', code_uid)
        example['pass_rate'] = 0.00
    else:
        if lang_cluster == 'C':
            os.chdir(f'./codes/{args.codes_dir_name}/c/{code_uid}')
            print(os.getcwd())

            compile_command = 'gcc -fno-optimize-sibling-calls -w -fno-strict-aliasing -DONLINE_JUDGE -include limits.h -fno-asm -s -O2 -DONLINE_JUDGE -include math.h -static -lm -o code code.c'#'gcc -fPIC -O0 code.c -o code'
            outcome = execute_command(compile_command)
            # print(outcome)

            num_passed = 0
            for index, hidden_unit_test in enumerate(hidden_unit_tests):
                input = hidden_unit_test['input']
                output = hidden_unit_test['output'][0]

                test_command = './code'
                outcome = execute_command(test_command, input)
                # print(outcome)
                is_passed = True if outcome.returncode == 0 and (
                        outcome.stdout in output 
                        or outcome.stdout.rstrip() in output 
                        or outcome.stdout.replace('\n','\r\n') in output 
                        or outcome.stdout.replace('\n', '\r\n').rstrip() in output
                        ) else False
                if is_passed is True:
                    num_passed += 1
                # print(is_passed)

                peak_mem, cpu_time, timeout_flag = count_memory_and_time(test_command, input)
                if timeout_flag == True:
                    # if code_uid in ['1d3a8804e288dee710091aedda77299c']:# 这些题目有问题，计算perf时会运行超时一直不退出（可能是死锁）
                    timeout_code_uids.append(code_uid)
                    example['pass_rate'] = 0
                    example['mean_cpu_time'] = 0
                    example['mean_peak_mem'] = 0
                    os.chdir('../../../..')
                    return example
                testcases_perf.append([
                    src_uid, lang, code_uid, input, output, is_passed, cpu_time, peak_mem
                ])
            testcases_perf_df = pd.DataFrame(testcases_perf, columns=['src_uid','lang','code_uid','test_input','test_output','is_passed','cpu_time','peak_mem'])
            testcases_perf_df.to_csv("./perf.csv", index=False)# stored the testcase-level performance of the code under its code_uid dir
            pass_rate = round(100. * num_passed / num_hidden_unit_tests, 2)
            print(f'Pass rate: {pass_rate}% [{num_passed}/{num_hidden_unit_tests}]')

            os.chdir('../../../..')
            # print(os.getcwd())

            example['pass_rate'] = pass_rate
            example['mean_cpu_time'] = testcases_perf_df['cpu_time'].mean()
            example['mean_peak_mem'] = testcases_perf_df['peak_mem'].mean()

        elif lang_cluster == 'C++':
            os.chdir(f'codes/{args.codes_dir_name}/cpp/{code_uid}')
            print(os.getcwd())

            compile_command = 'g++ -s -x c++ -O2 -w -DONLINE_JUDGE -include math.h -include limits.h -static -lm -o code code.cpp'# 'g++ -fPIC -O0 code.cpp -o code'
            outcome = execute_command(compile_command)
            # print(outcome)

            num_passed = 0
            for index, hidden_unit_test in enumerate(hidden_unit_tests):
                input = hidden_unit_test['input']
                output = hidden_unit_test['output'][0]

                test_command = './code'
                outcome = execute_command(test_command, input)
                # print(outcome)

                is_passed = True if outcome.returncode == 0 and (
                        outcome.stdout in output 
                        or outcome.stdout.rstrip() in output 
                        or outcome.stdout.replace('\n','\r\n') in output 
                        or outcome.stdout.replace('\n', '\r\n').rstrip() in output
                        ) else False
                if is_passed is True:
                    num_passed += 1
                # print(is_passed)

                peak_mem, cpu_time, timeout_flag = count_memory_and_time(test_command, input)
                if timeout_flag == True:
                    # if code_uid in ['1d3a8804e288dee710091aedda77299c']:# 这些题目有问题，计算perf时会运行超时一直不退出（可能是死锁）
                    timeout_code_uids.append(code_uid)
                    example['pass_rate'] = 0
                    example['mean_cpu_time'] = 0
                    example['mean_peak_mem'] = 0
                    os.chdir('../../../..')
                    return example
                testcases_perf.append([
                    src_uid, lang, code_uid, input, output, is_passed, cpu_time, peak_mem
                ])
            testcases_perf_df = pd.DataFrame(testcases_perf, columns=['src_uid','lang','code_uid','test_input','test_output','is_passed','cpu_time','peak_mem'])
            testcases_perf_df.to_csv("./perf.csv", index=False)# stored the testcase-level performance of the code under its code_uid dir
            pass_rate = round(100. * num_passed / num_hidden_unit_tests, 2)
            print(f'Pass rate: {pass_rate}% [{num_passed}/{num_hidden_unit_tests}]')

            os.chdir('../../../..')
            # print(os.getcwd())

            example['pass_rate'] = pass_rate
            example['mean_cpu_time'] = testcases_perf_df['cpu_time'].mean()
            example['mean_peak_mem'] = testcases_perf_df['peak_mem'].mean()
        elif lang_cluster == 'C#':
            os.chdir(f'codes/{args.codes_dir_name}/cs/{code_uid}')
            print(os.getcwd())

            compile_command = 'csc /out:code code.cs'# 'g++ -fPIC -O0 code.cpp -o code'
            outcome = execute_command(compile_command)
            # print(outcome)

            num_passed = 0
            for index, hidden_unit_test in enumerate(hidden_unit_tests):
                input = hidden_unit_test['input']
                output = hidden_unit_test['output'][0]

                test_command = 'mono code'# or ./code?
                outcome = execute_command(test_command, input)
                # print(outcome)

                is_passed = True if outcome.returncode == 0 and (
                        outcome.stdout in output 
                        or outcome.stdout.rstrip() in output 
                        or outcome.stdout.replace('\n','\r\n') in output 
                        or outcome.stdout.replace('\n', '\r\n').rstrip() in output
                        ) else False
                if is_passed is True:
                    num_passed += 1
                # print(is_passed)

                peak_mem, cpu_time, timeout_flag = count_memory_and_time(test_command, input)
                if timeout_flag == True:
                    # if code_uid in ['1d3a8804e288dee710091aedda77299c']:# 这些题目有问题，计算perf时会运行超时一直不退出（可能是死锁）
                    timeout_code_uids.append(code_uid)
                    example['pass_rate'] = 0
                    example['mean_cpu_time'] = 0
                    example['mean_peak_mem'] = 0
                    os.chdir('../../../..')
                    return example
                testcases_perf.append([
                    src_uid, lang, code_uid, input, output, is_passed, cpu_time, peak_mem
                ])
            testcases_perf_df = pd.DataFrame(testcases_perf, columns=['src_uid','lang','code_uid','test_input','test_output','is_passed','cpu_time','peak_mem'])
            testcases_perf_df.to_csv("./perf.csv", index=False)# stored the testcase-level performance of the code under its code_uid dir
            pass_rate = round(100. * num_passed / num_hidden_unit_tests, 2)
            print(f'Pass rate: {pass_rate}% [{num_passed}/{num_hidden_unit_tests}]')

            os.chdir('../../../..')
            # print(os.getcwd())

            example['pass_rate'] = pass_rate
            example['mean_cpu_time'] = testcases_perf_df['cpu_time'].mean()
            example['mean_peak_mem'] = testcases_perf_df['peak_mem'].mean()

        elif lang_cluster == 'Java':
            os.chdir(f'codes/{args.codes_dir_name}/java/{code_uid}')
            print(os.getcwd())

            # find class name in the java source code
            pattern = r'public\s+(?:final\s+)?class\s+(\w+)'
            matches = re.search(pattern, source_code)
            if matches:
                class_name = matches.group(1)
            else:
                print('Class name not found, use default class name.')
                class_name = 'code'

            compile_command = f'javac {class_name}.java'
            outcome = execute_command(compile_command)
            # print(outcome)

            num_passed = 0
            for index, hidden_unit_test in enumerate(hidden_unit_tests):
                input = hidden_unit_test['input']
                output = hidden_unit_test['output'][0]

                test_command = f'java {class_name}'
                outcome = execute_command(test_command, input)
                # print(outcome)

                is_passed = True if outcome.returncode == 0 and (
                        outcome.stdout in output 
                        or outcome.stdout.rstrip() in output 
                        or outcome.stdout.replace('\n','\r\n') in output 
                        or outcome.stdout.replace('\n', '\r\n').rstrip() in output
                        ) else False
                if is_passed is True:
                    num_passed += 1
                # print(is_passed)

                peak_mem, cpu_time, timeout_flag = count_memory_and_time(test_command, input)
                if timeout_flag == True:
                    # if code_uid in ['1d3a8804e288dee710091aedda77299c']:# 这些题目有问题，计算perf时会运行超时一直不退出（可能是死锁）
                    timeout_code_uids.append(code_uid)
                    example['pass_rate'] = 0
                    example['mean_cpu_time'] = 0
                    example['mean_peak_mem'] = 0
                    os.chdir('../../../..')
                    return example
                testcases_perf.append([
                    src_uid, lang, code_uid, input, output, is_passed, cpu_time, peak_mem
                ])
            testcases_perf_df = pd.DataFrame(testcases_perf, columns=['src_uid','lang','code_uid','test_input','test_output','is_passed','cpu_time','peak_mem'])
            testcases_perf_df.to_csv("./perf.csv", index=False)# stored the testcase-level performance of the code under its code_uid dir
            pass_rate = round(100. * num_passed / num_hidden_unit_tests, 2)
            print(f'Pass rate: {pass_rate}% [{num_passed}/{num_hidden_unit_tests}]')

            os.chdir('../../../..')
            # print(os.getcwd())

            example['pass_rate'] = pass_rate
            example['mean_cpu_time'] = testcases_perf_df['cpu_time'].mean()
            example['mean_peak_mem'] = testcases_perf_df['peak_mem'].mean()

        elif lang_cluster == 'Python':
            os.chdir(f'./codes/{args.codes_dir_name}/python/{code_uid}')
            print(os.getcwd())

            num_passed = 0
            for index, hidden_unit_test in enumerate(hidden_unit_tests):
                input = hidden_unit_test['input']
                output = hidden_unit_test['output'][0]

                test_command = 'python code.py'
                outcome = execute_command(test_command, input)
                # print(outcome)

                is_passed = True if outcome.returncode == 0 and (
                        outcome.stdout in output 
                        or outcome.stdout.rstrip() in output 
                        or outcome.stdout.replace('\n','\r\n') in output 
                        or outcome.stdout.replace('\n', '\r\n').rstrip() in output
                        ) else False
                if is_passed is True:
                    num_passed += 1
                # print(is_passed)

                peak_mem, cpu_time, timeout_flag = count_memory_and_time(test_command, input)
                if timeout_flag == True:
                    # if code_uid in ['1d3a8804e288dee710091aedda77299c']:# 这些题目有问题，计算perf时会运行超时一直不退出（可能是死锁）
                    timeout_code_uids.append(code_uid)
                    example['pass_rate'] = 0
                    example['mean_cpu_time'] = 0
                    example['mean_peak_mem'] = 0
                    os.chdir('../../../..')
                    return example
                testcases_perf.append([
                    src_uid, lang, code_uid, input, output, is_passed, cpu_time, peak_mem
                ])
            testcases_perf_df = pd.DataFrame(testcases_perf, columns=['src_uid','lang','code_uid','test_input','test_output','is_passed','cpu_time','peak_mem'])
            testcases_perf_df.to_csv("./perf.csv", index=False)# stored the testcase-level performance of the code under its code_uid dir
            pass_rate = round(100. * num_passed / num_hidden_unit_tests, 2)
            print(f'Pass rate: {pass_rate}% [{num_passed}/{num_hidden_unit_tests}]')
            
            os.chdir('../../../..')
            # print(os.getcwd())

            example['pass_rate'] = pass_rate
            example['mean_cpu_time'] = testcases_perf_df['cpu_time'].mean()
            example['mean_peak_mem'] = testcases_perf_df['peak_mem'].mean()

    return example


def main():
    load_path = Path(__file__).parent / Path('dataset') / Path(args.code_test_data_name)
    save_path = Path(__file__).parent / Path('dataset') / Path(args.temp_save_name)

    dataset = load_dataset('json', split='train', data_files=str(load_path))
    # dataset = dataset.select(range(40,60))
    dataset = dataset.filter(lambda x: x['lang']=="Java 8")
    print(dataset)
    dataset = dataset.map(add_passrate_perfmetrcs)
    print(dataset)

    dataset.to_json(save_path, lines=True)
    # dataset.save_to_disk(save_path)


if __name__ == '__main__':
    args = parse_arguments()
    timeout_code_uids = []
    main()
    pd.DataFrame(timeout_code_uids).to_csv("./cs_timeout_code_uids.csv")
    # python test_codes.py --code_test_data_name screened_sampled_dataset_2023-10-02_23:35:49/sampled_submissions.jsonl --codes_dir_name sample_verified --temp_save_name screened_sampled_dataset_2023-10-02_23:35:49/sampled_cs_submissions_with_perf.jsonl
    # python test_codes.py --code_test_data_name yunzhe/1002_submissions_results_H_Java.jsonl --codes_dir_name_hard --temp_save_name test_java_hard.jsonl
    # python test_codes.py --code_test_data_name yunzhe/1002_submissions_accepted_M_Java.jsonl --codes_dir_name java_test_median --temp_save_name test_java_median.jsonl