import os
import re
import argparse
import subprocess
import time
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset

import json

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_test_data_name', default='mem_code_opt_inference_vicuna_replenish.jsonl', type=str) # will read the submission dataset that is to be evaluated from './dataset/{code_test_data_name}'
    parser.add_argument('--codes_dir_name', default='vicuna_opt_parse', type=str) # same as --codes_dir_name in save_codes. source codes files to be evaluated should be stored at './codes/{args.codes_dir_name}/{lang_cluster}/{code_uid}
    # parser.add_argument('--temp_save_name', default='palm_data', type=str) # the converted dataset(add the is_passed and performance metrics for each submission data) will be placed at './dataset/{temp_save_name}'
    parser.add_argument('--opt_type', default='time', choices=['mem','time'] ,type=str) # 'mem' or 'time'
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

def test_opt(compile_command, test_command, hidden_unit_tests, src_uid, lang, code_name):
    if compile_command is not None:
        outcome = execute_command(compile_command)
    num_passed = 0
    tmp_testcases_perf = []
    for index, hidden_unit_test in enumerate(hidden_unit_tests):
        input = hidden_unit_test['input']
        output = hidden_unit_test['output'][0]
        # 1. run1 get is_passed
        outcome = execute_command(test_command, input)
        is_passed = True if outcome.returncode == 0 and (
                outcome.stdout in output 
                or outcome.stdout.rstrip() in output 
                or outcome.stdout.replace('\n','\r\n') in output 
                or outcome.stdout.replace('\n', '\r\n').rstrip() in output
                ) else False
        if is_passed is True:
            num_passed += 1
        # 2. run2 get peak_mem and cpu_time
        peak_mem, cpu_time, timeout_flag = count_memory_and_time(test_command, input)
        if timeout_flag == True:
            timeout_code_uids.append(f'{src_uid}_{code_name}')
            peak_mem = 0
            cpu_time = 0
            tmp_testcases_perf.append([
                code_name, index, src_uid, lang, None, is_passed, cpu_time, peak_mem
            ])
            break
        tmp_testcases_perf.append([
            code_name, index, src_uid, lang, None, is_passed, cpu_time, peak_mem
        ])
    pass_rate = round(100. * num_passed / len(hidden_unit_tests), 2)
    print(f'{code_name} pass rate: {pass_rate}% [{num_passed}/{len(hidden_unit_tests)}]')
    mean_time = np.mean(np.array(tmp_testcases_perf).T[-2]).tolist()
    mean_mem = np.mean(np.array(tmp_testcases_perf).T[-1]).tolist()
    return pass_rate, mean_time, mean_mem, tmp_testcases_perf

def test_unopt(compile_command, test_command, hidden_unit_tests, src_uid, lang, code_uid, retest=10):
    if compile_command is not None:
        outcome = execute_command(compile_command)
    num_passed = 0
    mean_mem_li, mean_time_li = [], []
    tmp_testcases_perf = []
    for index, hidden_unit_test in enumerate(hidden_unit_tests):
        input = hidden_unit_test['input']
        output = hidden_unit_test['output'][0]
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
        peak_mem_li, cpu_time_li = [], []# shape: (num_test_cases, 10)
        for i in range(retest):
            peak_mem, cpu_time, timeout_flag = count_memory_and_time(test_command, input)
            if timeout_flag == True:
                timeout_code_uids.append(f'{src_uid}_{code_uid}')
                peak_mem_li = [0]*10
                cpu_time_li = [0]*10
                break
            peak_mem_li.append(float(peak_mem))
            cpu_time_li.append(float(cpu_time))
        tmp_testcases_perf.append([
            'unopt', index, src_uid, lang, code_uid, is_passed, ','.join([str(x) for x in cpu_time_li]), ','.join([str(x) for x in peak_mem_li])
        ])
        mean_mem_li.append(peak_mem_li)
        mean_time_li.append(cpu_time_li)
    pass_rate = round(100. * num_passed / len(hidden_unit_tests), 2)
    print(f'unopt pass rate: {pass_rate}% [{num_passed}/{len(hidden_unit_tests)}]')
    mean_mem_li = np.mean(mean_mem_li, axis=0) # shape: (10,), calculate the mean testcases peak memory value for each run in the 10 runs
    mean_time_li = np.mean(mean_time_li, axis=0) # shape: (10,), calculate the mean testcases cpu time value for each run in the 10 runs
    return pass_rate, mean_time_li, mean_mem_li, tmp_testcases_perf

def cal_passrate_perfmetrcs(example):
    src_uid = example['src_uid']
    lang = example['lang']
    lang = example['lang']
    code_uid = example[f'{args.opt_type}_baseline_code_uid']
    source_code = example[f'{args.opt_type}_baseline_code']
    hidden_unit_tests = example['testcases']# eval(example['hidden_unit_tests'])
    num_hidden_unit_tests = len(hidden_unit_tests)
    testcases_perf = []
    code_perf = []

    if num_hidden_unit_tests == 0:
        print('Failed to generate hidden unit tests:', code_uid)
        # example['pass_rate'] = 0.00
    else:
        if lang == 'GNU C':
            os.chdir(f'./codes/{args.codes_dir_name}/{args.opt_type}/c/{src_uid}')
            print(os.getcwd())
            compile_command_template = 'gcc -fno-optimize-sibling-calls -w -fno-strict-aliasing -DONLINE_JUDGE -include limits.h -fno-asm -s -O2 -DONLINE_JUDGE -include math.h -static -lm -o {code_name} {code_name}.c'#'gcc -fPIC -O0 code.c -o code'
            test_command_template = './{code_name}'
        elif lang == 'GNU C++':
            os.chdir(f'./codes/{args.codes_dir_name}/{args.opt_type}/cpp/{src_uid}')
            print(os.getcwd())
            compile_command_template = 'g++ -s -x c++ -O2 -w -DONLINE_JUDGE -include math.h -include limits.h -static -lm -o {code_name} {code_name}.cpp'#'gcc -fPIC -O0 code.c -o code'
            test_command_template = './{code_name}'
        elif lang == 'Mono C#':
            os.chdir(f'./codes/{args.codes_dir_name}/{args.opt_type}/cs/{src_uid}')
            print(os.getcwd())
            compile_command_template = 'csc /out:{code_name} {code_name}.cs'# 'g++ -fPIC -O0 code.cpp -o code'
            test_command_template = 'mono {code_name}'
        elif lang == 'Java 8':
            os.chdir(f'./codes/{args.codes_dir_name}/{args.opt_type}/java/{src_uid}')
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

        elif lang == 'Python 3':
            os.chdir(f'./codes/{args.codes_dir_name}/{args.opt_type}/python/{src_uid}')
            print(os.getcwd())
            compile_command_template = None
            test_command_template = 'python {code_name}.py'
        else:
            print('Unsupported language:', lang)
            os.chdir('../../../../../')
            return example
        
        if os.path.exists(f'./codes_perf.csv'):
            os.chdir('../../../../../')
            return example
        # test unoptimized code
        retest = 20
        compile_command = compile_command_template.format(code_name='unopt') if compile_command_template is not None else None
        test_command = test_command_template.format(code_name='unopt')
        pass_rate, mean_time_li, mean_mem_li, tmp_testcases_perf = test_unopt(compile_command, test_command, hidden_unit_tests, src_uid, lang, code_uid, retest)
        testcases_perf.extend(tmp_testcases_perf)
        mean_time_str = ','.join([str(x) for x in mean_time_li])
        mean_mem_str = ','.join([str(x) for x in mean_mem_li])
        code_perf.append([
            'unopt', src_uid, lang, code_uid, pass_rate, mean_time_str, mean_mem_str
        ])
        # test optimized code
        for code in ['opt0','opt1','opt2','opt3','opt4']:
            compile_command = compile_command_template.format(code_name=code) if compile_command_template is not None else None
            test_command = test_command_template.format(code_name=code)
            pass_rate, mean_time, mean_mem, tmp_testcases_perf = test_opt(compile_command, test_command, hidden_unit_tests, src_uid, lang, code)
            testcases_perf.extend(tmp_testcases_perf)
            code_perf.append([
                code, src_uid, lang, code_uid, pass_rate, mean_time, mean_mem
            ])
        testcases_perf_df = pd.DataFrame(testcases_perf, columns=['code_name','test_idx','src_uid','lang','code_uid','is_passed','cpu_time','peak_mem'])
        testcases_perf_df.to_csv("./testcases_perf.csv", index=False)# stored the testcase-level performance of the code under its code_uid dir
        code_perf_df = pd.DataFrame(code_perf, columns=['code_name','src_uid','lang','code_uid','pass_rate','mean_cpu_time','mean_peak_mem'])
        code_perf_df.to_csv("./codes_perf.csv", index=False)# stored the code-level performance of the code under its code_uid dir
        os.chdir('../../../../../')
    return example


def main():
    load_path = Path(__file__).parent.parent / Path('code-opt-inference/results') / Path(args.code_test_data_name)
    # save_path = Path(__file__).parent / Path('dataset') / Path(args.temp_save_name)

    dataset = load_dataset('json', split='train', data_files=str(load_path))
    # dataset = dataset.select(range(40,60))
    # dataset = dataset.filter(lambda x: x['lang']=="Mono C#")
    print(dataset)
    dataset = dataset.map(cal_passrate_perfmetrcs)
    print(dataset)

    # dataset.to_json(save_path, lines=True)
    # dataset.save_to_disk(save_path)


if __name__ == '__main__':
    args = parse_arguments()
    timeout_code_uids = []
    main()
    pd.DataFrame(timeout_code_uids).to_csv("./c_perf_timeout_codes.csv")
    # python test_opt_codes.py --code_test_data_name mem_code_opt_inference_vicuna_replenish.jsonl --codes_dir_name vicuna_opt_parse --opt_type mem
    # python test_opt_codes.py --code_test_data_name time_code_opt_inference_vicuna_replenish.jsonl --codes_dir_name vicuna_opt_parse --opt_type time

    # python test_opt_codes.py --code_test_data_name mem_code_opt_inference_wizardcoder_replenish4.jsonl --codes_dir_name wizardcoder_opt_parse --opt_type mem > nohup_test_opt_codes_wizardcoder_mem.log 2>&1 &
    # python test_opt_codes.py --code_test_data_name time_code_opt_inference_wizardcoder_replenish4.jsonl --codes_dir_name wizardcoder_opt_parse --opt_type time > nohup_test_opt_codes_wizardcoder_time.log 2>&1 &

    # python test_opt_codes.py --code_test_data_name mem_code_opt_data_codellama_replenish5.jsonl --codes_dir_name codellama_opt_parse --opt_type mem > nohup_test_opt_codes_codellama_mem.log 2>&1 &
    # python test_opt_codes.py --code_test_data_name time_code_opt_data_codellama_replenish5.jsonl --codes_dir_name codellama_opt_parse --opt_type time > nohup_test_opt_codes_codellama_time.log 2>&1 &

    # python test_opt_codes.py --code_test_data_name mem_code_opt_data_gpt4.jsonl --codes_dir_name gpt4_opt_parse --opt_type mem > nohup_test_opt_codes_gpt4_mem.log 2>&1 &
    # python test_opt_codes.py --code_test_data_name time_code_opt_data_gpt4.jsonl --codes_dir_name gpt4_opt_parse --opt_type time > nohup_test_opt_codes_gpt4_time.log 2>&1 &

    # python test_opt_codes.py --code_test_data_name mem_code_opt_data_gpt3.jsonl --codes_dir_name gpt3_opt_parse --opt_type mem > nohup_test_opt_codes_gpt3_mem.log 2>&1 &
    # python test_opt_codes.py --code_test_data_name time_code_opt_data_gpt3.jsonl --codes_dir_name gpt3_opt_parse --opt_type time > nohup_test_opt_codes_gpt3_time.log 2>&1 &

    # python test_opt_codes.py --code_test_data_name mem_code_opt_data_llama2_replenish.jsonl --codes_dir_name llama2_opt_parse --opt_type mem > nohup_test_opt_codes_llama2_mem.log 2>&1 & > nohup_test_opt_codes_llama2_mem.log 2>&1 &
    # python test_opt_codes.py --code_test_data_name time_code_opt_data_llama2_replenish.jsonl --codes_dir_name llama2_opt_parse --opt_type time > nohup_test_opt_codes_llama2_time.log 2>&1 &

    # python test_opt_codes.py --code_test_data_name mem_code_opt_data_palm.jsonl --codes_dir_name palm_opt_parse --opt_type mem > nohup_test_opt_codes_palm_mem.log 2>&1 &
    # python test_opt_codes.py --code_test_data_name time_code_opt_data_palm.jsonl --codes_dir_name palm_opt_parse --opt_type time > nohup_test_opt_codes_palm_time.log 2>&1 &

    # python test_opt_codes.py --code_test_data_name mem_code_opt_data_starcoder.jsonl --codes_dir_name starcoder_opt_parse --opt_type mem
    # python test_opt_codes.py --code_test_data_name time_code_opt_data_starcoder.jsonl --codes_dir_name starcoder_opt_parse --opt_type time