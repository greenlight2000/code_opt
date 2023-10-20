from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,1'
# import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
import torch
from functools import partial
import argparse
import logging

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', default=None, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--checkpoint', default='/home/wyk/hf_cache/WizardCoder', type=str)
    parser.add_argument('--data_load_name', default='code_opt_dataset.jsonl', type=str)
    parser.add_argument('--result_save_name', default='test_opt_wizardcoder', type=str)
    parser.add_argument('--generate_func', default='sample', choices=['beam_search', 'sample'], type=str)
    parser.add_argument('--num_beams', default=10, type=int)# only valid when generate_func==beam_search
    parser.add_argument('--temperature', default=0.5, type=float)# only valid when generate_func==sample
    args = parser.parse_args()

    return args
# CUDA_VISIBLE_DEVICES=0,1 python test_inference.py --result_save_name test_opt_wizardcoder_smpl2_0.7 --generate_func sample --temperature 0.7
# CUDA_VISIBLE_DEVICES=2,7 python test_inference.py --result_save_name test_opt_wizardcoder_beam_10 --generate_func sample --num_beams 10
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def beam_search_generate_text(prompt, max_new_tokens, num_beams):
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=max_new_tokens,
        # temperature=temperature,
        # do_sample=False,
        # top_k=50,
        # top_p=0.95,
        num_beams=num_beams,# must greater than k
        num_return_sequences=k,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    ).to('cpu')
    responses = [tokenizer.decode(output, skip_special_tokens=True)
                 .split('### Response:')[-1].strip()
                  for output in outputs]

    return responses
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def sample_generate_text(prompt, max_new_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        # num_beams=15,
        num_return_sequences=k,
        # early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    ).to('cpu')
    responses = [tokenizer.decode(output, skip_special_tokens=True)
                 .split('### Response:')[-1].strip()
                  for output in outputs]

    return responses


def add_mem_optimization(example):
    src_uid = example['src_uid']
    task_description = example['task_description']
    baseline_code_uid = example['mem_baseline_code_uid']
    baseline_code = example['mem_baseline_code']
    baseline_perf = example['mem_baseline_perf']
    testcases = example['testcases']
    lang = example['lang']
    example_input = testcases[0]['input']
    example_output = testcases[0]['output'][0]
    user_message = f"""As an expert software developer with years of experience, please meticulously inspect the following the following unoptimized inefficient code and give an optimized version of the code, making it solve the same exact problem while achieving smaller memory usage.
To pass the testcases, the generated optimized code should strictly follow the same input/output format as the original unoptimized code.
The detailed information are as follows:
1. Description of the problem: {task_description}
2. Programming language: {lang}
3. Unoptimized code: 
```
{baseline_code}
```
4. Example testcase input: {example_input}
5. Example testcase output: {example_output}

Respond only the optimized code in the following JSON format:
{{"optimized_code": code string}}"""
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{user_message.strip()}

### Response:"""
    logging.info(f"\nstart mem optimizing for src_uid={src_uid}, lang={lang}")
    logging.info(f"unoptimized code:\n {baseline_code}")
    # input_tokens = count_message_tokens(prompt)
    # logging.info('input tokens: ' + str(input_tokens))
    # if input_tokens > max_input_tokens:
    #     logging.warning(f'Over input tokens limit ---- lang: {lang}, src_uid: {src_uid}')
    try:
        responses = generate_func(
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )
    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        responses = [None]*k
        # print('response: ' + str(response))
    finally:
        for i in range(k):
            if len(responses) <= i:
                logging.error(f"response number is {len(responses)}, optimization_{i} is set to empty string")
                optimization = ''
            elif responses[i] is None:
                logging.error(f"the {i}th response is None, optimization_{i} is set to empty string")
                optimization = ''
            else:
                # output_tokens = count_message_tokens(responses[i])
                # logging.info('the {i}th response tokens: ' + str(output_tokens))
                # if output_tokens > max_new_tokens:
                #     logging.warning(f'Over output tokens limit ---- lang: {lang}, src_uid: {src_uid}')
                optimization = responses[i]
            example[f'optimization_{i}'] = optimization
            logging.info(f'optimization_{i}: {str(optimization)}')
    return example

def add_time_optimization(example):
    src_uid = example['src_uid']
    task_description = example['task_description']
    baseline_code_uid = example['time_baseline_code_uid']
    baseline_code = example['time_baseline_code']
    baseline_perf = example['time_baseline_perf']
    testcases = example['testcases']
    lang = example['lang']
    example_input = testcases[0]['input']
    example_output = testcases[0]['output'][0]
    user_message = f"""As an expert software developer with years of experience, please meticulously inspect the following unoptimized inefficient code and give an optimized version of the code, making it solve the same exact problem while achieving faster execution time.
To pass the testcases, the generated optimized code should strictly follow the same input/output format as the original unoptimized code.
The detailed information are as follows:
1. Description of the problem: {task_description}
2. Programming language: {lang}
3. Unoptimized code: 
```
{baseline_code}
```
4. Example testcase input: {example_input}
5. Example testcase output: {example_output}

Respond only the optimized code in the following JSON format:
{{"optimized_code": code string}}"""
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{user_message.strip()}

### Response:"""
    logging.info(f"\nstart inferencing for src_uid={src_uid}, lang={lang}")
    logging.info(f"unoptimized code:\n {baseline_code}")
    # input_tokens = count_message_tokens(prompt)
    # logging.info('input tokens: ' + str(input_tokens))
    # if input_tokens > max_input_tokens:
    #     logging.warning(f'Over input tokens limit ---- lang: {lang}, src_uid: {src_uid}')
    try:
        responses = generate_func(
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )
    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        responses = [None]*k
        # print('response: ' + str(response))
    finally:
        for i in range(k):
            if len(responses) <= i:
                logging.error(f"generated sequence number is {len(responses)}, optimization_{i} is set to empty string")
                optimization = ''
            elif responses[i] is None:
                logging.error(f"the {i}th generated sequence is None, optimization_{i} is set to empty string")
                optimization = ''
            else:
                # output_tokens = count_message_tokens(responses[i])
                # logging.info('the {i}th response tokens: ' + str(output_tokens))
                # if output_tokens > max_new_tokens:
                #     logging.warning(f'Over output tokens limit ---- lang: {lang}, src_uid: {src_uid}')
                optimization = responses[i]
            example[f'optimization_{i}'] = optimization
            logging.info(f'optimization_{i}: {str(optimization)}')
    return example


if __name__ == '__main__':
    args = parse_arguments()
    device = 'cuda'
    checkpoint = args.checkpoint
    k = 5 # 对每一个unoptimizaed code生成10个optimization结果
    load_path = '/home/wyk/CodeLLMBenchmark/code_opt/code-opt-inference/data/code_opt_dataset.jsonl'
    time_opt_output_path = f'/home/wyk/CodeLLMBenchmark/code_opt/code-opt-inference/results/test/{args.result_save_name}_time.jsonl'
    mem_opt_output_path = f'/home/wyk/CodeLLMBenchmark/code_opt/code-opt-inference/results/test/{args.result_save_name}_mem.jsonl'
    log_path = f'/home/wyk/CodeLLMBenchmark/code_opt/code-opt-inference/results/test/{args.result_save_name}_log.log'
    # 配置日志输出到文件
    logging.basicConfig(filename=log_path, level=logging.DEBUG)
    # 创建一个日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # 创建一个控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    # 创建一个文件处理器
    file_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    # 创建一个格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 将格式化器添加到处理器
    console_handler.set_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 将格式化器添加到处理器
    console_handler.setFormatter(formatter) 
    file_handler.setFormatter(formatter)
    # 将处理器添加到日志记录器
    logger.addHandler(console_handler) 
    logger.addHandler(file_handler)

    logging.info('=========start=========')
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        use_fast=True,
        trust_remote_code=True,
        token=None,
        cache_dir=None
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
        # load_in_4bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='auto',
        token=None,
        cache_dir=None
    )
    max_input_tokens = tokenizer.model_max_length  # 8192
    max_new_tokens = 2048 # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    if args.generate_func == 'sample':
        temperature = args.temperature
        generate_func = partial(sample_generate_text, temperature=temperature)
        assert temperature >= 0 and temperature <= 1
        num_beams = int(args.num_beams)
    elif args.generate_func == 'beam_search':
        num_beams = int(args.num_beams)
        generate_func = partial(beam_search_generate_text, num_beams=num_beams)
        assert num_beams >= k
    else:
        print('unkonwn generate function')
        exit()

    dataset = load_dataset('json', split='train', data_files=str(load_path))
    dataset.cleanup_cache_files()  # for multiple evaluation
    dataset = dataset.select(range(5)) # 采10个样本试试
    logging.info("=====start mem optimiing=====")
    mem_dataset = dataset.map(add_mem_optimization)
    mem_dataset.to_json(mem_opt_output_path)
    logging.info("=====start time optimiing=====")
    time_dataset = dataset.map(add_time_optimization)
    time_dataset.to_json(time_opt_output_path)