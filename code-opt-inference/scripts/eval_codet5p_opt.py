import re
import json
import torch
import logging
import argparse
import warnings
import os
os.environ['CUDA_VISIBLE_DEVICES']='4,7'
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tenacity import retry, stop_after_attempt, wait_random_exponential


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', default=None, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--checkpoint', default='Salesforce/instructcodet5p-16b', type=str)
    parser.add_argument('--data_load_name', default='code_summarization_dataset_with_gt.jsonl', type=str)
    parser.add_argument('--result_save_name', default='code_summ_inference_codet5p.jsonl', type=str)
    parser.add_argument('--log_file_name', default='code_summ_inference_codet5p.log', type=str)
    args = parser.parse_args()

    return args


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text(prompt, temperature, max_length):
    encoding = tokenizer(prompt, return_tensors='pt', truncation=True, add_special_tokens=False).to(device)
    encoding['decoder_input_ids'] = encoding['input_ids'].clone()
    outputs = model.generate(
        **encoding,
        max_length=max_length,
        temperature=temperature,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    ).to('cpu')
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response.split('### Response:')[-1].strip()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def count_message_tokens(content):
    tokens = tokenizer(content)['input_ids']
    num_tokens = len(tokens)

    return num_tokens

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

    user_message = f"""As an expert software developer with years of experience, please meticulously inspect the following low performance code sample and give an optimized version of the code, making it solve the same exact problem but achieve faster execution time.
    To pass the testcases, the generated optimized code should strictly follow the same input output format as the original version of code.
    The detailed information are as follows:
    1. Description of the problem which the sample code solves: {task_description}
    2. Programming language: {lang}
    3. Original version code: 
    ```
    {baseline_code}
    ```
    4. Example testcase input: {example_input}
    5. Example testcase output: {example_output}

    Respond only with a string in the following JSON format:
    {{“optimized_version_of_the_code”: code string}}"""
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{user_message.strip()}

### Response:"""
    logging.info(f'src_uid: {src_uid}, baseline_code_uid: {baseline_code_uid}, baseline_perf: {baseline_perf}')
    input_tokens = count_message_tokens(prompt)
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning(f'Over input tokens limit ---- lang: {lang}, src_uid: {src_uid}')
    for i in range(2):
        try:
            response = generate_text(
                prompt=prompt,
                temperature=temperature,
                max_length=max_length
            )
            logging.info('response: ' + str(response))

            if response is not None:
                output_tokens = count_message_tokens(response)
                logging.info('output tokens: ' + str(output_tokens))
                if output_tokens > max_length:
                    logging.warning(f'Over output tokens limit ---- lang: {lang}, src_uid: {src_uid}')
                optimization = response
            else:
                logging.warning('Respond content is none.')
                optimization = ''
        except Exception as e:
            logging.error('Failed to generate text: ' + e.__str__())
            optimization = ''

        logging.info(f'optimization_{i}: ' + str(optimization))
        example[f'optimization_{i}'] = optimization

    return example

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

    user_message = f"""As an expert software developer with years of experience, please meticulously inspect the following low performance code sample and give a optimized version of the code, making it solve the same exact problem but achieve smaller memory usage.
    To pass the testcases, the generated optimized code should strictly follow the same input output format as the original version of code.
    The detailed information are as follows:
    1. Description of the problem which the sample code solves: {task_description}
    2. Programming language: {lang}
    3. Original version code: 
    ```
    {baseline_code}
    ```
    4. Example testcase input: {example_input}
    5. Example testcase output: {example_output}

    Respond only with a string in the following JSON format:
    {{“optimized_version_of_the_code”: code string}}"""
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{user_message.strip()}

### Response:"""
    logging.info(f'src_uid: {src_uid}, baseline_code_uid: {baseline_code_uid}, baseline_perf: {baseline_perf}')
    input_tokens = count_message_tokens(prompt)
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning(f'Over input tokens limit ---- lang: {lang}, src_uid: {src_uid}')
    for i in range(2):
        try:
            response = generate_text(
                prompt=prompt,
                temperature=temperature,
                max_length=max_length
            )
            logging.info('response: ' + str(response))

            if response is not None:
                output_tokens = count_message_tokens(response)
                logging.info('output tokens: ' + str(output_tokens))
                if output_tokens > max_length:
                    logging.warning(f'Over output tokens limit ---- lang: {lang}, src_uid: {src_uid}')
                optimization = response
            else:
                logging.warning('Respond content is none.')
                optimization = ''
        except Exception as e:
            logging.error('Failed to generate text: ' + e.__str__())
            optimization = ''

        logging.info(f'optimization_{i}: ' + str(optimization))
        example[f'optimization_{i}'] = optimization

    return example

def main():
    load_path = Path(__file__).parent.parent / Path('data') / Path(args.data_load_name)
    mem_save_path = Path(__file__).parent.parent / Path('results') / Path(f"mem_{args.result_save_name}")
    time_save_path = Path(__file__).parent.parent / Path('results') / Path(f"time_{args.result_save_name}")
    dataset = load_dataset('json', split='train', data_files=str(load_path))
    dataset.cleanup_cache_files()  # for multiple evaluation
    print(dataset)

    mem_ds = dataset.map(add_mem_optimization)
    mem_ds.to_json(mem_save_path, lines=True)

    time_ds = dataset.map(add_time_optimization)
    time_ds.to_json(time_save_path, lines=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parse_arguments()

    log_file_path = Path(__file__).parent.parent / Path('logs') / Path(args.log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filename=log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # References: https://huggingface.co/Salesforce/instructcodet5p-16b
    # References: https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/humaneval/generate_codet5p.py
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        use_fast=True,
        trust_remote_code=True,
        token=args.access_token,
        cache_dir=args.cache_dir
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float16,
        # load_in_4bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='auto',
        token=args.access_token,
        cache_dir=args.cache_dir
    )
    print(f'Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB')
    temperature = 0.7
    max_input_tokens = tokenizer.model_max_length  # 2048
    # The maximum length of the sequence to be generated.
    max_length = 2048

    main()
    # nohup python scripts/eval_codet5p.py --checkpoint /home/wyk/hf_cache/instructcodet5p-16b --data_load_name code_summarization_dataset_with_gt.jsonl --result_save_name code_summ_inference_codet5p.jsonl --log_file_name code_summ_inference_codet5p.log > codet5p.log 2>&1 &

