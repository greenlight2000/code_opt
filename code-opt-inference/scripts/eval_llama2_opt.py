import re
import json
import torch
import logging
import argparse
import warnings

from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tenacity import retry, stop_after_attempt, wait_random_exponential


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', default=None, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--checkpoint', default='meta-llama/Llama-2-7b-chat-hf', type=str)
    parser.add_argument('--data_load_name', default='code_summarization_dataset_with_gt.jsonl', type=str)
    parser.add_argument('--result_save_name', default='code_summ_inference_llama2.jsonl', type=str)
    parser.add_argument('--log_file_name', default='code_summ_inference_llama2.log', type=str)
    parser.add_argument('--temperature', default=0.7, type=float)
    args = parser.parse_args()

    return args


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text(prompt, temperature, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=k,
        pad_token_id=tokenizer.eos_token_id
    ).to('cpu')
    responses = [tokenizer.decode(output, skip_special_tokens=True)
                 .split('[/INST]')[-1].strip().replace('</s>','')
                  for output in outputs]

    return responses


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
    prompt = f'<s>[INST] {user_message.strip()} [/INST]'
    logging.info(f"\nstart inferencing for src_uid={src_uid}, lang={lang}")
    logging.info(f"unoptimized code:\n {baseline_code}")
    input_tokens = count_message_tokens(prompt)
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning(f'Over input tokens limit ---- lang: {lang}, src_uid: {src_uid}')
    try:
        responses = generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
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
                output_tokens = count_message_tokens(responses[i])
                logging.info('the {i}th response tokens: ' + str(output_tokens))
                if output_tokens > max_new_tokens:
                    logging.warning(f'Over output tokens limit ---- lang: {lang}, src_uid: {src_uid}')
                optimization = responses[i]
            example[f'optimization_{i}'] = optimization
            logging.info(f'optimization_{i}: {str(optimization)}')
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
    prompt = f'<s>[INST] {user_message.strip()} [/INST]'
    logging.info(f"\nstart mem optimizing for src_uid={src_uid}, lang={lang}")
    logging.info(f"unoptimized code:\n {baseline_code}")
    input_tokens = count_message_tokens(prompt)
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning(f'Over input tokens limit ---- lang: {lang}, src_uid: {src_uid}')
    try:
        responses = generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
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
                output_tokens = count_message_tokens(responses[i])
                logging.info('the {i}th response tokens: ' + str(output_tokens))
                if output_tokens > max_new_tokens:
                    logging.warning(f'Over output tokens limit ---- lang: {lang}, src_uid: {src_uid}')
                optimization = responses[i]
            example[f'optimization_{i}'] = optimization
            logging.info(f'optimization_{i}: {str(optimization)}')
    return example

def main():
    load_path = Path(__file__).parent.parent / Path('data') / Path(args.data_load_name)
    mem_save_path = Path(__file__).parent.parent / Path('results') / Path(f"mem_{args.result_save_name}")
    time_save_path = Path(__file__).parent.parent / Path('results') / Path(f"time_{args.result_save_name}")
    dataset = load_dataset('json', split='train', data_files=str(load_path))

    dataset.cleanup_cache_files()  # for multiple evaluation
    print(dataset)
    logging.info("=====start mem optimiing=====")
    mem_ds = dataset.map(add_mem_optimization)
    mem_ds.to_json(mem_save_path, lines=True)
    logging.info("=====start time optimiing=====")
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

    # References: https://huggingface.co/blog/llama2
    # References: https://colab.research.google.com/drive/1X1z9Q6domMKl2CnEM0QGHNwidLfR4dW2?usp=sharing
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        use_fast=True,
        trust_remote_code=True,
        token=args.access_token,
        cache_dir=args.cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
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
    temperature = args.temperature
    max_input_tokens = tokenizer.model_max_length  # 1000000000000000019884624838656
    # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    max_new_tokens = 2048
    k = 5 # 每道题生成k个optimized code

    main()
# python scripts/eval_llama2.py --checkpoint absolute_path_to_Llama_dir --data_load_name code_summarization_dataset_with_gt.jsonl --result_save_name code_summ_inference_llama2.jsonl --log_file_name code_summ_inference_llama2.log
