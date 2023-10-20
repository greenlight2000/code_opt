import re
import json
import logging
import argparse
import google.generativeai as palm

from pathlib import Path
from datasets import load_dataset
from google.api_core import retry


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', default=None, type=str)
    parser.add_argument('--data_load_name', default='code_review_data.jsonl', type=str)
    parser.add_argument('--result_save_name', default='code_review_eval_palm.jsonl', type=str)
    parser.add_argument('--log_file_name', default='code_review_eval_palm.log', type=str)
    parser.add_argument('--temperature', default=0.7, type=float)
    args = parser.parse_args()

    return args


@retry.Retry()
def generate_text(*args, **kwargs):
    return palm.generate_text(*args, **kwargs)


@retry.Retry()
def count_message_tokens(*args, **kwargs):
    return palm.count_message_tokens(*args, **kwargs)


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
    prompt = user_message

    logging.info(f"\nstart inferencing for src_uid={src_uid}, lang={lang}")
    logging.info(f"unoptimized code:\n {baseline_code}")

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: src_uid=' + str(src_uid))

    for i in range(k):
        try:
            response = generate_text(
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
            logging.info('response: ' + str(response))

            if response.result is not None:
                output_tokens = count_message_tokens(prompt=response.result)['token_count']
                logging.info('output tokens: ' + str(output_tokens))
                if output_tokens > max_output_tokens:
                    logging.warning('Over output tokens limit: src_uid=' + str(src_uid))
                
                optimization = response.result
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
    prompt = user_message
    logging.info(f"\nstart inferencing for src_uid={src_uid}, lang={lang}")
    logging.info(f"unoptimized code:\n {baseline_code}")

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: src_uid=' + str(src_uid))

    for i in range(5):
        try:
            response = generate_text(
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
            logging.info('response: ' + str(response))

            if response.result is not None:
                output_tokens = count_message_tokens(prompt=response.result)['token_count']
                logging.info('output tokens: ' + str(output_tokens))
                if output_tokens > max_output_tokens:
                    logging.warning('Over output tokens limit: src_uid=' + str(src_uid))
                
                optimization = response.result
            else:
                logging.warning('Respond content is none.')
                optimization = ''

        except Exception as e:
            logging.error('Failed to generate text: ' + e.__str__())
            optimization = ''

        logging.info(f'optimization_{i}: ' + str(optimization))
        example[f'optimization_{i}'] = optimization


    return example


def add_smell(example):
    code_uid = example['code_uid']
    lang_cluster = example['lang_cluster']
    smell_code = example['smell_code']
    source_code = example['source_code']
    prompt = f"""As an expert software developer with years of experience, please meticulously inspect the following smell code snippet and categorize it into one of the following categories:
- large class
- data class
- blob
- feature envy
- long method
The detailed information are as follows:
1. Programming language: {lang_cluster} 
2. Smell code snippet: 
```
{smell_code.strip()}
```
3. Source code containing code smells:
```
{source_code.strip()}
```
Respond only with one of the specified categories."""

    logging.info('code uid: ' + str(code_uid))

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: ' + str(code_uid))

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        logging.info('response: ' + str(response.result))

        if response.result is not None:
            output_tokens = count_message_tokens(prompt=response.result)['token_count']
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_output_tokens:
                logging.warning('Over output tokens limit ' + str(code_uid))

            supported_smells = ['large class', 'long method', 'data class', 'blob', 'feature envy']
            if all(supported_smell not in response.result.lower() for supported_smell in supported_smells):
                logging.warning('Respond content is invalid value.')
                smell = ''
            else:
                smell = ''
                # Find the smell that first occurs in the response.
                min_index = float('inf')
                for supported_smell in supported_smells:
                    first_index = response.result.lower().find(supported_smell)
                    if first_index != -1 and first_index < min_index:
                        min_index = first_index
                        smell = supported_smell
        else:
            logging.warning('Respond content is none.')
            smell = ''

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        smell = ''

    logging.info('smell: ' + str(smell))
    example['smell'] = smell

    return example


def add_diff_tag(example):
    code_uid = example['code_uid']
    lang_cluster = example['lang_cluster']
    old_code = example['old_code']
    diff_hunk = example['diff_hunk']

    prompt = f"""As an expert code reviewer with years of experience, please meticulously inspect the following code change and categorize its quality into one of the following categories:
- 0: Good quality that no review comments required.
- 1: Poor quality that requires review comments.
The detailed information are as follows:
1. Programming language: {lang_cluster} 
2. Original version code: 
```
{old_code.strip()}
```
3. Code diff chunk:
```
{diff_hunk.strip()}
```
Respond only with the number: 0 or 1."""

    logging.info('code uid: ' + str(code_uid))

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: ' + str(code_uid))

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        logging.info('response: ' + str(response.result))

        if response.result is not None:
            output_tokens = count_message_tokens(prompt=response.result)['token_count']
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_output_tokens:
                logging.warning('Over output tokens limit ' + str(code_uid))

            supported_diff_tags = ['0', '1']
            if all(supported_diff_tag not in response.result for supported_diff_tag in supported_diff_tags):
                logging.warning('Respond content is invalid value.')
                diff_tag = 2
            else:
                diff_tag = 2
                # Find the diff tag that first occurs in the response.
                min_index = float('inf')
                for supported_diff_tag in supported_diff_tags:
                    first_index = response.result.find(supported_diff_tag)
                    if first_index != -1 and first_index < min_index:
                        min_index = first_index
                        diff_tag = int(supported_diff_tag)
        else:
            logging.warning('Respond content is none.')
            diff_tag = 2

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        diff_tag = 2

    logging.info('diff_tag: ' + str(diff_tag))
    example['diff_tag'] = diff_tag

    return example


def add_review_comment(example):
    code_uid = example['code_uid']
    lang_cluster = example['lang_cluster']
    old_code = example['old_code']
    diff_hunk = example['diff_hunk']
    prompt = f"""As an expert code reviewer with years of experience, please meticulously inspect the following code change and provide a concise review comment.
The detailed information are as follows:
1. Programming language: {lang_cluster} 
2. Original version code: 
```
{old_code.strip()}
```
3. Code diff chunk:
```
{diff_hunk.strip()}
```
Respond only with a string that represents review comment."""

    logging.info('code uid: ' + str(code_uid))

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: ' + str(code_uid))

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        logging.info('response: ' + str(response.result))

        if response.result is not None:
            output_tokens = count_message_tokens(prompt=response.result)['token_count']
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_output_tokens:
                logging.warning('Over output tokens limit ' + str(code_uid))

            review_comment = response.result
        else:
            logging.warning('Respond content is none.')
            review_comment = ''

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        review_comment = ''

    logging.info('review_comment: ' + str(review_comment))
    example['review_comment'] = review_comment

    return example


def add_hidden_unit_tests(example):
    code_uid = example['code_uid']
    prob_desc_description = example['prob_desc_description']
    prob_desc_time_limit = example['prob_desc_time_limit']
    prob_desc_memory_limit = example['prob_desc_memory_limit']
    prob_desc_input_spec = example['prob_desc_input_spec']
    prob_desc_output_spec = example['prob_desc_output_spec']
    prob_desc_sample_inputs = example['prob_desc_sample_inputs']
    prob_desc_sample_outputs = example['prob_desc_sample_outputs']
    prob_desc_notes = example['prob_desc_notes']
    source_code = example['source_code']
    lang_cluster = example['lang_cluster']
    num_hidden_unit_tests = example['num_hidden_unit_tests']
    prompt = f"""As an expert code test developer with years of experience, please provide multiple test cases for a given problem along and its solution.
The detailed information are as follows:
1. Problem description: {prob_desc_description}
2. Time limit: {prob_desc_time_limit}
3. Memory limit: {prob_desc_memory_limit}
4. Input specification: {prob_desc_input_spec}
5. Output specification: {prob_desc_output_spec}
6. Sample inputs: {prob_desc_sample_inputs}
7. Sample outputs: {prob_desc_sample_outputs}
8. Sample explanations: {prob_desc_notes}
9. Programming language: {lang_cluster} 
10. Solution source code: 
```
{source_code.strip()}
```
Craft {num_hidden_unit_tests} test cases with these criteria:
1. Each test case contains a string for both input and output.
2. The solution source code successfully processes the test case's input with no errors.
3. The solution source code's outcome aligns with the test case's output.
4. All test cases are simple and achieve optimal branch and line coverage.
Respond only with a string in the following JSON format:
[{{"input": input string, "output": output string}}]"""

    logging.info('code uid: ' + str(code_uid))

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: ' + str(code_uid))

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        logging.info('response: ' + str(response.result))

        if response.result is not None:
            output_tokens = count_message_tokens(prompt=response.result)['token_count']
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_output_tokens:
                logging.warning('Over output tokens limit ' + str(code_uid))

            pattern = r'\[\s*\{.*?\}\s*\]'
            matches = re.search(pattern, response.result, re.DOTALL)
            if matches:
                json_array_string = matches.group().replace("'", '"')
                try:
                    json_array = json.loads(json_array_string, strict=False)
                    if isinstance(json_array, list):
                        for json_item in json_array:
                            if isinstance(json_item['input'], list):
                                json_item['input'] = str(json_item['input'][0])
                            if isinstance(json_item['output'], str):
                                json_item['output'] = [json_item['output']]
                        hidden_unit_tests = str(json_array)
                    else:
                        logging.warning('Respond content is not a list.')
                        hidden_unit_tests = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"
                except json.JSONDecodeError as e:
                    logging.warning('Failed to load json:', e)
                    hidden_unit_tests = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"
            else:
                logging.warning('JSON array object not found.')
                hidden_unit_tests = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"

        else:
            logging.warning('Respond content is none.')
            hidden_unit_tests = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        hidden_unit_tests = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"

    logging.info('hidden_unit_tests: ' + str(hidden_unit_tests))
    example['hidden_unit_tests'] = hidden_unit_tests

    return example


def main():
    load_path = Path(__file__).parent.parent / Path('data') / Path(args.data_load_name)
    mem_save_path = Path(__file__).parent.parent / Path('results') / Path(f"mem_{args.result_save_name}")
    time_save_path = Path(__file__).parent.parent / Path('results') / Path(f"time_{args.result_save_name}")
    dataset = load_dataset('json', split='train', data_files=str(load_path))
    # dataset = dataset.select(range(0,1))
    dataset.cleanup_cache_files()  # for multiple evaluation
    print(dataset)
    logging.info("=====start mem optimiing=====")
    mem_ds = dataset.map(add_mem_optimization)
    mem_ds.to_json(mem_save_path, lines=True)
    logging.info("=====start time optimiing=====")
    time_ds = dataset.map(add_time_optimization)
    time_ds.to_json(time_save_path, lines=True)


if __name__ == '__main__':
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

    # References: https://github.com/google/generative-ai-python/issues/29
    palm.configure(api_key=args.api_key, transport='rest')
    models = [model for model in palm.list_models() if 'generateText' in model.supported_generation_methods]
    temperature = args.temperature
    max_input_tokens = models[0].input_token_limit  # 8192
    max_output_tokens = models[0].output_token_limit  # 1024
    k = 5 # 每道题生成k个optimized code
    main()
    # python scripts/eval_palm.py
