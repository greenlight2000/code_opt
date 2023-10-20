import re
import argparse

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from datasets import load_dataset
import logging


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_test_data_name', default='sampled_submissions.jsonl', type=str)
    parser.add_argument('--codes_dir_name', default='palm_opt_codes', type=str)
    parser.add_argument('--opt_type', default='mem', choices=['mem', 'time'], type=str)
    parser.add_argument('--parse_code', action='store_true', default=False)
    args = parser.parse_args()

    return args
# /home/wyk/CodeLLMBenchmark/code_opt/code-opt-inference/results/mem_code_opt_inference_vicuna_replenish.jsonl

def main():
    supported_langs = ['GNU C', 'GNU C++', 'Java 8', 'Python 3', 'Mono C#']
    langs_map = {'GNU C':'c', 'GNU C++':'cpp', 'Java 8':'java', 'Python 3':'python', 'Mono C#':'cs'}# 用于把specific的语言转化成general的路径名
    load_path = Path(__file__).parent.parent / Path('code-opt-inference') / Path('results') / Path(args.code_test_data_name)
    codes_dir = Path(__file__).parent / Path('codes') / Path(args.codes_dir_name)
    if not codes_dir.is_dir():
        codes_dir.mkdir(parents=True, exist_ok=True)
    # for lang in supported_langs:
    #     lang_dir = codes_dir / Path(opt_type) / Path(langs_map[lang])
    #     if not lang_dir.is_dir():
    #         lang_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset('json', split='train', data_files=str(load_path))
    print(dataset)

    # lang_counts = Counter(dataset['lang'])
    # for lang, count in lang_counts.items():
    #     print(f'{lang}: {count}')

    # lang_cluster_counts = Counter(dataset['lang_cluster'])
    # for lang_cluster, count in lang_cluster_counts.items():
    #     print(f'{lang_cluster}: {count}')

    # save codes of four language clusters
    for example in tqdm(dataset):
        opt_type = args.opt_type
        lang = example['lang']
        src_uid = example['src_uid']
        unopt_code_uid = example[f'{opt_type}_baseline_code_uid']
        unopt_code = example[f'{opt_type}_baseline_code']
        opt0_code = example[f'optimization_0'].split('{"optimized_code": code string}')[-1]
        opt1_code = example[f'optimization_1'].split('{"optimized_code": code string}')[-1]
        opt2_code = example[f'optimization_2'].split('{"optimized_code": code string}')[-1]
        opt3_code = example[f'optimization_3'].split('{"optimized_code": code string}')[-1]
        opt4_code = example[f'optimization_4'].split('{"optimized_code": code string}')[-1]

        # create saved directory of four language clusters codes
        if lang == 'GNU C':
            lang_dir = codes_dir / Path(opt_type) / Path('c')
        elif lang == 'GNU C++':
            lang_dir = codes_dir / Path(opt_type) / Path('cpp')
        elif lang == 'Java 8':
            lang_dir = codes_dir / Path(opt_type) / Path('java')
        elif lang == 'Python 3':
            lang_dir = codes_dir / Path(opt_type) / Path('python')
        elif lang == 'Mono C#':
            lang_dir = codes_dir / Path(opt_type) / Path('cs')
        else:
            print('Language cluster not found, use default language cluster directory.')
            lang_dir = codes_dir
        file_dir = lang_dir / Path(src_uid)
        if not file_dir.is_dir():
            file_dir.mkdir(parents=True, exist_ok=True)
        if args.parse_code==True:
            parselog_path = file_dir / Path('parse.log')
            file = logging.FileHandler(filename=parselog_path, mode='w', encoding='utf-8')
            fmt = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
            file.setFormatter(fmt)
            logger = logging.Logger(name='parse log', level=logging.DEBUG)
            logger.addHandler(file)
            opt0_code = parse_code(opt0_code, logger)
            opt1_code = parse_code(opt1_code, logger)
            opt2_code = parse_code(opt2_code, logger)
            opt3_code = parse_code(opt3_code, logger)
            opt4_code = parse_code(opt4_code, logger)

        # create saved path of four language clusters codes
        if lang == 'GNU C':
            unopt_file_path = file_dir / Path('unopt.c')
            opt0_file_path = file_dir / Path('opt0.c')
            opt1_file_path = file_dir / Path('opt1.c')
            opt2_file_path = file_dir / Path('opt2.c')
            opt3_file_path = file_dir / Path('opt3.c')
            opt4_file_path = file_dir / Path('opt4.c')
        elif lang == 'GNU C++':
            unopt_file_path = file_dir / Path('unopt.cpp')
            opt0_file_path = file_dir / Path('opt0.cpp')
            opt1_file_path = file_dir / Path('opt1.cpp')
            opt2_file_path = file_dir / Path('opt2.cpp')
            opt3_file_path = file_dir / Path('opt3.cpp')
            opt4_file_path = file_dir / Path('opt4.cpp')
        elif lang == 'Java 8':
            unopt_file_dir = file_dir / Path('unopt')
            if not opt0_file_dir.is_dir():
                unopt_file_dir.mkdir(parents=True, exist_ok=True)
            unopt_class_name, unopt_code = parse_java_class_name(unopt_code)
            unopt_file_path = unopt_file_dir / Path(f'{unopt_class_name}.java')

            opt0_file_dir = file_dir / Path('opt0')
            if not opt0_file_dir.is_dir():
                opt0_file_dir.mkdir(parents=True, exist_ok=True)
            opt0_class_name, opt0_code = parse_java_class_name(opt0_code)
            opt0_file_path = opt0_file_dir / Path(f'{opt0_class_name}.java')

            opt1_file_dir = file_dir / Path('opt1')
            if not opt1_file_dir.is_dir():
                opt1_file_dir.mkdir(parents=True, exist_ok=True)
            opt1_class_name, opt1_code = parse_java_class_name(opt1_code)
            opt1_file_path = opt1_file_dir / Path(f'{opt1_class_name}.java')   

            opt2_file_dir = file_dir / Path('opt2')
            if not opt2_file_dir.is_dir():
                opt2_file_dir.mkdir(parents=True, exist_ok=True)
            opt2_class_name, opt2_code = parse_java_class_name(opt2_code)
            opt2_file_path = file_dir / Path('opt2') / Path(f'{opt2_class_name}.java')  

            opt3_file_dir = file_dir / Path('opt3')
            if not opt3_file_dir.is_dir():
                opt3_file_dir.mkdir(parents=True, exist_ok=True)
            opt3_class_name, opt3_code = parse_java_class_name(opt3_code)
            opt3_file_path = file_dir / Path('opt3') / Path(f'{opt3_class_name}.java')

            opt4_file_dir = file_dir / Path('opt4')
            if not opt4_file_dir.is_dir():
                opt4_file_dir.mkdir(parents=True, exist_ok=True)
            opt4_class_name, opt4_code = parse_java_class_name(opt4_code)
            opt4_file_path = file_dir / Path('opt4') / Path(f'{opt4_class_name}.java')


        elif lang == 'Python 3':
            unopt_file_path = file_dir / Path('unopt.py')
            opt0_file_path = file_dir / Path('opt0.py')
            opt1_file_path = file_dir / Path('opt1.py')
            opt2_file_path = file_dir / Path('opt2.py')
            opt3_file_path = file_dir / Path('opt3.py')
            opt4_file_path = file_dir / Path('opt4.py')
        elif lang == 'Mono C#':
            unopt_file_path = file_dir / Path('unopt.cs')
            opt0_file_path = file_dir / Path('opt0.cs')
            opt1_file_path = file_dir / Path('opt1.cs')
            opt2_file_path = file_dir / Path('opt2.cs')
            opt3_file_path = file_dir / Path('opt3.cs')
            opt4_file_path = file_dir / Path('opt4.cs')
        else:
            print('Language cluster not found, use default language cluster path.')
            # file_path = file_dir / Path('code')

        with open(str(unopt_file_path), mode='w', encoding='utf-8') as file:
            file.write(unopt_code)
        with open(str(opt0_file_path), mode='w', encoding='utf-8') as file:
            file.write(opt0_code)
        with open(str(opt1_file_path), mode='w', encoding='utf-8') as file:
            file.write(opt1_code)
        with open(str(opt2_file_path), mode='w', encoding='utf-8') as file:
            file.write(opt2_code)
        with open(str(opt3_file_path), mode='w', encoding='utf-8') as file:
            file.write(opt3_code)
        with open(str(opt4_file_path), mode='w', encoding='utf-8') as file:
            file.write(opt4_code)


def parse_java_class_name(source_code):
    # find class name in the java source code
    pattern = r'public\s+(?:final\s+)?class\s+(\w+)'
    matches = re.search(pattern, source_code)
    if matches:
        class_name = matches.group(1)
    else:
        print('Class name not found, use default class name.')
        class_name = 'code'

    # if java class does not have an explicit default constructor, the compiler will generate one for it and
    # since it is implicit it may be associated with that line of code, so the possible solution is to create
    # a private constructor for the java class, since this class can no longer be instantiated outside itself,
    # jacoco no longer counts it towards its code coverage metric.
    # References: https://www.nerd.vision/post/jacoco-coverage-of-util-classes
    constructor_code = f'\n\n\tprivate {class_name}() {{}}\n'
    pattern = r'public\s+(?:final\s+)?class\s+' + class_name + r'(\s+\w+\s+\w+\s*)?(\s+//implements\s+Runnable)?\s*{'
    matches = re.search(pattern, source_code)
    if matches:
        class_definition = matches.group(0)
        source_code = source_code.replace(class_definition, class_definition + constructor_code)
    else:
        print('Class definition not found, use default source code.')
    return class_name, source_code
import re
import json
def contains_json(text):
    brackets_pattern = r".*\{\s*\"optimized_code\":.*\}.*"
    return re.match(brackets_pattern, text, re.DOTALL)!=None
def get_json(text):
    lpos = text.find("\"optimized_code\"")
    rpos = lpos
    while text.find("}", rpos+1)!=-1:
        rpos = text.find("}", rpos+1)
    json_ret = "{"+text[lpos:rpos].strip()+"}"
    return json_ret
def contain_tick(text):
    tick_pattern = r".*?(`.*`).*?"
    return re.match(tick_pattern, text, re.DOTALL)!=None
def get_tick(text):
    tick_pattern = r".*?(`.*`).*?"
    return re.findall(tick_pattern, text, re.DOTALL)[0][1:-1]

def contains_code_snippets(text):
    pattern = r"```(.+?)```"
    results = re.findall(pattern, text, re.DOTALL)
    if len(results)==0:
        return False
    else:
        return True
def get_code_content(text):
    pattern = r"```(.+?)```"
    results = re.findall(pattern, text, re.DOTALL)
    lang_patterns = [r"^python", r"^java", r"^cpp", r"^csharp", r"^c\+\+", r"^c#", r"^C#", r"^swift", r"^c", r"^C"]# todo: add other languages
    if any(re.match(lang_pattern, result) 
           for result in results 
           for lang_pattern in lang_patterns):
        for lang_pattern in lang_patterns:
            results = [re.sub(lang_pattern, '', result) for result in results]
    return results
def parse_code(text, logger):
    logging.info(f"start parsing code for:\n{text}")
    if contains_json(text):
        # print("contains json")
        logger.debug("text contains json")
        json_ret = get_json(text)
        try:# {"optimized_code":"code"}
            # print(f"try parse json")
            logger.debug(f"try to parse json...")
            code = json.loads(json_ret)['optimized_code']
            logger.debug(f"succeed to parse json")
        except Exception as e:
            # print(f"failed to parse json")
            logger.debug(f"failed to parse json")
            if contains_code_snippets(json_ret):# ```code```
                logger.debug("json text contains tripple backtick:```code```")
                code = get_code_content(text)[0].replace("\\\"","\"").replace("\\\\n","\\n").replace("\\\\t","\\t")
            elif contain_tick(json_ret):# {"optimized_code":`code`}
                # print("contains tick")
                logger.debug("json text contains backtick:`code`")
                code = get_tick(json_ret)
                code = """ """.replace(" ", code)
            else:# {"optimized_code":"multirow code"}
                # print("json text not contain tick, try to select quoted code")
                logger.debug("json text does not contain tick, try to select quoted code")
                tmp = json_ret.replace("\"optimized_code\"", "").replace("\"\"\"", "\"")# 这样好吗
                lpos = tmp.find("\"")
                rpos = lpos
                while tmp.find("\"", rpos+1)!=-1:
                    rpos = tmp.find("\"", rpos+1)
                code = tmp[lpos+1:rpos].strip()
                code = """ """.replace(" ", code).replace("\\\"","\"").replace("\\\\n","\\n").replace("\\\\t","\\t")
    elif contains_code_snippets(text):# ```code```
        # print("contains code snippets")
        logger.debug("text contains ```code snippets```")
        code = get_code_content(text)[0]
    else:
        # print("unknown pattern")
        logger.warning("unknown pattern")
        code = text
    # print(code)
    logger.info(f"parsed code:\n{code}")
    return code
        
        


# time_opt_load_path = "../../code-opt-inference/results/test/test_opt_wizardcoder_smpl_0.3_time.jsonl"
# mem_opt_load_path = "../../code-opt-inference/results/test/test_opt_wizardcoder_smpl_0.3_mem.jsonl"
# time_dataset = load_dataset('json', split='train', data_files=str(time_opt_load_path))
# mem_dataset = load_dataset('json', split='train', data_files=str(mem_opt_load_path))

# logging.basicConfig(filemode = "w", filename='parse_results_sample_0.3.log', format='%(asctime)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s',level=logging.DEBUG)

# for sample in time_dataset:
#     src_uid = sample['src_uid']
#     lang = sample['lang']
#     time_baseline_code = sample['time_baseline_code']
#     mem_baseline_code = sample['mem_baseline_code']
#     testcases = sample['testcases'] # ['input'], ['output'][0]
#     optimized_codes = [sample[f'optimization_{i}'] for i in range(5)]
#     # print(time_baseline_code)
#     logging.info(f'start parsing src_uid={src_uid}, lang={lang}, unoptimized_code:\n{time_baseline_code}')
#     for i, optimized_code in enumerate(optimized_codes):
#         # print(f'optimized {i}:\n')
#         logging.info(f'optimized {i}:\n')
#         # print(optimized_code)
#         parse_code(optimized_code)

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    main()
# python save_opt_codes.py --code_test_data_name mem_code_opt_inference_vicuna_replenish.jsonl --codes_dir_name vicuna_opt_parse --opt_type mem --parse_code
# python save_opt_codes.py --code_test_data_name time_code_opt_inference_vicuna_replenish.jsonl --codes_dir_name vicuna_opt_parse --opt_type time --parse_code

# python save_opt_codes.py --code_test_data_name mem_code_opt_inference_wizardcoder_replenish4.jsonl --codes_dir_name wizardcoder_opt_parse --opt_type mem --parse_code
# python save_opt_codes.py --code_test_data_name time_code_opt_inference_wizardcoder_replenish4.jsonl --codes_dir_name wizardcoder_opt_parse --opt_type time --parse_code

# python save_opt_codes.py --code_test_data_name mem_code_opt_data_codellama_replenish5.jsonl --codes_dir_name codellama_opt_parse --opt_type mem --parse_code
# python save_opt_codes.py --code_test_data_name time_code_opt_data_codellama_replenish5.jsonl --codes_dir_name codellama_opt_parse --opt_type time --parse_code

# python save_opt_codes.py --code_test_data_name mem_code_opt_data_gpt4.jsonl --codes_dir_name gpt4_opt_parse --opt_type mem --parse_code
# python save_opt_codes.py --code_test_data_name time_code_opt_data_gpt4.jsonl --codes_dir_name gpt4_opt_parse --opt_type time --parse_code

# python save_opt_codes.py --code_test_data_name mem_code_opt_data_gpt3.jsonl --codes_dir_name gpt3_opt_parse --opt_type mem --parse_code
# python save_opt_codes.py --code_test_data_name time_code_opt_data_gpt3.jsonl --codes_dir_name gpt3_opt_parse --opt_type time --parse_code

# python save_opt_codes.py --code_test_data_name mem_code_opt_data_starcoder.jsonl --codes_dir_name starcoder_opt_parse --opt_type mem --parse_code
# python save_opt_codes.py --code_test_data_name time_code_opt_data_starcoder.jsonl --codes_dir_name starcoder_opt_parse --opt_type time --parse_code

# python save_opt_codes.py --code_test_data_name mem_code_opt_data_llama2_replenish.jsonl --codes_dir_name llama2_opt_parse --opt_type mem --parse_code
# python save_opt_codes.py --code_test_data_name time_code_opt_data_llama2_replenish.jsonl --codes_dir_name llama2_opt_parse --opt_type time --parse_code

# python save_opt_codes.py --code_test_data_name mem_code_opt_data_palm.jsonl --codes_dir_name palm_opt_parse --opt_type mem --parse_code
# python save_opt_codes.py --code_test_data_name time_code_opt_data_palm.jsonl --codes_dir_name palm_opt_parse --opt_type time --parse_code