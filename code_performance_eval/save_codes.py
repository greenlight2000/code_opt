import re
import argparse

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_test_data_name', default='sampled_submissions.jsonl', type=str)
    parser.add_argument('--codes_dir_name', default='palm_codes', type=str)
    args = parser.parse_args()

    return args


def main():
    supported_langs = ['GNU C', 'GNU C++', 'Java', 'Python 3', 'C#']
    langs_map = {'GNU C':'c', 'GNU C++':'cpp', 'Java':'java', 'Python 3':'python', 'C#':'cs'}# 用于把specific的语言转化成general的路径名
    load_path = Path(__file__).parent / Path('dataset') / Path(args.code_test_data_name)
    codes_dir = Path(__file__).parent / Path('codes') / Path(args.codes_dir_name)
    if not codes_dir.is_dir():
        codes_dir.mkdir(parents=True, exist_ok=True)
    for lang in supported_langs:
        lang_dir = codes_dir / Path(langs_map[lang])
        if not lang_dir.is_dir():
            lang_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset('json', split='train', data_files=str(load_path))
    dataset = dataset.filter(lambda x:x['lang']=='Java 8')
    print(dataset)

    # lang_counts = Counter(dataset['lang'])
    # for lang, count in lang_counts.items():
    #     print(f'{lang}: {count}')

    # lang_cluster_counts = Counter(dataset['lang_cluster'])
    # for lang_cluster, count in lang_cluster_counts.items():
    #     print(f'{lang_cluster}: {count}')

    # save codes of four language clusters
    for example in tqdm(dataset):
        lang_cluster = example['lang_cluster']
        code_uid = example['code_uid']
        source_code = example['source_code']

        # create saved directory of four language clusters codes
        if lang_cluster == 'C':
            lang_dir = codes_dir / Path('c')
        elif lang_cluster == 'C++':
            lang_dir = codes_dir / Path('cpp')
        elif lang_cluster == 'Java':
            lang_dir = codes_dir / Path('java')
        elif lang_cluster == 'Python':
            lang_dir = codes_dir / Path('python')
        elif lang_cluster == 'C#':
            lang_dir = codes_dir / Path('cs')
        else:
            print('Language cluster not found, use default language cluster directory.')
            lang_dir = codes_dir

        file_dir = lang_dir / Path(code_uid)
        if not file_dir.is_dir():
            file_dir.mkdir(parents=True, exist_ok=True)

        # create saved path of four language clusters codes
        if lang_cluster == 'C':
            file_path = file_dir / Path('code.c')
        elif lang_cluster == 'C++':
            file_path = file_dir / Path('code.cpp')
        elif lang_cluster == 'Java':
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

            file_path = file_dir / Path(f'{class_name}.java')
        elif lang_cluster == 'Python':
            file_path = file_dir / Path('code.py')
        elif lang_cluster == 'C#':
            file_path = file_dir / Path('code.cs')
        else:
            print('Language cluster not found, use default language cluster path.')
            file_path = file_dir / Path('code')

        with open(file_path, mode='w', encoding='utf-8') as file:
            file.write(source_code)


if __name__ == '__main__':
    args = parse_arguments()
    main()
    # python save_codes.py --code_test_data_name screened_sampled_dataset_2023-09-20_17:52:54/sampled_submissions.jsonl --codes_dir_name sample_verified

    # python save_codes.py --code_test_data_name screened_sampled_dataset_2023-10-02_23:35:49/sampled_submissions.jsonl --codes_dir_name sample_verified2
    # python save_codes.py --code_test_data_name yunzhe/1002_submissions_results_H_Java.jsonl --codes_dir_name java_test
    
