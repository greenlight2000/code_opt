import os
from pathlib import Path
import pandas as pd
import re

load_dir = Path(__file__).parent.parent / Path('codes') / Path('starcoder_opt_parse')

for lang in ['c','cpp','python','cs']:#,'cs'
    err_flag = False
    for opt_type in ['mem','time']:
        err_flag = False
        pass_rate = 0
        success_opt_rate = 0
        lang_dir = load_dir / Path(opt_type) / Path(lang)
        src_dirs = os.listdir(lang_dir)
        for src_dir in src_dirs:
            print(f"{lang}-{src_dir}")
            # 判断文件是否存在
            if not os.path.exists(lang_dir / Path(src_dir) / Path('codes_perf.csv')):
                print('------------------------------------')
                print(f"{lang}-{opt_type} is not fully tested")
                print('------------------------------------')
                err_flag = True
                break
            df = pd.read_csv(lang_dir / Path(src_dir) / Path('codes_perf.csv'),index_col=0)
            unopt = df.loc['unopt']
            if opt_type == 'mem':
                if "[" in unopt['mean_peak_mem']:
                    s = re.sub(r'\s+', ' ', unopt['mean_peak_mem'])
                    perf_li = s.replace("]","").replace("[","").strip().split(' ')
                    # print(perf_li)
                    # 从字符串解析数值
                    s = "0.00e+00"
                    num = float(s)
                    perf_li = [float(x) for x in perf_li]
                else:
                    perf_li = [float(x) for x in unopt['mean_peak_mem'].split(',')]
            elif opt_type == 'time':
                if "[" in unopt['mean_cpu_time']:
                    s = re.sub(r'\s+', ' ', unopt['mean_cpu_time'])
                    perf_li = s.replace("]","").replace("[","").strip().split(' ')
                    # print(perf_li)
                    perf_li = [float(x) for x in perf_li]
                else:
                    perf_li = [float(x) for x in unopt['mean_cpu_time'].split(',')]

            print(f"unopt {opt_type} performance: {min(perf_li)}~{max(perf_li)}")
            
            # 统计通过率
            pass_flag = False
            success_flag = False
            for opt_idx in range(5):
                opt = df.loc[f'opt{opt_idx}']
                if opt['pass_rate'] == 100.0:
                    pass_flag = True
                    perf = opt['mean_peak_mem'] if opt_type == 'mem' else opt['mean_cpu_time']
                    if float(perf) < min(perf_li):
                        print(f'opt{opt_idx} {opt_type} performance: {perf}. success to opt')
                        success_flag = True
                    else:
                        print(f'opt{opt_idx} {opt_type} performance: {perf}. failed to opt')
            if pass_flag:
                pass_rate += 1
            if success_flag:
                success_opt_rate += 1
        if err_flag:
            continue
        print('------------------------------------')
        print(f'{opt_type} pass_rate: {pass_rate/len(src_dirs)}')
        print(f'{opt_type} success_opt_rate: {success_opt_rate/len(src_dirs)}')
        print('------------------------------------')


