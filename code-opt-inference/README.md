# prerequisite
1. `cd code-opt-inference`
2. `pip install -r requirements.txt`

## GPT-3.5
code optimization:
`python scripts/eval_gpt_opt.py --api_key openai_api_key --model gpt-3.5-turbo-0613 --data_load_name code_opt_dataset.jsonl --result_save_name code_opt_data_gpt3.jsonl --log_file_name code_opt_data_gpt3.log`

## GPT-4
code optimization:
`python scripts/eval_gpt_opt.py --api_key openai_api_key --model gpt-4 --data_load_name code_opt_dataset.jsonl --result_save_name code_opt_data_gpt4.jsonl --log_file_name code_opt_data_gpt4.log`

## StarCoder
code optimization:
`python scripts/eval_starcoder_opt.py --access_token access_token --cache_dir cache_dir --checkpoint HuggingFaceH4/starchat-beta --data_load_name code_opt_dataset.jsonl --result_save_name code_opt_data_starcoder.jsonl --log_file_name code_opt_data_starcoder.log`

code summarization:
`python scripts/eval_starcoder_sum.py --access_token access_token --cache_dir cache_dir --checkpoint HuggingFaceH4/starchat-beta --data_load_name code_summarization_dataset_with_gt.jsonl --result_save_name code_sum_data_starcoder.jsonl --log_file_name code_sum_data_starcoder.log`

## LlaMA 1
code optimization:
`python scripts/eval_llama_opt.py --access_token access_token --cache_dir cache_dir --checkpoint elinas/llama-65b-hf-transformers-4.29 --data_load_name code_opt_dataset.jsonl --result_save_name code_opt_data_llama.jsonl --log_file_name code_opt_data_llama.log`
code summarization:
`python scripts/eval_llama_sum.py --access_token access_token --cache_dir cache_dir --checkpoint elinas/llama-65b-hf-transformers-4.29 --data_load_name code_summarization_dataset_with_gt.jsonl --result_save_name code_sum_data_llama.jsonl --log_file_name code_sum_data_llama.log`

## LlaMA 2
code optimization: 
`python scripts/eval_llama2_opt.py --access_token access_token --cache_dir cache_dir --checkpoint meta-llama/Llama-2-70b-chat-hf --data_load_name code_opt_dataset.jsonl --result_save_name code_opt_data_llama2.jsonl --log_file_name code_opt_data_llama2.log`