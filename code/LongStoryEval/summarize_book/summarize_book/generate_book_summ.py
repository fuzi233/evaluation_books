import os
import json
import time
import requests
import json
from tqdm import tqdm
import argparse
from multiprocessing import Pool
from pathlib import Path


def load_env_file(env_file):
    if not env_file:
        return
    env_path = Path(env_file)
    if not env_path.exists():
        raise FileNotFoundError(f"env file not found: {env_file}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def resolve_api_url(base_url):
    if not base_url:
        return ""
    base_url = base_url.strip().rstrip("/")
    if "llm-plus" in base_url.lower() or "llm_plus" in base_url.lower() or base_url.endswith("/generate"):
        return base_url
    if base_url.endswith("/chat/completions") or base_url.endswith("/completions"):
        return base_url
    return base_url + "/v1/chat/completions"


def resolve_api_key():
    return (
        os.getenv("API_KEY_OVERRIDE")
        or os.getenv("LLM_PLUS_API_KEY")
        or os.getenv("API_KEY_2")
        or os.getenv("API_KEY")
        or ""
    )


def is_llm_plus_url(url):
    lowered = url.lower()
    return "llm-plus" in lowered or "llm_plus" in lowered or lowered.endswith("/generate")


def extract_response_text(response_data):
    if isinstance(response_data, str):
        return response_data
    if not isinstance(response_data, dict):
        raise ValueError(f"Unsupported response type: {type(response_data).__name__}")
    try:
        value = response_data["choices"][0]["message"]["content"]
        if isinstance(value, str):
            return value
    except Exception:
        pass
    for key in ("text", "output_text", "content", "response", "result", "answer"):
        value = response_data.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return extract_response_text(value)
    raise KeyError("Could not extract text from API response")


def load_book_ids(book_ids_loc):
    if not book_ids_loc:
        return None
    if book_ids_loc.endswith(".json"):
        loaded = json.load(open(book_ids_loc, "r", encoding="utf-8"))
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict):
            return list(loaded.keys())
        raise ValueError("book_ids json must be a list or dict")
    return [line.strip() for line in open(book_ids_loc, "r", encoding="utf-8") if line.strip()]

class OpenAIClient():
    def __init__(self,
                 model="gpt-4o",
                 temperature=0.0,
                 top_p=1,
                 system_prompt=""):

        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.top_p = top_p
        self.url = resolve_api_url(os.getenv("BASE_URL", ""))
        self.api_key = resolve_api_key()

    def chat(self, prompt):
        try:
            if is_llm_plus_url(self.url):
                payload = {
                    "model": self.model,
                    "contents": [prompt],
                    "system_prompt": self.system_prompt,
                    "extra": {
                        "temperature": self.temperature,
                        "max_output_tokens": int(os.getenv("LLM_PLUS_MAX_OUTPUT_TOKENS", "4096")),
                        "top_p": self.top_p,
                        "top_k": int(os.getenv("LLM_PLUS_TOP_K", "40")),
                        "include_thoughts": os.getenv("LLM_PLUS_INCLUDE_THOUGHTS", "0").lower() in {"1", "true", "yes", "y"},
                        "thinking_budget": int(os.getenv("LLM_PLUS_THINKING_BUDGET", "1024")),
                        "thinking_level": os.getenv("LLM_PLUS_THINKING_LEVEL", "high"),
                        "response_format": {"result": "str"},
                    },
                }
                headers = {"Content-Type": "application/json", "X-API-Key": self.api_key}
                response = requests.post(self.url, headers=headers, json=payload, timeout=(180, 180))
            else:
                data = json.dumps({
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                })
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
                response = requests.post(self.url, headers=headers, data=data,timeout=(180, 180))         
            response_data = response.json()
            return extract_response_text(response_data)
        except Exception as e:
            print(e)
            print('TIMEOUT. Sleeping and trying again.')
            time.sleep(3)
            return 'TIMEOUT'


def process_response(response):
    cur_summ, overall_sum, cur_char, all_char = "", "", "", ""
    try:
        if("\n\n### Characters:" in response):
            summ, char = response.split("\n\n### Characters:")
            char = "\n".join(char.split("\n")[1:])
        else:
            summ = response
            char = ""
        
        summ = summ.replace("### Updated Plot Summary","### Overall Plot Summary")
        if ("\n\n### Overall Plot Summary" in summ):
            cur_summ, overall_sum = summ.split("\n\n### Overall Plot Summary")
            overall_sum = "\n".join(overall_sum.split("\n")[1:])
            if("### Summary of Current Segment" in cur_summ):
                cur_summ = "\n".join(cur_summ.split("\n")[1:])
        else:
            if ("### Plot Summary" in summ):
                overall_sum = "\n".join(summ.split("\n")[1:])
            else:
                overall_sum = summ
            cur_summ = overall_sum
        if ("\n- **Current Experience" in char):
            chars = char.split("\n- **Current Experience")
            all_char = chars[0]
            cur_char = char
            for tchar in chars[1:20]:
                all_char += "\n"
                all_char += "\n".join(tchar.split("\n")[1:])
        else:
            all_char = char
            cur_char = all_char
    except Exception as e:
        print(e)
    if("\n---" in cur_summ):cur_summ = cur_summ.replace("\n---","")
    if("\n---" in overall_sum):overall_sum = overall_sum.replace("\n---","")
    return cur_summ.replace("\n\n", "\n"), overall_sum.replace("\n\n", "\n"), cur_char, all_char


def summarize_book(args):
    tfile,file_loc,run_model,prompt_loc,out_loc = args
    openai_model = OpenAIClient(model=run_model,
                                temperature=0,
                                system_prompt="")
    try:
        chaps = json.load(open(file_loc +"/"+ tfile))["chaps"]
    except:
        print("no chaps for ", tfile)
        return

    promt_loc = prompt_loc
    prompt_start = open(promt_loc+"/beginning.txt").read()
    prompt_update = open(promt_loc+"/update.txt").read()
    summ_info = []
    t_sum = ""
    char = ""
    
    # step 1: combine the short chapters
    chap_contents = []
    long_chap = ""
    for t_chap in chaps[:]:
        chap_c = "\n".join(t_chap["content"])
        if (len(t_chap["title"]) > 0): chap_c = "### "+ t_chap["title"] + "\n\n" + chap_c
        if(long_chap!=""):long_chap+= ("\n\n\n"+chap_c)
        else:long_chap=chap_c
        if(len(str(long_chap).split(" "))>4096):
            chap_contents.append(long_chap)
            long_chap =""
    if(long_chap!=""):chap_contents.append(long_chap)
    

    # step 2: summarize the first chapter
    chap_1 = chap_contents[0]
    response = openai_model.chat(str(prompt_start.replace("{Beginning}", chap_1)))
    while (response == 'TIMEOUT'):
        response = openai_model.chat(str(prompt_start.replace("{Beginning}", chap_1)))
    cur_summ, overall_sum, cur_char, overall_char = process_response(response)
    summ_info.append({"response": response, "cur_sum": cur_summ, "overall_sum": overall_sum, "cur_char": cur_char,
                      "overall_char": overall_char})

    # continue the summarization
    for chap_i,chap_1 in enumerate(tqdm(chap_contents[1:])):
        temp_update_prompt = prompt_update.replace("{Beginning}", chap_1).replace("{Chars}", overall_char).replace("{Sum}", overall_sum)
        if(chap_i==len(chap_contents)-2):
            if("epilogue" not in temp_update_prompt.lower()):
                temp_update_prompt = "### Epilogue\n\n" + temp_update_prompt
            print("now is the ending of {}".format(tfile))
        response = openai_model.chat(temp_update_prompt)
        while (response == 'TIMEOUT'):
            response = openai_model.chat(temp_update_prompt)
        cur_summ, overall_sum, cur_char, overall_char = process_response(response)

        if(overall_sum=="" or len(overall_sum.split(" "))<20):overall_sum = summ_info[-1]["overall_sum"]
        if(overall_char=="" or len(overall_char.split(" "))<20):overall_char = summ_info[-1]["overall_char"]
        summ_info.append({"response": response, "cur_sum": cur_summ, "overall_sum": overall_sum, "cur_char": cur_char,
                          "overall_char": overall_char})
    with open(out_loc+"/" + tfile, "w") as fout:
        json.dump(summ_info, fout, indent=2, ensure_ascii=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_model", type=str, default="")
    parser.add_argument("--prompt_loc", type=str, default="./prompts/")
    parser.add_argument("--file_loc", type=str, default="../dataset/books_json/")
    parser.add_argument("--out_loc", type=str, default="../dataset/summaries_debug/")
    parser.add_argument("--book_ids_loc", type=str, default="")
    parser.add_argument("--env_file", type=str, default="")
    parser.add_argument("--processes", type=int, default=50)
    args = parser.parse_args()
    load_env_file(args.env_file)
    if not args.run_model:
        args.run_model = os.getenv("MODEL", "gpt-4o")
    run_model = args.run_model

    ori_files = []
    if not os.path.exists(args.out_loc):
        os.makedirs(args.out_loc)
    
    processed_files = os.listdir(args.out_loc)
    print("processed_files", len(processed_files))
    
    file_loc = args.file_loc
    selected_book_ids = load_book_ids(args.book_ids_loc)
    if selected_book_ids is not None:
        ori_files = [book_id + ".json" for book_id in selected_book_ids]
    else:
        ori_files = os.listdir(file_loc)
    for ti, fname in enumerate(ori_files):
        ori_files[ti] = file_loc +"/"+ fname
    # sort the book files by size
    sorted_file_paths = sorted(ori_files, key=os.path.getsize)
    files_with_args = []
    for file_path in sorted_file_paths:
        if(file_path.split("/")[-1] in processed_files):continue
        files_with_args.append((file_path.split("/")[-1], args.file_loc, args.run_model, args.prompt_loc, args.out_loc))
    
    with Pool(processes=args.processes) as pool:
        results = pool.map(summarize_book, files_with_args)
