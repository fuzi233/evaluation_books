import os
import json
import time
import requests
import json
import sys
from tqdm import tqdm
from multiprocessing import Pool
import time
import argparse
import random 
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
    code = response_data.get("code")
    if code is not None and code != 200:
        raise ValueError(f"API error (code={code}): {response_data}")
    results = response_data.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, str) and first.strip():
            return first
        if isinstance(first, dict):
            return extract_response_text(first)
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
                 system_prompt=""):

        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.api_key = resolve_api_key()
        self.url = resolve_api_url(os.getenv("BASE_URL", ""))
            

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
                        "top_p": float(os.getenv("LLM_PLUS_TOP_P", "0.95")),
                        "top_k": int(os.getenv("LLM_PLUS_TOP_K", "40")),
                        "include_thoughts": os.getenv("LLM_PLUS_INCLUDE_THOUGHTS", "0").lower() in {"1", "true", "yes", "y"},
                        "thinking_budget": int(os.getenv("LLM_PLUS_THINKING_BUDGET", "1024")),
                        "thinking_level": os.getenv("LLM_PLUS_THINKING_LEVEL", "high"),
                        "response_format": {"result": "str"},
                    },
                }
                headers = {"Content-Type": "application/json", "X-API-Key": self.api_key}
                response = requests.request("POST", self.url, headers=headers, json=payload, timeout=(180, 180))
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
                })
                headers = {
                    'Content-Type': 'application/json',
                    "Authorization": f"Bearer {self.api_key}",
                }
                response = requests.request("POST", self.url, headers=headers, data=data, timeout=(180, 180))
            # print(response)
            response_data = response.json()

            return extract_response_text(response_data)
        except Exception as e:
            print(e)
            print('TIMEOUT. Sleeping and trying again.')
            time.sleep(3)
            return 'TIMEOUT'

def process_the_characters(char_sum):
    chars = char_sum.split("\n\n")
    rights = []
    for char in chars:
        # not mentioned remove
        if("not mentioned" in char.lower()):
            continue
        if("---" in char):continue
        if("Overall Experience" not in char or "Profile" not in char):
            if("**New Character" not in char):continue
        rights.append(char)
    return "\n\n".join(rights[:8])

def get_excerpt(content, max_length=500):
    t_content = []
    para_length = 0
    for i,t_line in enumerate(content):
        if("chapter " in t_line.lower()):continue
        t_len = len(content[i].split(" "))
        para_length+=t_len
        if(para_length>max_length):
            break
        t_content.append(content[i].replace("\n"," ").strip())
    return "\n".join(t_content[:])

def process_response(response):
    response = response.replace("Updated Review","Review")
    response = response.replace("Updated Assessment","Overall Assessment")
    response = response.replace("Updated Score","Score")
    detailed_reviews = response.split("\n")
    left_reviews = []
    for t_review in detailed_reviews:
        if("current review" not in t_review.lower() and "current assessment" not in t_review.lower()):left_reviews.append(t_review)
    try:
        return "\n".join(left_reviews)
    except:
        print("error")
        return ""

def evaluate_book(args):
    tfile, run_model, book_loc, sum_loc, prompt_template, all_book_info, output_loc = args
    t_title = all_book_info[tfile]["title"]
    try:
        t_premise = str(all_book_info[tfile]["premise"])
    except:
        t_premise = str(all_book_info[tfile]["basic"])
    temp_genres = all_book_info[tfile]["genres"][0:3]

    t_genre = ", ".join(temp_genres)
    openai_model = OpenAIClient(model=run_model,
                                temperature=0,
                                system_prompt="")

    chaps = json.load(open(book_loc + tfile + ".json"))["chaps"]
    sum_infos = json.load(open(sum_loc + tfile + ".json"))
    prompt_template = open(prompt_template).read()

    # get the prompt
    char_summary = process_the_characters(sum_infos[-1]["overall_char"])
    event_summary = sum_infos[-1]["overall_sum"]
    event_summary  = event_summary.replace("In this segment, ","Later, ") 

    
    chap_num = len(chaps)
    chap_i = random.randrange(0, chap_num, 1) #0

    t_excerpt = get_excerpt(chaps[chap_i]["content"],max_length=500)
        
    t_prompt = prompt_template.replace("{Title}",t_title).replace("{Genre}",t_genre).replace("{Premise}",t_premise).replace("{Plot_Summary}",event_summary).replace("{Character_Summary}",char_summary).replace("{Excerpt}",t_excerpt)

    #print("input",t_prompt)

    # evaluate the first chapter
    response = 'TIMEOUT'
    while(response == 'TIMEOUT'):
        response = openai_model.chat(str(t_prompt))

    print("output",response)

    with open(output_loc +"{}: {}.txt".format(all_book_info[tfile]["score"], tfile), "w") as fout:
        fout.write(response)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_model", type=str, default="")
    parser.add_argument("--book_loc", type=str, default="../../dataset/books_json/")
    parser.add_argument("--sum_loc", type=str, default="../../dataset/summaries/")
    parser.add_argument("--output_loc", type=str, default="../outputs/sum_based/gpt-4o/run_1/")
    parser.add_argument("--prompt_template", type=str, default="../prompt_template/no_criteria.txt")
    parser.add_argument("--book_info_loc", type=str, default="../../dataset/all_book_info.json")
    parser.add_argument("--book_ids_loc", type=str, default="")
    parser.add_argument("--env_file", type=str, default="")
    parser.add_argument("--processes", type=int, default=50)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    load_env_file(args.env_file)
    if not args.run_model:
        args.run_model = os.getenv("MODEL", "gpt-4o")

    sum_loc = args.sum_loc
    book_info_loc = args.book_info_loc

    if args.debug:
        book_info_loc = "../../dataset/book_debug_info.json"
        sum_loc = "../../dataset/summaries_debug/"

    all_book_info = json.load(open(book_info_loc))

    if not os.path.exists(args.output_loc):
        os.makedirs(args.output_loc)
    processed_files = os.listdir(args.output_loc)

    if(args.debug):
        files_to_eval = os.listdir(sum_loc)
        files_to_eval = [tfile.split(".json")[0] for tfile in files_to_eval]
    else:
        custom_book_ids = load_book_ids(args.book_ids_loc)
        if custom_book_ids is not None:
            files_to_eval = custom_book_ids
        elif os.path.exists(book_info_loc):
            files_to_eval = list(all_book_info.keys())
        else:
            test_files = json.load(open("../../dataset/test_infos.json"))
            files_to_eval = []
            for genre in test_files.keys():
                files_to_eval += [tfile["id"] for tfile in test_files[genre]]

    files_with_args = []

    for tfile in files_to_eval:
        if("{}: {}.txt".format(all_book_info[tfile]["score"], tfile) in processed_files):
            print("processed",tfile)
            continue
        files_with_args.append((tfile, args.run_model, args.book_loc, sum_loc, args.prompt_template, all_book_info, args.output_loc))

    # evaluate
    with Pool(processes=args.processes) as pool:
        results = pool.map(evaluate_book, files_with_args)
