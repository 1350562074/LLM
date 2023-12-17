from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer,GenerationConfig
from tqdm import tqdm
import json
model_path = "../Chinese-Llama-2-7b/Llama2-7b-chat/"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
#streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

#instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            #If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

instruction = """[INST] {} [/INST]"""
with open('./final_eval_sft.json','r',encoding='utf-8') as f:
    data = json.load(f)

generation_config = GenerationConfig(
      temperature=0.1,
      top_p=0.75,
      top_k=40,
      num_beams=4,
      repetition_penalty=2.0)
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2


new_data = []
for temp in tqdm(data):
    code_before = temp["input"]
    code_after = temp["output"]
    if isinstance(code_before, tuple):
        code_before = code_before[0]
        code_after = code_after[0]
            
    input = temp["instruction"]+code_before


    prompt = instruction.format(input)
    inputs = tokenizer(input, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    max_new_tokens = len(input_ids[0])
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    generate_ids = model.generate(input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
    )
    s = generate_ids.sequences[0]
    output = tokenizer.decode(s)
    output = output.split('[/INST]')[1].strip()
    new_data.append({"instruction":prompt ,"input":input, "ground_truth":code_after,"output":output})

  
with open('/data/lh/LLM/all_res_llama2_ori_XXX.json','w+',encoding='utf-8') as f:
    json.dump(new_data,f)


