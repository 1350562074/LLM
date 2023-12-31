import os
import sys
import json
import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
<<<<<<< HEAD
    base_model: str = "./Llama2-7b-chat/",
=======
    base_model: str = "../Chinese-Llama-2-7b/Llama2-7b-chat/",
>>>>>>> a3716ed1355ea26820eb563c9d6171195e8b83c7
    lora_weights: str = "./checkpoints_XXX/",
    prompt_template: str = "llama2",  # The prompt template to use, will default to alpaca.
    temperature: float=0.1,
    top_p: float=0.75,
    top_k: int=40,
    num_beams: int=4,
<<<<<<< HEAD
    max_new_tokens: int=512,
=======
    max_new_tokens: int=128,
>>>>>>> a3716ed1355ea26820eb563c9d6171195e8b83c7
    stream_output: bool=False,
    **kwargs,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
<<<<<<< HEAD
    tokenizer.padding_side = "left"
=======
>>>>>>> a3716ed1355ea26820eb563c9d6171195e8b83c7

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    with open('./final_eval_sft.json','r',encoding='utf-8') as f:
        data = json.load(f)

      
<<<<<<< HEAD
    f = open('./512_res_XXX.json','a+',encoding='utf-8')
=======
    f = open('./res_XXX.json','a+',encoding='utf-8')
>>>>>>> a3716ed1355ea26820eb563c9d6171195e8b83c7
    new_data = []
    instruction="Please refactor the following code:"
    
    for temp in tqdm(data):
        input = temp["input"]
        code_after = temp['output']
        # print("code_after:",code_after)

        if isinstance(input , tuple):
            input  = input[0]
            code_after = code_after[0]

        # print("input:",input)
        
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        # print("len:",len(input_ids[0]))
        
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=2.0,
            **kwargs,
        )
        
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
<<<<<<< HEAD
            "max_new_tokens": max_new_tokens,
            #"max_new_tokens": len(input_ids[0]),
=======
            #"max_new_tokens": max_new_tokens,
            "max_new_tokens": len(input_ids[0]),
>>>>>>> a3716ed1355ea26820eb563c9d6171195e8b83c7
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        # print("output:",output) 
        output = prompter.get_response(output)
        # print("output:",output) 
        
        
        d = {"instruction":instruction, "input":input, "ground_truth":code_after,"output":output}
        # print("d:",d) 
        f.write(str(d)+"\n")
        new_data.append({"instruction":instruction,"input":input, "ground_truth":code_after,"output":output})
    #print(11111)    
<<<<<<< HEAD
    with open('./512_all_XXX_all.json','w+',encoding='utf-8') as f:
=======
    with open('./all_XXX_all.json','w+',encoding='utf-8') as f:
>>>>>>> a3716ed1355ea26820eb563c9d6171195e8b83c7
        json.dump(new_data,f)
    f.close()
    
        
        

    
    

if __name__ == "__main__":
    fire.Fire(main)
