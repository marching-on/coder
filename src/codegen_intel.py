from transformers import AutoTokenizer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, BitsAndBytesConfig, RtnConfig 

class Coder:
    def __init__(self, device="cuda"):
        self.device = device
        if self.device == "cuda":
            self.woq_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        else:
            self.woq_config = RtnConfig(bits=4, compute_dtype="int8")
        #device = "cuda" # the device to load the model onto
        self.generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=1)
        self.tokenizer = AutoTokenizer.from_pretrained("./models--Qwen--CodeQwen1.5-7B/snapshots/5ce5a1554e50a9e3bb236de7c0b8a2a1746186e4/")
        self.model = AutoModelForCausalLM.from_pretrained("./models--Qwen--CodeQwen1.5-7B/snapshots/5ce5a1554e50a9e3bb236de7c0b8a2a1746186e4/", 
                                              quantization_config=self.woq_config)
    def infer(self, query, history, max_new_tokens=512):
        # history to be added later
        input_text = "# " + query + " 下面开始编码"
        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens, **self.generate_kwargs)[0]
        # The generated_ids include prompt_ids, so we only need to decode the tokens after prompt_ids.
        output_text = self.tokenizer.decode(generated_ids[len(model_inputs.input_ids[0]):], skip_special_tokens=True)
        
        new_conversation = f'''
            user:{query}
            assistant:{output_text}
            '''
        history.append(new_conversation)

        return output_text, history
