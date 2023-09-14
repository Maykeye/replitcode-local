from flask import Flask, request, jsonify
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from auto_gptq import AutoGPTQForCausalLM
from optparse import OptionParser 

opt_parser = OptionParser()
opt_parser.add_option(
    "-m", "--model",
    action="store", dest="model", type="string",
    help="model path or id.")
opt_parser.add_option(
    "-l", "--loader",
    action="store", dest="loader", type="string",
    help="Loader. One of [autogptq, transformers]"
)
opt_parser.add_option(
    "-t", "--template",
    action="store", dest="template", type="string",
    help="template file")

options, args = opt_parser.parse_args()

assert options.model, "Model required"
assert options.loader, "Loader required"
assert options.template, "Template required"

print("Loading template")
KEY = "{{|INPUT_CODE|}}"
template = Path(options.template).read_text().strip()
assert KEY in template, f"Make sure template has {KEY}"

print("Loading model")
if options.loader == "autogptq":
    model_dir = Path(options.model)
    assert model_dir.is_dir(), "must point at dir"
    weights = list(model_dir.glob("*.safetensors"))
    assert len(weights), "no safetensors found"
    weights_basename = weights[0].name

    
    model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path = options.model, 
            trust_remote_code=True,
            use_safetensors=True,
            device="cuda:0")
elif options.loader == "transformers":
    model = AutoModelForCausalLM.from_pretrained(options.model, trust_remote_code=True).to("cuda")
else:
    raise ValueError("Error: invalid loader")

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    options.model,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    truncation_side="left",
)
model.eval()
print("NN initialization complete")
#pretrained_model_name_or_path="replit/replit-code-v1-3b",


MAX_TOKENS_LENGTH = 1024

app = Flask(__name__)
set_seed(0)

# Define the API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request
    prompt = template.replace(KEY, request.json["input_data"])
    x = tokenizer.encode(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENS_LENGTH
    )
    x = x.to(device="cuda:0")

    y = model.generate(
        inputs=x,
        max_new_tokens=50,
        temperature=0.2,
        top_p=0.9,
        top_k=4,
        use_cache=True,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_code = tokenizer.decode(
        y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    generated_code = generated_code.replace(prompt, "")
    generated_code = generated_code.rsplit("\n\n", -1)[0]

    # Return the prediction as a JSON response
    return jsonify({"generated_code": generated_code})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
