import os
import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import json
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import os
from datasets import load_from_disk,load_dataset
import torch
import json
from tqdm import tqdm
import re	

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm

def visualize_attention(multihead_attention, output_path="atten_map_1.png", title="Layer 5"):
    # Assuming the input is a numpy array of shape (1, num_heads, n_tokens, n_tokens)
    # First, we average the attention scores over the multiple heads
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()  # Shape: (n_tokens, n_tokens)

    # Pooling the attention scores with stride 20
    averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), 20, stride=20).squeeze(0).squeeze(0)
    
    cmap = plt.cm.get_cmap("viridis")
    plt.figure(figsize=(5, 5), dpi=400)

    # Log normalization
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())

    ax = sns.heatmap(
        averaged_attention,
        cmap=cmap,  # Custom color map
        norm=log_norm,  
    )
    
    # Dynamically adjust label density
    max_ticks = 20  # Maximum number of ticks to display
    n_ticks = averaged_attention.shape[0]
    tick_spacing = max(1, n_ticks // max_ticks)

    x_ticks = [str(i * 20) for i in range(0, n_ticks, tick_spacing)]
    y_ticks = [str(i * 20) for i in range(0, n_ticks, tick_spacing)]

    ax.set_xticks(range(0, n_ticks, tick_spacing))
    ax.set_yticks(range(0, n_ticks, tick_spacing))
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    # Adjust font size and rotation for readability
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5, rotation=0)

    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight')

    top_five_attentions = []
    for row in averaged_attention:
        # Use torch.topk to get the top 5 values and their indices
        top_values, top_indices = torch.topk(row, 10)
        # Convert to lists and append to the overall list
        top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_five_attentions.append(top_five_line)
        
    return top_five_attentions, averaged_attention
 

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-path',
        type=str,
        required=False,
        # default="../llama3-llava-next-8b",
        default="../llava-onevision-qwen2-0.5b-ov",
        help='Path to the pretrained model'
    )
    parser.add_argument(
        '--image-path',
        type=str,
        required=False,
        default="./src/LLaVA/images/llava_logo.png",
        help='Path to the image to be processed (default: "./src/LLaVA/images/llava_logo.png")'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        required=False,
        default="Describe the image in details.",
        help='Prompt to describe the image (default: "Describe the image in details.")'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=False,
        default="./output_example",
        help='Path to save the output and attention maps (default: "./output_example")'
    )
    pargs = parser.parse_args()

    class InferenceArgs:
        model_path = pargs.model_path
        model_base = None
        image_file = None
        device = "cuda"
        conv_mode = None
        temperature = 0.2
        max_new_tokens = 512
        load_8bit = False
        load_4bit = True if "llama" in model_path.lower() else False
        debug = False
        image_aspect_ratio = 'anyres'
        image_grid_pinpoints = None

    args = InferenceArgs()
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

    args.image_grid_pinpoints = model.config.image_grid_pinpoints

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    elif "llama3" in model_name.lower():
        conv_mode = "llava_llama_3"
    elif "qwen" in model_name.lower():
        conv_mode = "qwen_2"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    model.config.use_fast_v = False
    model.model.reset_fastv()

    total_layers = model.config.num_hidden_layers

    # %%
    def inference(prompts,images,append_output=""):
        outputs = []
        outputs_attention = []
        for prompt,image in tqdm(zip(prompts,images),total=len(prompts)):
            image = load_image(image)
            image_size = image.size
            image_tensor = process_images([image], image_processor, args)
            #image_tensor.shape is (1,5,3,384,384)
            #the second dimension is the number of image parts
            #save each part as an image (total is 5)
            for i in range(image_tensor.shape[1]):
                img = image_tensor[0][i].cpu().detach().numpy().transpose(1,2,0)
                img = Image.fromarray((img * 255).astype('uint8'))
                img.save("output_example/anyres_image_parts/image_part_"+str(i)+".png")
            conv = conv_templates[args.conv_mode].copy()
            conv.tokenizer = tokenizer
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            inp = prompt

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp # False
                    # inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN # False
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + append_output

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
            #1d attention mask
            attention_mask = None
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=256,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    output_attentions=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    image_sizes=[image_size],
                    pad_token_id=tokenizer.pad_token_id,
                    )
            

            output = tokenizer.decode(output_ids['sequences'][0, input_ids.shape[1]:],skip_spectial_tokens=True).strip().replace("</s>","")
            outputs.append(output)
            print(output)

            outputs_attention.append(output_ids['attentions'])
        

        return outputs,outputs_attention
    

        # %%

    # %%

    prompts = [pargs.prompt]
    images = [pargs.image_path]

    model_output_ori,outputs_attention = inference(prompts,images)
    # model_output,outputs_attention = inference(prompts,images,append_output=model_output_ori[0])

   

    output_path = pargs.output_path

    try:
        os.mkdir(output_path)
    except:
        pass

    try:
        os.mkdir(output_path+"/attn_maps")
    except:
        pass


    with open(output_path+"/output.json","w") as f:
        # json dumps
        json.dump({"prompt":pargs.prompt,"image":pargs.image_path,"output": model_output_ori},f,indent=4)

    # draw attention maps
    for i in outputs_attention:
        for j in range(0,total_layers):
            top5_attention,average_attentions = visualize_attention(i[0][j].cpu(),output_path=output_path+"/attn_maps/atten_map_"+str(j)+".png",title="Layer "+str(j+1))

    print(model_output_ori)




