import os
from collections import defaultdict
from diffusers import DDIMScheduler, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, AutoencoderKL
from transformers import pipeline
from safetensors.torch import load_file
from compel import Compel
from diffusers.utils import load_image
import torch
from PIL import Image 
import utils

# For Super-resolution module
import argparse
import cv2
import glob
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
import time
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import numpy as np
sdsd


def load_lora(pipe, dict_lora):
    lora_path = dict_lora[0]
    weight = dict_lora[1]
    # print(lora_path)
    # print(weight)
    pipe._lora_scale = weight
    state_dict, network_alphas = pipe.lora_state_dict(
        lora_path,
    )

    for key in network_alphas:
        network_alphas[key] = network_alphas[key] * weight
    
    pipe.load_lora_into_unet(
        state_dict,
        network_alphas = network_alphas,
        unet = pipe.unet
    )

    pipe.load_lora_into_text_encoder(
        state_dict = state_dict,
        network_alphas = network_alphas,
        text_encoder = pipe.text_encoder
    )

    return pipe

def init(mode):

    # model name
    if mode == 'R':
        model_name = 'RealESRGAN_x2plus'
    elif mode == 'C':
        model_name = 'RealESRGAN_x4plus_anime_6B'
    elif mode == 'SR':
        model_name = 'realesr-general-x4v3'
    elif mode == 'SC':
        model_name = 'realesr-animevideov3'
    model_name = model_name.split('.')[0]
    print(model_name)
    if model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=0)

    return upsampler

def initiate_pipe(controlnet, device="cuda:0", webtoon=False, **dict_init):
    model_path = dict_init["model_path"]     

    ## pipeline instantiation
    if webtoon is False:
        pipe_i2i = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                    model_path, 
                    controlnet=controlnet, 
                    torch_dtype=torch.bfloat16, 
                    safety_checker=None, 
                    requires_safety_checker=False
                    ).to(device)
    else:
        pipe_i2i = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                    model_path, 
                    controlnet=controlnet, 
                    torch_dtype=torch.bfloat16, 
                    vae=AutoencoderKL.from_single_file("/hdd/jinnnn/GMIII.vae.pt"),
                    safety_checker=None, 
                    requires_safety_checker=False
                    ).to(device)
    pipe_i2i.scheduler = DDIMScheduler.from_config(pipe_i2i.scheduler.config)  

    ## ClipSkip=2
    # clip_layers = pipe_i2i.text_encoder.text_model.encoder.layers
    # clip_skip = 2
    # if clip_skip > 0:
    #     pipe_i2i.text_encoder.text_model.encoder.layers = clip_layers[:-clip_skip]
 
    return pipe_i2i


def inference_pipe(pipe_i2i, rsz_image, control_image, prompt_embeds, negative_prompt_embeds, device="cuda:0", **infer):
    num_inference_steps = infer["num_inference_steps"]
    num_images_per_prompt = infer["num_images_per_prompt"]
    guidance_scale = infer["guidance_scale"]
    controlnet_conditioning_scale = infer["controlnet_conditioning_scale"]
    controlnet_guidance_start = infer["controlnet_guidance_start"]
    controlnet_guidance_end = infer["controlnet_guidance_end"]
    strength = infer["strength"]
    seed = infer["seed"]
    print(num_inference_steps, num_images_per_prompt, guidance_scale, controlnet_conditioning_scale, controlnet_guidance_start, controlnet_guidance_end, strength)
    pipe_i2i.safety_checker = None
    pipe_i2i.requires_safety_checker = False
    generated = pipe_i2i(
            control_image = control_image,
            image=rsz_image,
            prompt_embeds=prompt_embeds, 
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps, 
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale = controlnet_conditioning_scale,
            control_guidance_start = controlnet_guidance_start,
            control_guidance_end = controlnet_guidance_end,
            strength=strength,
            generator=torch.Generator(device=device).manual_seed(seed)
    ).images
    return generated



if __name__ == '__main__':

    # enable xformers memory attention
    # pipe.enable_xformers_memory_efficient_attention()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%0
    

    #%% Parameters
    device = "cuda:0"   

    yaml_real = "/home/jinnnn/etriworkspace/pipe_param_real.yaml"
    yaml_anime = "/home/jinnnn/etriworkspace/pipe_param_anime.yaml"
    yaml_webtoon = "/home/jinnnn/etriworkspace/pipe_param_webtoon.yaml"
    infer_config = "/home/jinnnn/etriworkspace/pipe_config.yaml"

    dict_pipe_real = utils.load_param(yaml_real)
    dict_pipe_anime = utils.load_param(yaml_anime)
    dict_pipe_webtoon = utils.load_param(yaml_webtoon)
    dict_infer = utils.load_param(infer_config)

    content_path = dict_infer["content_path"]
    output_path = dict_infer["output_path"]
    negative_prompt = dict_infer["negative_prompt"]


    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #%% ControlNet  ### (common)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.bfloat16)
    low_threshold = 50
    high_threshold = 100


    #%% Initiate Pipelines
    pipe_i2i_real = initiate_pipe(controlnet, device, **dict_pipe_real["init_pipe"])
    pipe_i2i_anime = initiate_pipe(controlnet, device, **dict_pipe_anime["init_pipe"])
    pipe_i2i_webtoon = initiate_pipe(controlnet, device, webtoon=True, **dict_pipe_webtoon["init_pipe"])

    #%% 
    # list_pipe_i2i = [None, pipe_i2i_oil, pipe_i2i_watercolor, pipe_i2i_ink, pipe_i2i_webtoon]
    list_pipe_i2i = [pipe_i2i_real, pipe_i2i_anime, pipe_i2i_webtoon]

    dict_from_type_idx = {
        "real" : 0,
        "anime" : 1,
        "webtoon" : 2
    }

    upsampler_r = init('R')
    upsampler_c = init('C')

    #%% Generation Loop
    while True:

        user_input = input("Enter image name to use(Including Extension):")
        image_base = content_path + user_input

        if not os.path.exists(image_base):
            print(f"Directory '{image_base}' does not exist.")
            continue

        ## change parameters
        from_type, style = utils.change_param()
        running_type = from_type +  "_to_" + style
        print("From " + from_type + " to " + style + '!')
        if style == "webtoon":
            from_type = "webtoon"
        from_type_idx = dict_from_type_idx[from_type]
        pipe_i2i = list_pipe_i2i[from_type_idx]
        if from_type_idx != 2:
            dict_lora = dict_infer[running_type]["lora_info"]
            pipe_i2i = load_lora(pipe_i2i, dict_lora)  
        positive_prompt = dict_infer[running_type]["prompt"]
        compel_proc = Compel(tokenizer=pipe_i2i.tokenizer, text_encoder=pipe_i2i.text_encoder)

        positive_prompt = utils.get_input(positive_prompt)

        ## transform text prompt to embeddings
        prompt_embeds = compel_proc([positive_prompt])
        negative_prompt_embeds = compel_proc([negative_prompt])
        print(f"content_path :\n {image_base}\n")
        print(f"positive_prompt :\n {positive_prompt}")

        ## load / pre-process image    
        init_image = load_image(image_base).convert("RGB")
        rsz_image = utils.resize_keep_ratio(init_image)


        ## controlnet image (canny)
        control_image = utils.to_canny_image(rsz_image, low_threshold, high_threshold)

        ## inference image
        generated = inference_pipe(pipe_i2i, rsz_image, control_image, prompt_embeds, negative_prompt_embeds, device, **dict_infer[running_type])
        grid = utils.image_grid(generated, 1, 4)
        grid.save(output_path + running_type + "_" + os.path.splitext(user_input)[0] + '.png', format='PNG')
        image_num = input("Select Image (1 / 2 / 3 / 4) :")
        image_to_SR = generated[int(image_num)-1]
        image_to_SR_array = np.array(image_to_SR)
        image_to_SR_array_BGR = cv2.cvtColor(image_to_SR_array, cv2.COLOR_RGB2BGR)
        if from_type_idx == 0:
            SR_output, _ = upsampler_r.enhance(image_to_SR_array_BGR, outscale=2)
        else:
            SR_output, _ = upsampler_c.enhance(image_to_SR_array_BGR, outscale=2)
        cv2.imwrite(output_path + running_type + "_" + os.path.splitext(user_input)[0] + "_SR" + ".png", SR_output)
        print('Done generation!')
        if from_type_idx != 2:
            pipe_i2i = pipe_i2i.unload_lora_weights()

        flag_quit = input("\n\nq to quit / enter to continue : ")
        list_quit = ['q', "quit"]
        if flag_quit.lower() in list_quit:
            break
            
#%%
