import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os.path as osp
import os
from PIL import Image
import torch
import math

def direct_resize_wo_pad(image, target_resolution):
    return image.resize(target_resolution)
def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image
# def process_anyres_image(image, processor, grid_pinpoints):
#     """
#     Process an image with variable resolutions.
#
#     Args:
#         image (PIL.Image.Image): The input image to be processed.
#         processor: The image processor object.
#         grid_pinpoints (str): A string representation of a list of possible resolutions.
#
#     Returns:
#         torch.Tensor: A tensor containing the processed image patches.
#     """
#     if type(grid_pinpoints) is list:
#         possible_resolutions = grid_pinpoints # [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
#     else:
#         possible_resolutions = ast.literal_eval(grid_pinpoints)
#     best_resolution = select_best_resolution(image.size, possible_resolutions) #  find the best fit resolution from the list of options
#     image_padded = resize_and_pad_image(image, best_resolution)  # resize and zero-padding the new image into the best_resolution
#
#     patches = divide_to_patches(image_padded, processor.crop_size['height'])
#
#     # print(f'### In anyres -- image size {image.size}, crop_size {processor.crop_size["height"]} resulted in {len(patches)} produced...')
#
#     if 'shortest_edge' in processor.size:
#         image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))
#     else:
#         image_original_resize = image.resize((processor.crop_size['height'], processor.crop_size['height']))
#
#     image_patches = [image_original_resize] + patches
#     image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
#                      for image_patch in image_patches]
#
#     image_patches = [(x if len(x.shape)==3 else x[0]) for x in image_patches]
#
#     return torch.stack(image_patches, dim=0)
#
# def process_images(images, image_processor, ):
#     # image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
#     image_aspect_ratio =  "anyres"
#     new_images = []
#     if image_aspect_ratio == 'pad':
#         for image in images:
#             image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
#             image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
#             new_images.append(image)
#     elif image_aspect_ratio == "anyres":
#         for image in images:
#             image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
#             new_images.append(image)
#     else:
#         return image_processor(images, return_tensors='pt')['pixel_values']
#     if all(x.shape == new_images[0].shape for x in new_images):
#         new_images = torch.stack(new_images, dim=0)
#     return new_images


def process_img(image_path):
    image = Image.open(image_path).convert('RGB')
    # padded_img = resize_and_pad_image(image, (336, 336))
    padded_img = direct_resize_wo_pad(image, (336, 336))
    # image_tensor = process_images([image], self.image_processor, self.model_config)[0]
    return padded_img

def apply_gauss_kernel(attention_array, ):
    """
    Apply a Gaussian kernel to the attention array.

    Args:
        attention_array (np.ndarray): The attention array to which the kernel will be applied.
        kernel_size (int): The size of the kernel.

    Returns:
        np.ndarray: The attention array after the kernel has been applied.
    """
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    kernel_size = kernel.shape[0]
    pad_size = (kernel_size - 1) // 2
    kernel = kernel / np.sum(kernel)
    kernel = np.expand_dims(np.expand_dims(kernel, axis=0), axis=0)
    kernel = torch.tensor(kernel)
    attention_array = np.expand_dims(attention_array, axis=0)
    attention_array = torch.tensor(attention_array, dtype=torch.float64)
    attention_array = attention_array.unsqueeze(0)
    attention_array = torch.nn.functional.pad(attention_array, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
    attention_array = torch.nn.functional.conv2d(attention_array, kernel, padding=0)
    # attention_array = torch.nn.functional.conv2d(attention_array, kernel, padding= 'same', padding_mode='replicate')

    attention_array = attention_array.squeeze(0)
    attention_array = attention_array.squeeze(0)
    attention_array = attention_array.numpy()
    return attention_array

if __name__ == '__main__':
    # layer_id = 15

    # dataset_name = 'mlvu'
    dataset_name = 'coco2017_cap_val'
    data_dir = f'/system/user/publicwork/lin/FastV_results/attention_scores/layer1_{dataset_name}'
    plot_dir = f'/system/user/publicwork/lin/FastV_results/attention_scores/layer1_{dataset_name}_plots'
    attention_files = glob.glob(osp.join(data_dir, '*.npy'))
    attention_files = sorted(attention_files, key=lambda x: (int(x.split('/')[-1].split('_')[0].replace('sample', '')), int(x.split('/')[-1].split('.')[0].split('_')[3])))

    os.makedirs(plot_dir, exist_ok=True)
    dict_of_list = {}
    for attention_file in attention_files:
        sample_id = int(attention_file.split('/')[-1].split('_')[0].replace('sample', ''))
        if sample_id not in dict_of_list:
            dict_of_list[sample_id] = []
        dict_of_list[sample_id].append(attention_file)
    if dataset_name == 'mlvu':
        n_plots_per_sample = 2
    elif dataset_name == 'coco2017_cap_val':
        n_plots_per_sample = 5

    for sample_id, attention_files in dict_of_list.items():


        fig = plt.figure(figsize=(40, 40))
        for idx in range(n_plots_per_sample):
            attention_file = attention_files[idx]
            filename = attention_file.split('/')[-1]
            if dataset_name == 'coco2017_cap_val':
                items = filename.split('.')[0].split('_')
                global_str, local_str = items[-2].replace('global', ''), items[-1].replace('local', '')
                global_start, global_end = int(global_str.split('to')[0]), int(global_str.split('to')[1])
                local_start, local_end = int(local_str.split('to')[0]), int(local_str.split('to')[1])
                img_token_range = np.arange(global_start, local_end-1)
                positions_to_draw_vertical_lines = [global_start, global_end,  local_end -1]
            elif dataset_name == 'mlvu':
                img_token_range = np.arange(14, 14 + 196 * 32)
                positions_to_draw_vertical_lines = list(range(14, 14 + 196 * 32 + 1, 196))

            attention = np.load(attention_file, allow_pickle=True)
            attention = attention.mean(axis=0)

            attention_img_token = attention[img_token_range]
            max_attention_img_token = np.max(attention_img_token)
            # the position of max attention in image token in the global attention sequence
            max_attention_img_token_pos = np.argmax(attention_img_token) +14
            # plot a vertical line at the position of max attention
            plt.axvline(x=max_attention_img_token_pos, color='g', linestyle='--')
            plt.text(max_attention_img_token_pos , max_attention_img_token * 0.9, f'max {max_attention_img_token_pos}', color='green', fontsize=8)



            # attention = attention[0, 0, 0, :]
            plt.subplot(n_plots_per_sample, 1,  idx + 1)
            plt.plot(attention)
            plt.title(f'{filename} max attention {max_attention_img_token} at {max_attention_img_token_pos}')
            plt.xlabel('Token ID')
            plt.ylabel('Attention Score')
            plt.ylim(0, max_attention_img_token)

            for frame_id, pos in enumerate(positions_to_draw_vertical_lines):
                plt.axvline(x=pos, color='r', linestyle='--')
                # add the text of pos id
                plt.text(pos, max_attention_img_token * 0.9, str(frame_id), color='red', fontsize=8)

        plt.tight_layout()
        save_filename = '_'.join( filename.split('.')[0].split('_')[:-2])
        plt.savefig(osp.join(plot_dir, f'{save_filename}.png'))


    #
    #
    # kernel_type = [ 'gauss' ][0]
    # normalized_ = [False, True][0]
    # normalized_str = ['unnormalized', 'normalized'][normalized_]
    # for layer_id in [0,1,2,3,4,5]:
    #     data_dir = f'/system/user/publicdata/LMM_benchmarks/SEED-Bench/answers_generate_5008/xx_anyres_debug/{layer_id}'
    #
    #     attention_file = osp.join( data_dir,   f'attn_weights_{layer_id}.npy')
    #     attention = np.load(attention_file)
    #     t = 1
    #     n_heads = attention.shape[1]
    #     img_path = '/system/user/publicdata/LMM_benchmarks/SEED-Bench/answers_generate_5008/xx/124217_564854171'
    #     # target_token_id = 657
    #
    #
    #     padded_img = process_img(img_path)
    #
    #     # for target_token_id in range(-47, 0):
    #     for target_token_id in [-1]:
    #
    #         subfolder_blended = osp.join(data_dir, normalized_str,  f'attention_target_token{target_token_id}_att_global_blended')
    #         subfolder_seq = osp.join(data_dir,  normalized_str, f'attention_target_token{target_token_id}_seq')
    #         subfolder_att_global = osp.join(data_dir, normalized_str, f'attention_target_token{target_token_id}_att_global')
    #         subfolder_att_local_collage = osp.join(data_dir, normalized_str,  f'attention_target_token{target_token_id}_local_collage')
    #
    #         subfolder_att_global_w_kernel = osp.join(data_dir, normalized_str, f'attention_target_token{target_token_id}_att_global_w_{kernel_type}')
    #         subfolder_att_local_collage_w_kernel = osp.join(data_dir, normalized_str, f'attention_target_token{target_token_id}_local_collage_w_{kernel_type}')
    #
    #
    #
    #
    #         for folder_name in [subfolder_blended, subfolder_seq, subfolder_att_global, subfolder_att_local_collage,
    #                             subfolder_att_local_collage_w_kernel, subfolder_att_global_w_kernel]:
    #             if not osp.exists(folder_name):
    #                 os.makedirs(folder_name)
    #         for head_id in range(n_heads):
    #
    #             if False:
    #                 attention_seq_full = attention[0, head_id, target_token_id, :]
    #                 # plot the attention seq
    #                 plt.figure(figsize=(10, 10))
    #                 plt.plot(attention_seq_full)
    #                 plt.title(f'Attention head {head_id} target token {target_token_id}')
    #                 plt.savefig(osp.join(subfolder_seq, f'attentionhead_{head_id}_target_token{target_token_id}_seq.png'))
    #                 plt.close()
    #
    #
    #             # reshape into 24x24
    #             attention_seq_global = attention[0, head_id, target_token_id, 35:611]
    #             attention_array_global = attention_seq_global.reshape(24, 24)
    #             # attention_array = np.reshape(attention_seq, (24, 24), order='F')
    #
    #
    #             attention_seq_local_collage = attention[0, head_id, target_token_id, 611:2277]
    #             attention_array_local_collage = attention_seq_local_collage.reshape(34, 49)
    #
    #             if kernel_type == 'gauss':
    #                 attention_array_local_collage_w_kernel = apply_gauss_kernel(attention_array_local_collage)
    #                 attention_array_global_w_kernel = apply_gauss_kernel(attention_array_global)
    #                 if normalized_:
    #                     attention_array_local_collage_w_kernel = attention_array_local_collage_w_kernel / np.max( attention_array_local_collage_w_kernel)
    #                     attention_array_global_w_kernel = attention_array_global_w_kernel / np.max(attention_array_global_w_kernel)
    #
    #                 # plot attention array and save to file
    #                 plt.figure(figsize=(10, 10))
    #                 plt.imshow(attention_array_local_collage_w_kernel, cmap='viridis', interpolation=None)
    #                 plt.colorbar()
    #                 # save plot to file
    #                 plt.savefig(osp.join(subfolder_att_local_collage_w_kernel,
    #                                      f'attentionhead_{head_id}_target_token{target_token_id}.png'))
    #                 plt.close()
    #
    #
    #                 plt.figure(figsize=(10, 10))
    #                 plt.imshow(attention_array_global_w_kernel, cmap='viridis', interpolation=None)
    #                 plt.colorbar()
    #                 # save plot to file
    #                 plt.savefig(osp.join(subfolder_att_global_w_kernel,
    #                                      f'attentionhead_{head_id}_target_token{target_token_id}.png'))
    #                 plt.close()
    #
    #             if normalized_:
    #                 attention_array_local_collage = attention_array_local_collage / np.max(attention_array_local_collage)
    #                 # normalize the array for visualization
    #                 attention_array_global = attention_array_global / np.max(attention_array_global)
    #
    #             if True:
    #                 # plot attention array and save to file
    #                 plt.figure(figsize=(10, 10))
    #                 plt.imshow(attention_array_local_collage, cmap='viridis', interpolation= None)
    #                 plt.colorbar()
    #                 # save plot to file
    #                 plt.savefig(osp.join(subfolder_att_local_collage, f'attentionhead_{head_id}_target_token{target_token_id}.png'))
    #                 plt.close()
    #
    #
    #             if False:
    #                 result_attention_array_global = Image.fromarray((np.uint8(cm.jet(attention_array_global) * 255))).resize((336, 336))
    #                 blended_img_attention_array_global = Image.blend(padded_img, result_attention_array_global.convert('RGB'), alpha=0.3)
    #
    #                 blended_img_attention_array_global.save(osp.join(subfolder_blended, f'attentionhead_{head_id}_target_token{target_token_id}.png'))
    #
    #             # plot attention array and save to file
    #             plt.figure(figsize=(10, 10))
    #             plt.imshow(attention_array_global, cmap='viridis', interpolation= None)
    #             plt.colorbar()
    #             # save plot to file
    #             plt.savefig(osp.join(subfolder_att_global, f'attentionhead_{head_id}_target_token{target_token_id}.png'))
    #             plt.close()
    #
    #
    #
    #
    #
    #         # plt.figure(figsize=(10, 10))
    #         # plt.imshow(attention_array, cmap='hot', interpolation='nearest')
    #         # plt.colorbar()
    #         # # save plot to file
    #         # plt.savefig(osp.join(data_dir, f'attention_{head_id}_target_token{target_token_id}.png'))
    #
    #     # print(attention[0, -1, :, :])
