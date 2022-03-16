import argparse
from catr.configuration import Config
from catr.datasets import coco
from catr.models import utils, caption
import os
import torch
from PIL import Image
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import math


parser = argparse.ArgumentParser(
    description="catr visualization")
parser.add_argument("--p2_path", default="./hw3_data/p2_data/images", type=str)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--epoch", default=50, type=int)
parser.add_argument("--dataset", default="p2")
parser.add_argument("--model_path", default="./model_pth", type=str)
parser.add_argument("--predict_path", default="./predict", type=str)
args = parser.parse_args()

config = Config()
model, criterion = caption.build_model(config)
checkpoint = torch.hub.load_state_dict_from_url(
    url='https://github.com/saahiluppal/catr/releases/download/0.2/weight493084032.pth',
    map_location='cpu'
)
model.load_state_dict(checkpoint['model'])
# model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

base_dir = args.p2_path


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


@torch.no_grad()
def evaluate():
    model.eval()
    attention_map_list = []
    for i in range(config.max_position_embeddings - 1):
        predictions, viz = model(image, caption, cap_mask)

        # print(predictions.shape)
        predictions = predictions[:, i, :]
        attention_map_list.append(viz[:, i, :])
        # print(predictions.shape)
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption, viz

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption, viz

for img in os.listdir(base_dir):
    image_path = os.path.join(base_dir, img)
    image_show = Image.open(image_path)
    image = coco.val_transform(image_show)
    image = image.unsqueeze(0)


    caption, cap_mask = create_caption_and_mask(
        start_token, config.max_position_embeddings)
    output, viz = evaluate()
    viz = viz.reshape((viz.shape[1], viz.shape[2]))
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True).capitalize()
    result_split = result.split()
    
    fig, axs = plt.subplots(math.ceil((len(result_split)+2)/5), 5)
    [axi.set_axis_off() for axi in axs.ravel()]
    axs[0][0].imshow(image_show)
    axs[0][0].set_title("<start>")
    last_index = 0
    for i, out in enumerate(result_split):
        index = i+1
        axs[index//5][index%5].imshow(image_show)
        axs[index//5][index%5].set_title(out)
        attention_map = viz[index]
        # print(attention_map.shape)
        attention_map = attention_map.reshape(1, 1, attention_map.shape[0]//19, 19)
        attention_map = torch.nn.functional.upsample(attention_map.data, size=(image_show.size[1], image_show.size[0]), mode="bilinear")
        attention_map = attention_map.reshape(image_show.size[1], image_show.size[0])
        axs[index//5][index%5].imshow(attention_map.cpu().numpy(), alpha=0.4, cmap='rainbow')
        last_index = index
    index = last_index+1
    axs[index//5][index%5].imshow(image_show)
    axs[index//5][index%5].set_title("<end>")
    attention_map = viz[index]
    # print(attention_map.shape)
    attention_map = attention_map.reshape(1, 1, attention_map.shape[0]//19, 19)
    attention_map = torch.nn.functional.upsample(attention_map.data, size=(image_show.size[1], image_show.size[0]), mode="bilinear")
    attention_map = attention_map.reshape(image_show.size[1], image_show.size[0])
    axs[index//5][index%5].imshow(attention_map.cpu().numpy(), alpha=0.4, cmap='rainbow')
    
    # for i, out in enumerate(out_actual):
    #     if(out == 101):
    #         print("<start>")
    #     elif(out == 1012):
    #         print("<end>")
    #     else:
    #         result = tokenizer.decode(out, skip_special_tokens=True)
    #         print(result)
    # print()
    # print(result.capitalize())

    fig.savefig(os.path.join(args.predict_path, img[:-4]+".png"))
