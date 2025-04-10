from argparse import ArgumentParser
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch
from einops import rearrange
import os
import csv
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from model.SwinIR import SwinIR
from model.ResNet50 import ResNet50
from dataset.OODDataset import OOD_Dataset
from utils import (instantiate_from_config
                   , load_model_from_checkpoint
                #    , resize_short_edge_to
                   , pad_to_multiples_of
                   , median_filter_4d
                   , cosine_similarity)


def stage1_handler(swin_model, image, device='cpu'):
    # image to tensor
    image = torch.tensor((image / 255.).clip(0, 1), dtype=torch.float32, device=device)
    # image = rearrange(image, "n h w c -> n c h w").contiguous()

    # if min(image.shape[2:]) < 512:
    #     image = resize_short_edge_to(image, size=512)
    
    ori_h, ori_w = image.shape[2:]
    # pad: ensure that height & width are multiples of 64
    pad_image = pad_to_multiples_of(image, multiple=64)

    # run
    features = swin_model(pad_image)
    # remove padding
    # features = features[:, :, :ori_h, :ori_w]

    # tensor to image
    # sample = rearrange(features * 255., "n c h w -> n h w c")

    return features

    


def main(args):
    
    # config 
    cfg = OmegaConf.load(args.config)


    # device
    device = args.device


    # Initialize SwinIR
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = load_model_from_checkpoint(cfg.test.swin_check_dir)
    swinir.load_state_dict(sd, strict=True)
    swinir.eval().to(device)


   # Initialize ResNet
    resnet_model = instantiate_from_config(cfg.model.resnet)
    rd = load_model_from_checkpoint(cfg.test.res_check_dir)
    resnet_model.load_state_dict(rd, strict=True)
    resnet_model.eval().to(device)

    # Setup data
    dataset: OOD_Dataset = instantiate_from_config(cfg.dataset)
    test_loader = DataLoader(
        dataset=dataset, batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=True)
    

    # Setup result path
    result_path = os.path.join(cfg.test.test_result_dir, 'results.csv')

    with open(result_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['P1', 'P2', 'Final Probability', 'Label'])

    

    # Lists to store actual and predicted labels
    all_preds = []
    all_labels = []

    for data, label in tqdm(test_loader):

        # SwinIR
        data_features = stage1_handler(swinir, data)


        smoothed_data_features = median_filter_4d(data_features)
       


        p1 = cosine_similarity(data_features, smoothed_data_features)


        # ResNet
        with torch.no_grad():  # Disable gradient calculation for inference
            output = resnet_model(data)
            p2 = output.squeeze()

        

        
        # Calculate final_prob
        final_prob = p1 / p2

        # Write results to CSV file
        with open(result_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(data)):
                writer.writerow([p1[i].item(), p2[i].item(), final_prob[i].item(), label[i].item()])


        preds = (output >= 0.5).int()
        
        # Append to lists
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    
    print(f'Results written to {result_path}')

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy * 100:.2f}%')



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
    main(args)