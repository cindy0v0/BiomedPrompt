import torch
import h5py
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from collections import Counter

def lookup_final_label(slide_name, df):
    matching_row = df[df['slide_name'] == slide_name]
    if not matching_row.empty:
        final_label = matching_row.iloc[0]['Final-Labels']
        return final_label
    else:
        return "Not Found"


def load_imgs(imgs_path):
    '''
    returns patch level features, slide names and slide labels
    '''
    fpaths = sorted(glob.glob(os.path.join(imgs_path, "*.h5")))
    metadata = pd.read_csv("/projects/ovcare/classification/cshi/OCEAN/data/final_meta.csv")
    print(f"Loading {len(fpaths)} images from", imgs_path)

    input_arrs, input_snames, input_labels = [], [], []
    slide_labels = []
    for fpath in fpaths:
        f = h5py.File(fpath, 'r')
        input_arr =  np.array(f['features']['20x'], dtype=np.float32)
        input_arr = torch.tensor(input_arr)
        input_arrs.append(input_arr)
        bag_size = input_arr.shape[0]

        slide_name = os.path.basename(fpath).split('.')[0]
        input_snames.extend([slide_name] * bag_size)
        slabel = lookup_final_label(slide_name, metadata)
        slide_labels.append(slabel)
        input_labels.extend([slabel] * bag_size)
    # input_snames = [os.path.basename(fpath).split('.')[0] for fpath in fpaths]
    print(len(input_snames), len(input_labels))
    print(f"class distribution: {Counter(input_labels)}\n")
    return input_arrs, input_snames, input_labels

def compute_wasserstein_distance(img_embeddings, text_embeddings):
    """
    Compute Wasserstein distance between image and text embeddings.
    
    :param img_embeddings: torch.Tensor of shape (N, 768)
    :param text_embeddings: torch.Tensor of shape (M, 512)
    :return: numpy array of shape (N, M) containing pairwise distances
    """
    if type(img_embeddings) == torch.Tensor:
        img_embeddings = img_embeddings.cpu().numpy()
    if type(text_embeddings) == torch.Tensor:
        text_embeddings_np = text_embeddings.cpu().numpy()
    
    N, M = img_embeddings_np.shape[0], text_embeddings_np.shape[0]
    distances = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            # Normalize the embeddings to sum to 1 (to treat them as distributions)
            img_dist = img_embeddings_np[i] / np.sum(img_embeddings_np[i])
            text_dist = text_embeddings_np[j] / np.sum(text_embeddings_np[j])
            
            distances[i, j] = wasserstein_distance(img_dist, text_dist)
    
    return distances