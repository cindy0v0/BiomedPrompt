# ref: https://github.com/openai/CLIP?tab=readme-ov-file#zero-shot-prediction

import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import click
from collections import Counter
from encoders.plip import image_encoder, text_encoder
from utils import load_imgs, compute_wasserstein_distance

@click.command()
@click.option('--source_dataset_path', default='', type=str, help='img path')
@click.option('--hashtag', default='', type=str, help='')
@click.option('--model', default='', type=str, help='')
@click.option('--save_path', default='./', type=str, help='')
def main(source_dataset_path, hashtag, model, save_path):
    source_dataset_path = "/projects/ovcare/classification/Ali/Multi_mag_backbone/dataset/ocean_comp/features/test_sets/public/OV/finetuned/fold-1/20x/OV/PLIP/"
    hashtag = "OvarianPath"
    classes = ["clear cell (CC)", "endometrioid (EC)", "high-grade serous (HGSC)", "low-grade serous (LGSC)", "mucinous (MC)", "Unusual ROI with in conclusive details"] # an unknown rare outlier subtype of
    model = text_encoder()
    output = text_to_image_retrieval(source_dataset_path, hashtag, classes, model)
    print(output['retrieved_image_labels'])
    save_path="/projects/ovcare/classification/cshi/OCEAN_COMPETITION/BiomedPrompt/output/"
    with open(save_path, 'wb') as f:
        pickle.dump(output, f)




# ref: https://github.com/DeepMicroscopy/Prototype-sampling/tree/main
def text_to_image_retrieval(source_dataset_path, hashtag, classes, model):
    """
    OpenPath data structure
    
    set(df["source"]): {'Twitter reply', 'Twitter', 'PathLAION'}
    set(df.loc[data["source"]=="Twitter", "hashtag"]): 
        {'OralPath', 'ENTPath', 'SurgPath', 'nephpath',  'IDpath', 'PathGME', 'MolDx', 'PulmPath', 
        'BSTpath', 'dermpath', 'EndoPath', 'NeuroPath', 'blooducation', 'Autopsy', 'EyePath', 
        'patientbloodmanagement', 'CardiacPath', 'GUPath', 'pathInformatics', 'HPBpath', 'Gynpath', 
        'ForensicPath', 'ClinPath', 'HemePath', 'RenalPath', 'BloodBank', 'GIPath', 'PediPath', 
        'BreastPath', 'pancpath', 'Cytopath', 'FNApath'}
    set(df.loc[data["source"]=="Twitter reply", "hashtag"]): 
        {'OralPath', 'ENTPath', 'SurgPath', 'nephpath', 'IDpath', 'MolDx', 'PulmPath', 
        'BSTpath', 'dermpath', 'EndoPath', 'NeuroPath', 'blooducation', 'Autopsy', 'EyePath', 
        'CardiacPath', 'GUPath', 'HPBpath', 'Gynpath', 
        'ForensicPath', 'ClinPath', 'HemePath', 'RenalPath', 'BloodBank', 'GIPath', 'PediPath', 
        'BreastPath', 'pancpath', 'Cytopath', 'FNApath'}
    set(df.loc[data["source"]=="PathLAION", "hashtag"]):
        {'----'}
    """
    # image_embeddings = torch.tensor(np.load(os.path.join(source_dataset_path,
    #                                                      "OpenPath_image_embeddings_normalized.npy")))  # (208414, 512)
    # df = pd.read_csv(os.path.join(source_dataset_path, "dataframe_208K_rows.csv"),
    #                  header=None,
    #                  names=["id", "source", "hashtag", "web", "unknown1", "unknown2"],
    #                  usecols=["id", "source", "hashtag", "web"])

    # breast_df = df.loc[df["hashtag"] == hashtag]  # DateFrame (7513, 4)
    # breast_image_embeddings = image_embeddings[breast_df["id"].values].float()  # (7513, 512)
    
    # features_size = {'Phikon': 768, 'CTransPath': 768, 'KimiaNet': 1024, 'PLIP': 512}
    
    img_embeddings, img_names, img_labels= load_imgs(source_dataset_path)
    
    print("Retreiving images for ", hashtag)
    # prompts = [f"an H&E image of {c} ovarian carcinoma" for c in classes]
    prompts = ["a patch image of ovarian carcinoma", 
                "a H&E image of ovarian carcinoma with an unusual region of interest (ROI) and/or with inconclusive details",
                "a non-endometrial H&E image with an unusual region of interest (ROI) and/or with inconclusive details and is not ovarian carcinoma"]
    outlier_class = 0
    query_embeddings = model(prompts)  # (2, 512)
    query_embeddings /= torch.linalg.vector_norm(query_embeddings, dim=1, keepdim=True)

    
    cos_sim = []
    for img_embedding in img_embeddings:
        sim = (100. * img_embedding @ query_embeddings.t()).softmax(dim=1)
        cos_sim.append(sim)
    cos_sim = torch.cat(cos_sim)
    # cos_sim = (100. * img_embeddings @ query_embeddings.t()).softmax(dim=1)  # (7513, 2)
    # wasserstein_distance = compute_wasserstein_distance(img_embeddings, query_embeddings) # (7513, 2)
    # assign a label for the image (e.g., normal/tumor), return also the similarity to the class
    values, indices = cos_sim.topk(1)  # (7513, 1), (7513, 1)
    print(values.shape, indices.shape)
    values, indices = values.detach().cpu().numpy().flatten(), indices.detach().cpu().numpy().flatten()
    # select the 100 images with the highest feature similarity to the outlier tissue prompt
    # k=1
    # indices = np.argpartition(wasserstein_distance, -k, axis=-1)[:, -k:]
    # values = np.take_along_axis(wasserstein_distance, indices, axis=-1)
    most_similar = np.argsort(values[indices == outlier_class])[-48279:]

    # import pdb; pdb.set_trace()

    # retrieved_embedding = img_embeddings[indices == 1][most_similar]  # (100, 512)
    # retrieved_image_files = breast_df.loc[indices == 1, "web"].values[most_similar]
    filtered_img_names = [name for name, index in zip(img_names, indices) if index == outlier_class]
    filtered_img_labels = [label for label, index in zip(img_labels, indices) if index == outlier_class]
    retrieved_image_files = [filtered_img_names[i] for i in most_similar]
    retrieved_image_labels = [filtered_img_labels[i] for i in most_similar]

    count = Counter(retrieved_image_labels)
    percentage = {key: (value / 48279) * 100 for key, value in count.items()}
    print(percentage)

    import pdb; pdb.set_trace()

    return {'retrieved_embedding': retrieved_embedding, 'retrieved_image_files': retrieved_image_files, "retrieved_image_labels": retrieved_image_labels}


if __name__ == '__main__':
    main()
    '''
    multiprocessing.freeze_support()
    main()  # pylint: disable=no-value-for-parameter
    '''