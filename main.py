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
    # source_dataset_path = "/projects/ovcare/classification/Ali/Multi_mag_backbone/dataset/ocean_comp/features/test_sets/public/OV/finetuned/fold-1/20x/OV/PLIP/"
    source_dataset_path= "/projects/ovcare/classification/cshi/OCEAN/train_data_rep2/features/public/PLIP"

    hashtag = "OvarianPath"
    prompts = ["a patch image of ovarian carcinoma", 
                "a H&E image of ovarian carcinoma with an unusual region of interest (ROI) and/or with inconclusive details",
                "an endometrial H&E image with usual region of interest (ROI) and/or with conclusive details"
                ]
    # prompts = ["a H&E image of clear cell ovarian cancer carcinoma", 
    #             "a H&E image of high grade serous ovarian carcinoma",
    #             "a H&E image of low grade serous ovarian carcinoma",
    #             "a H&E image of mucinous ovarian carcinoma",
    #             "a H&E image of endometrioid ovarian carcinoma",
    #             "a H&E image of ovarian carcinoma with an unusual region of interest (ROI) and/or with inconclusive details",
    #             ]
    classes = ["CC", "EC", "HGSC", "LGSC", "MC", "OOD", "Not OVC"] # an unknown rare outlier subtype of
    classes = ['OVC Major Subtype', 'OOD OVC', 'Not OVC']
    model = text_encoder()

    top_k = 48279 * 3
    y_true, y_pred, y_score = text_to_image_retrieval(source_dataset_path, hashtag, prompts, model, top_k)

    # metrics = evaluate_predictions(y_true, y_pred, y_score)

    # print(f"Precision: {metrics['precision']:.3f}")
    # print(f"Recall: {metrics['recall']:.3f}")
    # print(f"F1-Score: {metrics['f1_score']:.3f}")
    # print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    # print(f"Cohen's Kappa: {metrics['cohen_kappa']:.3f}")
    # print(f"Confusion Matrix: \n{metrics['confusion_matrix']}")

    # plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    # print(output['retrieved_image_labels'])
    # save_path="/projects/ovcare/classification/cshi/OCEAN_COMPETITION/BiomedPrompt/output/"
    # with open(save_path, 'wb') as f:
    #     pickle.dump(output, f)


# ref: https://github.com/DeepMicroscopy/Prototype-sampling/tree/main
def text_to_image_retrieval(source_dataset_path, hashtag, prompts, model, top_k):
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
    
    img_embeddings, img_names, img_labels= load_imgs(source_dataset_path)
    
    query_embeddings = model(prompts)  # (3, 512)
    query_embeddings /= torch.linalg.vector_norm(query_embeddings, dim=1, keepdim=True)

    cos_sim = []
    for img_embedding in img_embeddings:
        sim = (100. * img_embedding @ query_embeddings.t()).softmax(dim=1)
        cos_sim.append(sim)
    cos_sim = torch.cat(cos_sim)
    values, indices = cos_sim.topk(1)
    print(values.shape, indices.shape, '\n')

    values, indices = values.detach().cpu().numpy().flatten(), indices.detach().cpu().numpy().flatten()
    # select top k% images with the highest feature similarity to the class_index prompt
    for class_index in range(len(prompts)):
        most_similar = np.argsort(values[indices == class_index])[-top_k:]
        filtered_img_names = [name for name, index in zip(img_names, indices) if index == class_index]
        filtered_img_labels = [label for label, index in zip(img_labels, indices) if index == class_index]
        retrieved_image_files = [filtered_img_names[i] for i in most_similar]
        retrieved_image_labels = [filtered_img_labels[i] for i in most_similar]

        count = Counter(retrieved_image_labels)
        percentage = {key: (value / len(retrieved_image_labels)) * 100 for key, value in count.items()}
        print(f"{prompts[class_index]}: {len(retrieved_image_labels)}")
        print(percentage)
        print(f"slide counts: {len(set(retrieved_image_files))}")

    return None, None, None


if __name__ == '__main__':
    main()
    '''
    multiprocessing.freeze_support()
    main()  # pylint: disable=no-value-for-parameter
    '''

# # If you want to plot ROC curve
# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import roc_curve, auc

# y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
# n_classes = y_true_bin.shape[1]

# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# plt.figure(figsize=(10, 8))
# for i, color in zip(range(n_classes), ['blue', 'red', 'green']):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()