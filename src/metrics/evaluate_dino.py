from .distances import compute_cosine_distance
import numpy as np
import torch

def evaluate_dino_score(real_image, generated_image, device, fidelity):
    #tensor_image_1 = torch.from_numpy(np.asarray(real_image)).permute(2, 0, 1).unsqueeze(0)
    #tensor_image_2 = torch.from_numpy(np.asarray(generated_image)).permute(2, 0, 1).unsqueeze(0)
    preprocess = fidelity.get_transform()
    tensor_image_1 = preprocess(real_image).unsqueeze(0)
    tensor_image_2 = preprocess(generated_image).unsqueeze(0)
    emb_1 = fidelity.get_embeddings(tensor_image_1.float().to(device))
    emb_2 = fidelity.get_embeddings(tensor_image_2.float().to(device))
    assert emb_1.shape == emb_2.shape
    score = compute_cosine_distance(emb_1.detach().cpu().numpy(), emb_2.detach().permute(1, 0).cpu().numpy())
    return score[0][0]

def evaluate_dino_score_list(real_image, generated_image_list, device, fidelity):
    score_list = []
    total = len(generated_image_list)
    for i in range(total):
        score = evaluate_dino_score(real_image, generated_image_list[i], device, fidelity)
        score_list.append(score)
    # max_score = max(score_list)
    # max_index = score_list.index(max_score)
    # print("The best result is at {} th iteration with dino score {}".format(max_index + 1, max_score))
    # return max_score, max_index
    return score_list

def evaluate_clipi_score(real_image, generated_image, device, clip_model):
    preprocess = clip_model.get_transform()
    tensor_image_1 = preprocess(real_image).unsqueeze(0)
    tensor_image_2 = preprocess(generated_image).unsqueeze(0)
    emb_1 = clip_model.encode_image(tensor_image_1.float().to(device))
    emb_2 = clip_model.encode_image(tensor_image_2.float().to(device))
    assert emb_1.shape == emb_2.shape
    score = compute_cosine_distance(emb_1.detach().cpu().numpy(), emb_2.detach().permute(1, 0).cpu().numpy())
    return score[0][0]

def evaluate_clipi_score_list(real_image, generated_image_list, device, clip_model):
    score_list = []
    total = len(generated_image_list)
    for i in range(total):
        score = evaluate_clipi_score(real_image, generated_image_list[i], device, clip_model)
        score_list.append(score)
    # max_score = max(score_list)
    # max_index = score_list.index(max_score)
    # print("The best result is at {} th iteration with dino score {}".format(max_index + 1, max_score))
    # return max_score, max_index
    return score_list