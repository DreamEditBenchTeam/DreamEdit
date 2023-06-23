from metrics.dino_vit import VITs16
from metrics.clip_vit import CLIP
from metrics.evaluate_dino import *
import csv
import os
from utils.mask_helper import *
from utils.visual_helper import *
import glob

def geo_mean(iterable):
    iterable_new = np.where(iterable<0, 0.001, iterable)
    a = np.array(iterable_new)
    return a.prod()**(1.0/len(a))


subject_folder_names = ["dog", "dog2", "dog3", "dog5", "dog6", "dog7", "dog8", "cat", "cat2",
                        "bear_plushie", "backpack", "backpack_dog", "berry_bowl", "can", "candle",
                        "clock", "colorful_sneaker", "duck_toy", "fancy_boot", "grey_sloth_plushie",
                        "monster_toy", "pink_sunglasses", "poop_emoji", "rc_car", "red_cartoon", "robot_toy",
                        "shiny_sneaker", "teapot", "vase", "wolf_plushie"]
# subject_folder_names = ["poop_emoji"]
task = "add"
method = "dreambooth"
# result_folder_name = "/home/data/dream_edit_project/results/2023-05-31-replacement"
cvpr_folder_name = "/home/data/dream_edit_project/benchmark/cvpr_dataset"
if task == "replace":
    if method == "dreamedit":
        replacement_index = [3, 6, 9, 12, 15]
        result_folder_name = "/home/data/dream_edit_project/results/2023-05-31-replacement"
        device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
    elif method == "diffedit":
        replacement_index = [4]
        result_folder_name = "/home/data/dream_edit_project/results/2023-05-31-replacement-diffedit"
        device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")
    elif method == "copygen":
        replacement_index = [4, 7, 10, 13, 16]
        result_folder_name = "/home/data/dream_edit_project/results/2023-06-13-replacement-copy-paste"
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    elif method == "copypaste":
        replacement_index = [1]
        result_folder_name = "/home/data/dream_edit_project/results/2023-06-13-replacement-copy-paste"
        device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
    elif method == "dreambooth":
        replacement_index = [0]
        result_folder_name = "/home/data/dream_edit_project/results/2023-06-13-replacement-dreambooth"
        device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

else:
    if method == "dreamedit":
        replacement_index = [5, 8, 11, 14, 17]
        result_folder_name = "/home/data/dream_edit_project/results/2023-06-12-addition-combined"
        device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    elif method == "copygen":
        replacement_index = [4, 7, 10, 13, 16]
        result_folder_name = "/home/data/dream_edit_project/results/2023-05-31-addition-copy-paste"
        device = torch.device("cuda:6") if torch.cuda.is_available() else torch.device("cpu")
    elif method == "copypaste":
        replacement_index = [1]
        result_folder_name = "/home/data/dream_edit_project/results/2023-05-31-addition-copy-paste"
        device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
    elif method == "diffedit":
        replacement_index = [6]
        result_folder_name = "/home/data/dream_edit_project/results/2023-05-31-addition-diffedit"
        device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")
    elif method == "dreambooth":
        replacement_index = [0]
        result_folder_name = "/home/data/dream_edit_project/results/2023-06-13-addition-dreambooth"
        device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

iterations = len(replacement_index)
dino_model = VITs16(device)
clip_model = CLIP(device)
number_sub_images = 10
image_size = 512


if task == "replace":
    subject_folder_to_bench_dict = {"dog2": "dog", "dog3": "dog", "dog5": "dog", "dog6": "dog",
                                    "dog7": "dog", "dog8": "dog", "cat2": "cat", "backpack_dog": "backpack"}
else:
    subject_folder_to_bench_dict = {"dog2": "dog", "dog3": "dog", "dog5": "dog", "dog6": "dog",
                                    "dog7": "dog", "dog8": "dog", "cat2": "cat", "backpack_dog": "backpack", "shiny_sneaker": "sneaker",
                                    "colorful_sneaker": "sneaker"}

dino_sub_score_list = []
clipi_sub_score_list = []
dino_back_score_list = []
clipi_back_score_list = []
overall_list = []

dino_sub_score_list_iteration = [[] for i in range(iterations)]
dino_back_score_list_iteration = [[] for i in range(iterations)]
clipi_sub_score_list_iteration = [[] for i in range(iterations)]
clipi_back_score_list_iteration = [[] for i in range(iterations)]
overall_list_iteration = [[] for i in range(iterations)]


for subject_folder in subject_folder_names:
    print(subject_folder, flush=True)
    if task == "replace":
        config_folder_name = "replace_" + subject_folder + "_config_01"
        image_path = os.path.join(result_folder_name, config_folder_name, subject_folder, "replace", "result_all.jpg")
    else:
        config_folder_name = "add_" + subject_folder + "_config_01"
        image_path = os.path.join(result_folder_name, config_folder_name, subject_folder, "add", "result_all.jpg")
    if method == "dreambooth":
        image_path = os.path.join(result_folder_name, subject_folder+".jpg")
    src_img = Image.open(image_path)
    obj_image_path_list = glob.glob("/home/data/dream_edit_project/benchmark/cvpr_dataset/"+subject_folder+"/*.jpg")
    print("obj_image_path_list: ", obj_image_path_list, flush=True)
    print(src_img.size, flush=True)
    for i in range(number_sub_images):
        print(i, flush=True)
        if subject_folder in subject_folder_to_bench_dict:
            bench_folder = subject_folder_to_bench_dict[subject_folder]
        else:
            bench_folder = subject_folder
        if task == "replace":
            background_img_path = "/home/data/dream_edit_project/benchmark/ref_images/" + bench_folder + "/found" + str(i) + ".jpg"
        else:
            background_img_path = "/home/data/dream_edit_project/benchmark/background_images_refine/" + bench_folder + "/found" + str(
                i) + ".jpg"
        iterations_image = []
        iterations_dino_sub = []
        iterations_clip_sub = []
        if method != "dreambooth":
            for index in replacement_index:
                array_src = np.asarray(src_img)
                extract_img = Image.fromarray(np.uint8(array_src[i * image_size:(i + 1) * image_size, index * image_size:(index+1) * image_size, :]))
                iterations_image.append(extract_img)
        else:
            array_src = np.asarray(src_img)
            extract_img = Image.fromarray(np.uint8(
                array_src[0:image_size, i * image_size:(i + 1) * image_size, :]))
            iterations_image.append(extract_img )
        obj_image_list = [Image.open(obj_path).resize((512, 512)) for obj_path in obj_image_path_list]
        background_img = Image.open(background_img_path).resize((512, 512))
        for obj_img in obj_image_list:
            dino_score_list_subject = evaluate_dino_score_list(obj_img, iterations_image, device, dino_model)
            clip_score_list_subject = evaluate_clipi_score_list(obj_img, iterations_image, device, clip_model)
            iterations_dino_sub.append(dino_score_list_subject)
            iterations_clip_sub.append(clip_score_list_subject)
        iterations_dino_sub_avg = np.array(iterations_dino_sub).mean(axis=0).tolist()
        iterations_clip_sub_avg = np.array(iterations_clip_sub).mean(axis=0).tolist()
        iterations_dino_back = evaluate_dino_score_list(background_img, iterations_image, device, dino_model)
        iterations_clip_back = evaluate_clipi_score_list(background_img, iterations_image, device, clip_model)
        iterations_result = [iterations_dino_sub_avg, iterations_clip_sub_avg, iterations_dino_back, iterations_dino_back]
        iterations_result = np.array(iterations_result).T
        iterations_geo_avg = []
        for it in range(iterations):
            dino_sub_score_list_iteration[it].append(iterations_dino_sub_avg[it])
            dino_back_score_list_iteration[it].append(iterations_dino_back[it])
            clipi_sub_score_list_iteration[it].append(iterations_clip_sub_avg[it])
            clipi_back_score_list_iteration[it].append(iterations_clip_back[it])
            overall_list_iteration[it].append(geo_mean(iterations_result[it]))
            iterations_geo_avg.append(geo_mean(iterations_result[it]))
        # print("iterations_result: ", iterations_result)
        # print(overall_list_iteration[0][-1])
        best_overall = max(iterations_geo_avg)
        best_iter = iterations_geo_avg.index(best_overall)
        dino_sub_score_list.append(iterations_dino_sub_avg[best_iter])
        dino_back_score_list.append(iterations_dino_back[best_iter])
        clipi_sub_score_list.append(iterations_clip_sub_avg[best_iter])
        clipi_back_score_list.append(iterations_clip_back[best_iter])
        overall_list.append(best_overall)


print("dino sub: ", sum(dino_sub_score_list)/len(dino_sub_score_list), flush=True)
print("dino back: ", sum(dino_back_score_list)/len(dino_back_score_list), flush=True)
print("clip sub: ", sum(clipi_sub_score_list)/len(clipi_sub_score_list), flush=True)
print("clip back: ", sum(clipi_back_score_list)/len(clipi_back_score_list), flush=True)
print("overall: ", sum(overall_list)/len(overall_list))

for i in range(iterations):
    print("=============================", flush=True)
    print("dino sub in iteration {}: {}".format(i+1, sum(dino_sub_score_list_iteration[i])/len(dino_sub_score_list_iteration[i])), flush=True)
    print("dino back in iteration {}: {}".format(i+1, sum(dino_back_score_list_iteration[i]) / len(
        dino_back_score_list_iteration[i])), flush=True)
    print("clip sub in iteration {}: {}".format(i+1, sum(clipi_sub_score_list_iteration[i]) / len(
        clipi_sub_score_list_iteration[i])), flush=True)
    print("clip back in iteration {}: {}".format(i+1, sum(clipi_back_score_list_iteration[i]) / len(
        clipi_back_score_list_iteration[i])), flush=True)
    print("overall in iteration {}: {}".format(i+1, sum(overall_list_iteration[i]) / len(
        overall_list_iteration[i])), flush=True)
    print("=============================", flush=True)


