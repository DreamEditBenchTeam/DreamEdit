from IPython.display import Image, display
from clip_retrieval.clip_client import ClipClient, Modality
import urllib.request
import logging
import argparse
from pathlib import Path
import os
import socket


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Script for clip retrieval.")
    parser.add_argument(
        "--text_prompt",
        type=str,
        default=None,
        required=True,
        help="The text prompt of image want to be retrieved.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to the retrieved image.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        required=False,
        help="Number of images to be retrived.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def log_result(result):
    id, caption, url, similarity = result["id"], result["caption"], result["url"], result["similarity"]
    print(f"id: {id}")
    print(f"caption: {caption}")
    print(f"url: {url}")
    print(f"similarity: {similarity}")
    display(Image(url=url, unconfined=True))


def main(args):
    # logging_dir = Path(args.output_dir, "0", args.logging_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger('clip_image_retrieval')

    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-H-14",
        aesthetic_score=9,
        aesthetic_weight=0.5,
        modality=Modality.IMAGE,
        num_images=args.num_images)

    results = client.query(text=args.text_prompt)
    logger.info("{} of {} are retrieved".format(len(results), args.text_prompt))
    for i, result in enumerate(results):
        if not result["url"].endswith(".jpg"):
            continue
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        try:
            urllib.request.urlretrieve(result["url"], args.output_path + "found" + str(i) + ".jpeg")
        except:
            logger.info("failure")
            continue


if __name__ == "__main__":
    args = parse_args()
    main(args)
