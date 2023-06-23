from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

class BLIP_Model():
    
    def __init__(self, device="cuda", processor_weights = "Salesforce/blip-image-captioning-base", conditional_generation_weights = "Salesforce/blip-image-captioning-base"):
        
        self.processor = BlipProcessor.from_pretrained(processor_weights)
        self.model = BlipForConditionalGeneration.from_pretrained(conditional_generation_weights)
        self.device = device
        self.model.to(self.device)

    def predict_one_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

if __name__ == "__main__":
    
    import requests
    
    BLIP_Model = BLIP_Model()
    
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    generated_text = BLIP_Model.predict_one_image(image)
    #print(generated_text)
    assert generated_text == "two cats sleeping on a couch"
