from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image
import numpy as np
import csv
import torch


processor = AutoProcessor.from_pretrained("microsoft/git-large-r-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-r-coco")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict(image_urls):
  images = []
  for url in image_urls:
    url = url + '?w=300&fm=webp&q=80'
    print(url)
    image = Image.open(requests.get(url, stream=True).raw)
    images.append(image)


  pixel_values = processor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)
  generated_ids = model.generate(pixel_values=pixel_values, max_length=50)

  preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
  preds = [pred for pred in preds]

  return preds

def read_csv():
  urls = []
  with open('missing-alt.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    next(reader) # skip header
    for row in reader:
      urls.append(row)

  return urls

def write_csv(altList):
  with open('missing-alt-predicted.csv', 'a') as csvFile:
    for alt in altList:
      csvFile.write(alt[0] + ',' + alt[1] + ',' + alt[2] + '\n')
      csvFile.flush()

imageUrls = np.array(read_csv())

for segment in np.array_split(imageUrls, 10):
  description = predict(segment[:,1])
  write_csv(list(zip(segment[:,0], segment[:,1], description)))
