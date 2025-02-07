import argparse 
import json
import utils

parser = argparse.ArgumentParser(description='use a neural network to classify an image!')
parser.add_argument('image_input', help='image file to classifiy (required)')
parser.add_argument('checkpoint', help='model used for classification (required)')
parser.add_argument('--top_k', help='how many prediction categories to show [default 5].')
parser.add_argument('--category_names', help='file for category names')
parser.add_argument('--gpu', action='store_true', help='gpu option')
args = parser.parse_args()

top_k = int(args.top_k) if args.top_k is not None else 1  
category_names = args.category_names if args.category_names is not None else "cat_to_name.json"  
gpu = True if args.gpu else False  

model = utils.load_model(args.checkpoint)
device = "cuda" if gpu else "cpu"
model.to(device)

probs, predict_classes = utils.predict(args.image_input, model, top_k)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
classes = []  
for predict_class in predict_classes:
    classes.append(cat_to_name[str(predict_class)])
print(probs)
print(classes)
