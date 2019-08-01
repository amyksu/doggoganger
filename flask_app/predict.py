
# 
# Module dependencies.
#

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import cv2
from keras.models import load_model
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image                  
from tqdm import tqdm
base_ResNet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
ResNet50_mod = ResNet50(weights='imagenet')
MODEL_PATH = './models/dog_breed_model_1.h5'


# Load your trained model
dog_breed_model = load_model(MODEL_PATH)
graph = tf.get_default_graph()
print('Model loaded. Start serving...')


# Dog dictionary.
dogs_dict = {
  'Affenpinscher': 'affenpinscher.jpg',
  'Afghan hound': 'afghan_hound.jpg',
  'Airedale terrier': 'airedale+terrier.jpg',
  'Akita': 'akita.jpg',
  'Alaskan malamute': 'alaskan_malamute.jpg',
  'American eskimo dog': 'american_eskimo.jpg',
  'American foxhound': 'american_foxhound.jpg',
  'American staffordshire terrier': 'american_staffordshire_terrier.jpg',
  'American water spaniel': 'american_water_spaniel.jpg',
  'Anatolian shepherd dog': 'anatolian_shepherd.jpg',
  'Australian cattle dog': 'australian shepherd.jpg',
  'Australian shepherd': 'australian+terrier.jpg',
  'Australian terrier': 'australian_cattle_dog.jpg',
  'Basenji': 'basenji.jpg',
  'Basset hound': 'basset+hound.jpg',
  'Beagle': 'beagle.jpg',
  'Bearded collie': 'bearded+collie.jpg',
  'Beauceron': 'beauceron.jpg',
  'Bedlington terrier': 'bedlington+terrier.jpg',
  'Belgian malinois': 'belgian+malinois.jpg',
  'Belgian sheepdog': 'belgian+sheepdog.jpg',
  'Belgian tervuren': 'belgian+tervuren.jpg',
  'Bernese mountain dog': 'bernese+mountain+dog.jpg',
  'Bichon frise': 'bichon+frise.jpg',
  'Black and tan coonhound': 'black+and+tan+coonhound.jpg',
  'Black russian terrier': 'black+russian+terrier.jpg',
  'Bloodhound': 'bloodhound.jpg',
  'Bluetick coonhound': 'bluetick+coonhound.jpg',
  'Border collie': 'border+collie.jpg',
  'Border terrier': 'border+terrier.jpg',
  'Borzoi': 'borzoi.jpg',
  'Boston terrier': 'boston+terrier.jpg',
  'Bouvier des flandres': 'bouvier+des+flandres.jpg',
  'Boxer': 'boxer.jpg',
  'Boykin spaniel': 'boykin+spaniel.jpg',
  'Briard': 'briard.jpg',
  'Brittany': 'brittany.jpg',
  'Brussels griffon': 'brussels+griffon.jpg',
  'Bull terrier': 'bull+terrier.jpg',
  'Bulldog': 'bulldog.jpg',
  'Bullmastiff': 'bullmastiff.jpg',
  'Cairn terrier': 'cairn+terrier.jpg',
  'Canaan dog': 'canaan+dog.jpg',
  'Cane corso': 'cane+corso.jpg',
  'Cardigan welsh corgi': 'cardigan+welsh+corgi.jpg',
  'Cavalier king charles spaniel': 'cavalier+king+charles+spaniel.jpg',
  'Chesapeake bay retriever': 'chesapeake+bay+retriever.jpg',
  'Chihuahua': 'chihuahua.jpg',
  'Chinese crested': 'chinese+crested.jpg',
  'Chinese shar-pei': 'chinese+shar-pei.jpg',
  'Chow chow': 'chow+chow.jpg',
  'Clumber spaniel': 'clumber+spaniel.jpg',
  'Cocker spaniel': 'cocker+spaniel.jpg',
  'Collie': 'collie.jpg',
  'Curly-coated retriever': 'curly+coated+retriever.jpg',
  'Dachshund': 'dachshund.jpg',
  'Dalmatian': 'dalmatian.jpg',
  'Dandie dinmont terrier': 'dandie+dinmont+terrier.jpg',
  'Doberman pinscher': 'doberman+pinscher.jpg',
  'Dogue de bordeaux': 'dogue+de+bordeaux.jpg',
  'English cocker spaniel': 'english+cocker+spaniel.jpg',
  'English setter': 'english+setter.jpg',
  'English springer spaniel': 'english+springer+spaniel.jpg',
  'English toy spaniel': 'english+toy+spaniel.jpg',
  'Entlebucher mountain dog': 'entlebucher+mountain+dog.jpg',
  'Field spaniel': 'field+spaniel.jpg',
  'Finnish spitz': 'finnish+spitz.jpg',
  'Flat-coated retriever': 'flat-coated+retriever.jpg',
  'French bulldog': 'french+bulldog.jpg',
  'German pinscher': 'german+pinscher.jpg',
  'German shepherd dog': 'german+shepherd+dog.jpg',
  'German shorthaired pointer': 'german+shorthaired+pointer.jpg',
  'German wirehaired pointer': 'german+wirehaired+pointer.jpg',
  'Giant schnauzer': 'giant+schnauzer.jpg',
  'Glen of imaal terrier': 'glen+of+imaal+terrier.jpg',
  'Golden retriever': 'golden+retriever.jpg',
  'Gordon setter': 'gordon+setter.jpg',
  'Great dane': 'great+dane.png',
  'Great pyrenees': 'great+pyrenees.jpg',
  'Greater swiss mountain dog': 'greater+swiss+mountain+dog.jpg',
  'Greyhound': 'greyhound.jpg',
  'Havanese': 'havanese.jpg',
  'Ibizan hound': 'ibizan+hound.jpg',
  'Icelandic sheepdog': 'icelandic+sheepdog.jpg',
  'Irish red and white setter': 'irish+red+and+white+setter.jpg',
  'Irish setter': 'irish+setter.jpg',
  'Irish terrier': 'irish+terrier.jpg',
  'Irish water spaniel': 'irish+water+spaniel.jpg',
  'Irish wolfhound': 'irish+wolfhound.jpg',
  'Italian greyhound': 'italian+greyhound.jpg',
  'Japanese chin': 'japanese_chin.png',
  'Keeshond': 'keeshond.jpg',
  'Kerry blue terrier': 'kerry+blue+terrier.jpg',
  'Komondor': 'komondor.jpg',
  'Kuvasz': 'Kuvasz.jpg',
  'Labrador retriever': 'Labrador_retriever.jpg',
  'Lakeland terrier': 'lakeland_terrier.jpg',
  'Leonberger': 'leonberger.jpg',
  'Lhasa apso': 'lhasa+apso.jpg',
  'Lowchen': 'lowchen.jpg',
  'Maltese': 'maltese.png',
  'Manchester terrier': 'manchester+terrier.jpg',
  'Mastiff': 'mastiff.jpg',
  'Miniature schnauzer': 'Miniature+schnauzer.jpg',
  'Neapolitan mastiff': 'neapolitan+mastiff.jpg',
  'Newfoundland': 'newfoundland.png',
  'Norfolk terrier': 'norfolk+terrier.jpg',
  'Norwegian buhund': 'norwegian+buhund.jpg',
  'Norwegian elkhound': 'norwegian+elkhound.jpg',
  'Norwegian lundehund': 'norwegian+lundehund.jpg',
  'Norwich terrier': 'norwich+terrier.jpg',
  'Nova scotia duck tolling retriever': 'nova+scotia+duck+tolling+retriever.jpg',
  'Old english sheepdog': 'old+english+sheepdog.jpg',
  'Otterhound': 'otterhound.jpg',
  'Papillon': 'papillon.jpg',
  'Parson russell terrier': 'parson+russell+terrier.jpg',
  'Pekingese': 'pekingese.jpg',
  'Pembroke welsh corgi': 'pembroke+welsh+corgi.jpg',
  'Petit basset griffon vendeen': 'petit+basset+griffon+vendeen.jpg',
  'Pharaoh hound': 'pharaoh+hound.jpg',
  'Plott': 'plott.jpg',
  'Pointer': 'pointer.jpg',
  'Pomeranian': 'pomeranian.jpg',
  'Poodle': 'poodle.jpg',
  'Portuguese water dog': 'portuguese+water+dog.jpg',
  'Saint bernard': 'saint+bernard.jpg',
  'Silky terrier': 'silky+terrier.jpg',
  'Smooth fox terrier': 'smooth+fox+terrier.jpg',
  'Tibetan mastiff': 'tibetan+mastiff.jpg',
  'Welsh springer spaniel': 'welsh+springer+spaniel.jpg',
  'Wirehaired pointing griffon': 'wirehaired+pointing+griffon.jpg',
  'Xoloitzcuintli': 'xoloitzcuintli.jpg',
  'Yorkshire terrier': 'yorkshire+terrier.jpg'
}

# Link to petfinder.
dogs_link_dict = {
  'Affenpinscher': 'affenpinscher/',
  'Afghan hound': 'afghan-hound/',
  'Airedale terrier': 'airedale-terrier/',
  'Akita': 'akita/',
  'Alaskan malamute': 'alaskan-malamute/',
  'American eskimo dog': 'american-eskimo-dog-standard/',
  'American foxhound': 'american-foxhound/',
  'American staffordshire terrier': 'american-staffordshire-terrier/',
  'American water spaniel': 'american-water-spaniel/',
  'Anatolian shepherd dog': 'anatolian-shepherd/',
  'Australian cattle dog': 'australian-shepherd/',
  'Australian shepherd': 'australian-terrier/',
  'Australian terrier': 'australian-cattle-dog/',
  'Basenji': 'basenji/',
  'Basset hound': 'basset-hound/',
  'Beagle': 'beagle/',
  'Bearded collie': 'bearded-collie/',
  'Beauceron': 'beauceron/',
  'Bedlington terrier': 'bedlington terrier/',
  'Belgian malinois': 'belgian-malinois/',
  'Belgian sheepdog': 'belgian-sheepdog/',
  'Belgian tervuren': 'belgian-tervuren/',
  'Bernese mountain dog': 'bernese-mountain-dog/',
  'Bichon frise': 'bichon-frise/',
  'Black and tan coonhound': 'black-and-tan-coonhound/',
  'Black russian terrier': 'black-russian-terrier/',
  'Bloodhound': 'bloodhound/',
  'Bluetick coonhound': 'bluetick-coonhound/',
  'Border collie': 'border-collie/',
  'Border terrier': 'border-terrier/',
  'Borzoi': 'borzoi/',
  'Boston terrier': 'boston-terrier/',
  'Bouvier des flandres': 'bouvier-des-flandres/',
  'Boxer': 'boxer/',
  'Boykin spaniel': 'boykin-spaniel/',
  'Briard': 'briard/',
  'Brittany': 'brittany/',
  'Brussels griffon': 'brussels-griffon/',
  'Bull terrier': 'bull-terrier/',
  'Bulldog': 'bulldog/',
  'Bullmastiff': 'bullmastiff/',
  'Cairn terrier': 'cairn-terrier/',
  'Canaan dog': 'canaan-dog/',
  'Cane corso': 'cane-corso/',
  'Cardigan welsh corgi': 'cardigan-welsh-corgi/',
  'Cavalier king charles spaniel': 'cavalier-king-charles-spaniel/',
  'Chesapeake bay retriever': 'chesapeake-bay-retriever/',
  'Chihuahua': 'chihuahua/',
  'Chinese crested': 'chinese-crested-dog/',
  'Chinese shar-pei': 'chinese-shar-pei/',
  'Chow chow': 'chow-chow/',
  'Clumber spaniel': 'clumber-spaniel/',
  'Cocker spaniel': 'american-cocker-spaniel/',
  'Collie': 'collie/',
  'Curly-coated retriever': 'curly-coated-retriever/',
  'Dachshund': 'dachshund/',
  'Dalmatian': 'dalmatian/',
  'Dandie dinmont terrier': 'dandie-dinmont-terrier/',
  'Doberman pinscher': 'doberman-pinscher/',
  'Dogue de bordeaux': 'dogue-de-bordeaux/',
  'English cocker spaniel': 'english-cocker-spaniel/',
  'English setter': 'english-setter/',
  'English springer spaniel': 'english-springer-spaniel/',
  'English toy spaniel': 'english-toy-spaniel/',
  'Entlebucher mountain dog': 'entlebucher-mountain-dog/',
  'Field spaniel': 'field-spaniel/',
  'Finnish spitz': 'finnish-spitz/',
  'Flat-coated retriever': 'flat-coated-retriever/',
  'French bulldog': 'french-bulldog/',
  'German pinscher': 'german-pinscher/',
  'German shepherd dog': 'german-shepherd/',
  'German shorthaired pointer': 'german-shorthaired-pointer/',
  'German wirehaired pointer': 'german-wirehaired-pointer/',
  'Giant schnauzer': 'giant-schnauzer/',
  'Glen of imaal terrier': 'glen-of-imaal-terrier/',
  'Golden retriever': 'golden-retriever/',
  'Gordon setter': 'gordon-setter/',
  'Great dane': 'great-dane/',
  'Great pyrenees': 'great-pyrenees/',
  'Greater swiss mountain dog': 'greater-swiss-mountain-dog/',
  'Greyhound': 'greyhound/',
  'Havanese': 'havanese/',
  'Ibizan hound': 'ibizan-hound/',
  'Icelandic sheepdog': 'icelandic-sheepdog/',
  'Irish red and white setter': 'irish-setter/',
  'Irish setter': 'irish-setter/',
  'Irish terrier': 'irish-terrier/',
  'Irish water spaniel': 'irish-water-spaniel/',
  'Irish wolfhound': 'irish-wolfhound/',
  'Italian greyhound': 'italian-greyhound/',
  'Japanese chin': 'japanese-chin/',
  'Keeshond': 'keeshond/',
  'Kerry blue terrier': 'kerry-blue-terrier/',
  'Komondor': 'komondor/',
  'Kuvasz': 'Kuvasz/',
  'Labrador retriever': 'Labrador-retriever/',
  'Lakeland terrier': 'lakeland-terrier/',
  'Leonberger': 'leonberger/',
  'Lhasa apso': 'lhasa-apso/',
  'Lowchen': 'lowchen/',
  'Maltese': 'maltese/',
  'Manchester terrier': 'manchester-terrier/',
  'Mastiff': 'mastiff/',
  'Miniature schnauzer': 'miniature-schnauzer/',
  'Neapolitan mastiff': 'neapolitan-mastiff/',
  'Newfoundland': 'newfoundland/',
  'Norfolk terrier': 'norfolk-terrier/',
  'Norwegian buhund': 'norwegian-buhund/',
  'Norwegian elkhound': 'norwegian-elkhound/',
  'Norwegian lundehund': 'norwegian-lundehund/',
  'Norwich terrier': 'norwich-terrier/',
  'Nova scotia duck tolling retriever': 'nova-scotia-duck-tolling-retriever/',
  'Old english sheepdog': 'old-english-sheepdog/',
  'Otterhound': 'otterhound/',
  'Papillon': 'papillon/',
  'Parson russell terrier': 'parson-russell-terrier/',
  'Pekingese': 'pekingese/',
  'Pembroke welsh corgi': 'pembroke welsh corgi/',
  'Petit basset griffon vendeen': 'petit-basset-griffon-vendeen/',
  'Pharaoh hound': 'pharaoh-hound/',
  'Plott': 'plott-hound/',
  'Pointer': 'english-pointer/',
  'Pomeranian': 'pomeranian/',
  'Poodle': 'poodle/',
  'Portuguese water dog': 'portuguese-water-dog/',
  'Saint bernard': 'saint-bernard/',
  'Silky terrier': 'silky-terrier/',
  'Smooth fox terrier': 'smooth-fox-terrier/',
  'Tibetan mastiff': 'tibetan-mastiff/',
  'Welsh springer spaniel': 'welsh-springer-spaniel/',
  'Wirehaired pointing griffon': 'wirehaired-pointing-griffon/',
  'Xoloitzcuintli': 'xoloitzcuintli/',
  'Yorkshire terrier': 'yorkshire-terrier/'
}


def face_detector(img_path):
  img = cv2.imread(img_path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray)
  return len(faces) > 0


# Pre-process data
def convert_image_to_4D_tensor(img_path):
  # loads RGB image as PIL.Image.Image type
  img = image.load_img(img_path, target_size=(224, 224))
  # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
  x = image.img_to_array(img)
  # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
  return np.expand_dims(x, axis=0)


# Function to extract the bottleneck features for ResNet50 
def extract_Resnet50(tensor):
  """
  INPUT: image tensor 
  OUTPUT: ResNet50 bottleneck features 
  """
  return base_ResNet.predict(preprocess_input(tensor))


def predict_breed(img_path):
  """
  INPUT: path to an image
  OUTPUT: returns a prediction of dog breed
  """
  # extract bottleneck features
  global graph
  with graph.as_default():
    bottleneck_feature = extract_Resnet50(convert_image_to_4D_tensor(img_path))

    # obtain predicted vector
    predicted_vector = dog_breed_model.predict(bottleneck_feature.reshape(1,1,1,2048))

    # get class with highest probability and match to label for class
    predicted_index = np.argmax(predicted_vector)

    breed = list(dogs_dict.keys())[predicted_index]
    file_name = dogs_dict[breed]

    return {
      "name": breed,
      "file_name": file_name,
      "uri": dogs_link_dict[breed]
    }
	




