import os, numpy, PIL
from shutil import copyfile
from PIL import Image, ImageChops

labels = ["siren", "car_horn", "gun_shot", "street_music", "drilling", "dog_bark", "jackhammer", "air_conditioner", "children_playing", "engine_idling"]
size = [-9, -12, -12, -16, -12, -12, -14, -19, -20, -17]
# Access all PNG files in directory
i = 0
for label in labels:
  print(label)
  allfiles=os.listdir('./images/' + label)
#  imlist=[filename for filename in allfiles if  filename[size[i]:] in [label + ".png"]]
  notlist=[filename for filename in allfiles if filename[size[i]:] not in [label + ".png"]]
  notlist=[filename for filename in notlist if filename[-8:] not in ["flat.png"]]
  notlist=[filename for filename in notlist if filename[-8:] not in ["ilon.png"]]
  notlist=[filename for filename in notlist if filename[-4:] in [".png"]]
  for x in notlist:
      name = x.split("_")
      print(name)
      wronglist = [filename for filename in notlist if filename.split('_')[0] in [name[0]]]
      for y in wronglist:
          name = y.split('_')[0]
          wronglabel = x.split('_', 1)[1]
          print(wronglabel)
          imagename = "./images/" + label +"/" + name + "_" + wronglabel
          im1 = Image.open(imagename).convert('LA').convert('RGBA')
          imagename = "./images/" + label +"/" + name + "_" + "lrp_flat.png"
          im2 = Image.open(imagename)
          im3 = ImageChops.multiply(im2, im1)
          im3.save("./images/final/" + label + "/miss/" + name + "_miss_" + wronglabel[:-4] + "_overlayed.png")
  i += 1
i = 0
for label in labels:
  for x in imlist:
      name = x.split("_")
      correctlist = [filename for filename in imlist if  filename[0:len(name[0])] in [name[0]]]
      for y in correctlist:
          name = y.split('_')[0]
          imagename = "./images/" + label +"/" + name + "_" + label + ".png"
          im1 = Image.open(imagename).convert('LA').convert('RGBA')
          imagename = "./images/" + label +"/" + name + "_" + "lrp_flat.png"
          im2 = Image.open(imagename)
          im3 = ImageChops.multiply(im2, im1)
          im3.save("./images/final/" + label + "/" + name + "_overlayed.png")
  i += 1
