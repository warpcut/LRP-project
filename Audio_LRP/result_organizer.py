import os, numpy, PIL
from shutil import copyfile
labels = ["siren", "car_horn", "gun_shot", "street_music", "drilling", "dog_bark", "jackhammer", "air_conditioner", "children_playing", "engine_idling"]
#label = "siren"
#label = "car_horn"
#label = "gun_shot"
#label = "street_music"
#label = "drilling"
#label = "dog_bark"
#label = "jackhammer"
#label = "air_conditioner"
#label = "children_playing"
#label = "engine_idling"

# Access all PNG files in directory
for label in labels:
  allfiles=os.listdir('../../cq/results/' + label)
  print(label)
  imlist=[filename for filename in allfiles if  filename[-17:] in [label + ".png"]]
  print(len(imlist))
  for x in imlist:
      name = x.split("_")
      correctlist = [filename for filename in allfiles if  filename[0:len(name[0])] in [name[0]]]
      for y in correctlist:
          if y[-8:] == "flat.png":
              copyfile("../../cq/results/"+ label +"/" + y, "../../cq/results/"+ label +"/correct/"+ y)
