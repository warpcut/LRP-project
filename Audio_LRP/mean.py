import os, numpy, PIL
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
labels = ["siren","car_horn","gun_shot","street_music","drilling","dog_bark","jackhammer","air_conditioner","children_playing","engine_idling" ]
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
    allfiles=os.listdir('../../cq/results/' + label + '/correct/')
    imlist=[filename for filename in allfiles if  filename[-4:] in [".png"]]
    N=len(imlist)
    avg=Image.open('../../cq/results/' + label + '/correct/'+ imlist[0])
    for i in range(1,N):
        img=Image.open('../../cq/results/' + label + '/correct/' + imlist[i])
        avg=Image.blend(avg,img,1.0/float(i+1))
    avg.save("../../cq/avgs/" + label + "_average.png")
