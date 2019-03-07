
import xml.etree.ElementTree as ET
import glob

labels=[]
box=[]
file=[]
c=0
sizes=[]
box2=[]
for f in glob.glob("C:/Users/Tina/data/VOCdevkit/VOC2012/Annotations/*.xml"): #path to annotation files of Pascal VOC dataset
    s=[]
    tree = ET.parse(f)
    root = tree.getroot()
    file.append(root.find('./filename').text)
    s.append((float(root.find('./object/bndbox/xmin').text),float(root.find('./object/bndbox/ymin').text),float(root.find('./object/bndbox/xmax').text),float(root.find('./object/bndbox/ymax').text)))
    sizes.append((float(root.find('./size/width').text),(float(root.find('./size/height').text))))
    box.append(s[0])
    box2.append(sizes[0])
    c=c+1
print(len(box))
print(sizes)
print(box2[1][1])
print('done 1')
with open('labels.txt', 'w') as f:
    for i in enumerate(file):

        f.write(str(file[i[0]]) + " " + str(box[i[0]][0]/sizes[i[0]][0]) + " " + str(box[i[0]][1]/sizes[i[0]][1]) + " " + str(box[i[0]][2]/sizes[i[0]][0]) + " " + str(box[i[0]][3]/sizes[i[0]][1]) + " " + str(box2[i[0]][0])+" "+str(box2[i[0]][1])+ " "+"\n")

f.close()
