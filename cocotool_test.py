from pycocotools.coco import COCO

import matplotlib.pyplot as plt
import cv2

import os
import numpy as np
import random

cocoRoot = 'D:\\soft\\shendu\\dataset\\coco2017\\'
dataType = "val2017"

annFile = os.path.join(cocoRoot, f'annotations\\instances_{dataType}.json')
print(f'Annotation file: {annFile}')

coco=COCO(annFile)


# 利用getCatIds函数获取某个类别对应的ID，
# 这个函数可以实现更复杂的功能，请参考官方文档
ids = coco.getCatIds('person')[0]
print(f'"person" 对应的序号: {ids}')

# 利用loadCats获取序号对应的文字类别
# 这个函数可以实现更复杂的功能，请参考官方文档
cats = coco.loadCats(1)
print(f'"1" 对应的类别名称: {cats}')

# 获取包含person的所有图片
imgIds = coco.getImgIds(catIds=[1])
print(f'包含person的图片共有：{len(imgIds)}张')

# 获取包含dog的所有图片
id = coco.getCatIds(['dog'])[0]
imgIds = coco.catToImgs[id]
print(f'包含dog的图片共有：{len(imgIds)}张, 分别是：')
print(imgIds)

imgId = imgIds[10]

imgInfo = coco.loadImgs(imgId)[0]
print(f'图像{imgId}的信息如下：\n{imgInfo}')

imPath = os.path.join(cocoRoot, 'images', dataType, imgInfo['file_name'])
imPath = 'D:\\soft\\shendu\dataset\\coco2017\\val2017\\000000000776.jpg'
im = cv2.imread(imPath)
plt.axis('off')
plt.imshow(im)
plt.show()


plt.imshow(im);
plt.axis('off')

# 获取该图像对应的anns的Id
annIds = coco.getAnnIds(imgIds=imgInfo['id'])
anns = coco.loadAnns(annIds)
print(f'图像{imgInfo["id"]}包含{len(anns)}个ann对象，分别是:\n{annIds}')

coco.showAnns(anns)

print(f'ann{annIds[3]}对应的mask如下：')
mask = coco.annToMask(anns[3])
plt.imshow(mask); plt.axis('off')

print(f'ann{annIds[3]}对应的mask如下：')
mask = coco.annToMask(anns[3])
plt.imshow(mask); plt.axis('off')


