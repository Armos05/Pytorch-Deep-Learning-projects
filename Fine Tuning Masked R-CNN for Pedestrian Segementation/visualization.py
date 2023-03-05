from PIL import Image
root = 'C:/bio/PORTFOLIO/data/PennFudanPed/'
mask = Image.open((root + 'PennFudanPed/PNGImages/FudanPed00001.png'))
# each mask instance has a different color, from zero to N, where
# N is the number of instances. In order to make visualization easier,
# let's adda color palette to the mask.
mask.putpalette([
    0, 0, 0, # black background
    255, 0, 0, # index 1 is red
    255, 255, 0, # index 2 is yellow
    255, 153, 0, # index 3 is orange
    ])
print(mask)