# useful_scripts
##  	generate_OCR_synthetic_data.py
Generation pipeline is as following:
*  <span style="color:blue">Randomly generate a *string of numbers*</span>
* Randomly chodose a text style to write this string in an image and label each digit with a bounding box
* Do a small rotation for the image and the bounding boxes
* Perform projective transformation for the images and bounding boxes
* Add noise and blur the images

## label_generator_v1.py 
 see  __generate_OCR_synthetic_data.py__
 
## label_generator_v2.py
* Randomly generate a <span style="color:blue">*random string*</span>
* Randomly chodose a text style to write this string in an image and label each digit with a bounding box(8dim for 4 points)
* Do a small rotation for the image and the bounding boxes
* Perform projective transformation for the images and bounding boxes
* Add noise and blur the images
