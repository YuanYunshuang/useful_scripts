# useful_scripts
##  	generate_OCR_synthetic_data.py
Generation pipeline is as following:
* Randomly generate a string of numbers
* Randomly chodose a text style to write this string in an image and label each digit with a bounding box
* Do a small rotation for the image and the bounding boxes
* Perform projective transformation for the images and bounding boxes
* Add noise and blur the images
