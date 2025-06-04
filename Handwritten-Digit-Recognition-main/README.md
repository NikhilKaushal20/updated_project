# Digit Recognition
Handwritten Digit Recognition

**for complete notes on how the code works.**

# Dependencies
1. `cv2`
2. `sklearn`
3. `skimage`
4. `numpy`
5. `collections`

# Contents
This repository contains the following files-

1. `generateClassifier.py` - Python Script to create the classifier file `digits_cls.pkl`.
2. `performRecognition.py` - Python Script to test the classifier.
3. `digits_cls.pkl` - Classifier file for digit recognition.
4. `photo_1.jpg` - Test image number 1 to test the classifier
5. `photo_2.jpg` - Test image numbre 2 to test the classifier

## Usage 

* Clone the repository - 
```bash
cd 
git clone https://github.com/bikz05/digit-recognition.git
cd digit-recognition
```
* The next step is to train the classifier. To do so run the script `generateClassifier.py`. It will produce the classifier named `digits_cls.pkl`. 

**NOTE** - *I have already created the `digits_cls.pkl`, so this step is not necessary.*
```python
python generateClassifier.py
```
* To test the classifier, run the `performRecognition.py` script.
```python
python performRecognition.py -c <path to classifier file> -i <path to test image>
```
ex -
```python
python performRecognition.py -c digits_cls.pkl -i photo_1.jpg
```

## Results

### Sample Image 1
![photo_1](https://github.com/user-attachments/assets/8cf62a9f-b1fb-4931-b02c-39fb4a7f72c4)

### Sample Image 2
![photo_2](https://github.com/user-attachments/assets/b3b06188-a4d7-4df4-a164-5223c44552c9)


## TODO

* Add a CNN Based approach
* Reject bounding boxes lesser than some area
* Look into user errors
