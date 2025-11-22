# Face-emotion-detector

# Emotion detection using deep learning

## Introduction

This project aims to classify the emotion on a person's face into one of **seven categories**, using deep convolutional neural networks. The model is trained on the **FER-2013** dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with **seven emotions** - angry, disgusted, fearful, happy, neutral, sad and surprised.

![personal](https://github.com/NIVION-HUB/Face-emotion-detection/face-detect/imgs/personal.jpg)

## Dependencies

- Python 3, [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/)
- To install the required packages, run `pip install -r requirements.txt`.

## Requirements

- numpy==1.23.5
- opencv-python==4.6.0.66
- tensorflow==2.10.0
- matplotlib

## Basic Usage

The repository is currently compatible with `tensorflow-2.10.0` and makes use of the Keras API using the `tensorflow.keras` library.

1. Firstly, clone the repo to your local machine and move to the diffrent driectory, using the following command.

```bash
git clone https://github.com/NIVION-HUB/Face-emotion-detection.git
cd Emotion-detection
```

2. Then download all the required dependencies by runnning following command.

```bash
pip install -r requirements.txt
```

3. After sucessfull istallation of all dependencies move to src and finally run emotion detection file.

```bash
cd src
python emotions.py --mode display
```

- The folder structure is of the form:  
  src:

  - data (folder)
  - `emotions.py` (file)
  - `haarcascade_frontalface_default.xml` (file)
  - `model.h5` (file)

## Camfeed Inference

This allows you to run the in real-time face emotion detection using your local camera feed. It open a window displaying the real-time video feed, with detected faces marked by bounding boxes and detected face expressions displayed on top of each boxes.

![Face detection](https://github.com/NIVION-HUB/Face-emotion-detection/face-detect/imgs/seven_emotion.jpeg)

- This implementation by default detects emotions on all faces in the webcam feed. With a simple 4-layer CNN, the test accuracy reached 63.2% in 50 epochs.

![Accuracy plot](imgs/accuracy.png)

## Data Preparation (optional)

- The [original FER2013 dataset in Kaggle](https://www.kaggle.com/deadskull7/fer2013) is available as a single csv file. I had converted into a dataset of images in the PNG format for training/testing.

- In case you are looking to experiment with new datasets, you may have to deal with data in the csv format. I have provided the code I wrote for data preprocessing in the `dataset_prepare.py` file which can be used for reference.

## Algorithm

- First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to **48x48** and is passed as input to the CNN.

* The network outputs a list of **softmax scores** for the seven classes of emotions.

* The emotion with maximum score is displayed on the screen.

- If you want to view the predictions without training again, you can download the pre-trained model from [here](https://drive.google.com/file/d/1FUn0XNOzf-nQV7QjbBPA6-8GLoHNNgv-/view?usp=sharing) and then run:

* i have already provided it insside folder if not, download it unzip and put it inside your src folder..

## Data Preparation (optional)

- The [original FER2013 dataset in Kaggle](https://www.kaggle.com/deadskull7/fer2013) is available as a single csv file. converted into a dataset of images in the PNG format for training/testing.

* download it unzip and put it inside your src folder..

- In case you are looking to experiment with new datasets, you may have to deal with data in the csv format. have provided the code wrote for data preprocessing in the `dataset_prepare.py` file which can be used for reference.

## References

- "Challenges in Representation Learning: A report on three machine learning contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
  Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,  
   X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
  M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
  Y. Bengio. arXiv 2013.
