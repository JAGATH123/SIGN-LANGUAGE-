# Sign Language to Text and Speech Conversion

## ABSTRACT:

Sign language is one of the oldest and most natural forms of language for communication, hence we have come up with a real-time method using neural networks for finger spelling-based American Sign Language. Automatic human gesture recognition from camera images is an interesting topic for developing vision. We propose a convolution neural network (CNN) method to recognize hand gestures of human actions from an image captured by a camera. The purpose is to recognize hand gestures of human task activities from a camera image. The position of the hand and orientation are applied to obtain the training and testing data for the CNN. The hand is first passed through a filter and after the filter is applied where the hand is passed through a classifier which predicts the class of the hand gestures. Then the calibrated images are used to train CNN.

The Final Outcome Of Our Project...



## Introduction:

American sign language is a predominant sign language. Since the only disability D&M people have been communication-related and they cannot use spoken languages, the only way for them to communicate is through sign language. Communication is the process of exchange of thoughts and messages in various ways such as speech, signals, behavior, and visuals. Deaf and dumb (D&M) people make use of their hands to express different gestures to express their ideas with other people. Gestures are nonverbally exchanged messages, and these gestures are understood with vision. This nonverbal communication of deaf and dumb people is called sign language.

In our project, we basically focus on producing a model which can recognize Fingerspelling based hand gestures in order to form a complete word by combining each gesture. The gestures we aim to train are as given in the image below.



## Requirements:

More than 70 million deaf people around the world use sign languages to communicate. Sign language allows them to learn, work, access services, and be included in the communities.

It is hard to make everybody learn the use of sign language with the goal of ensuring that people with disabilities can enjoy their rights on an equal basis with others.

So, the aim is to develop a user-friendly human-computer interface (HCI) where the computer understands the American sign language. This Project will help dumb and deaf people by making their life easier.

## Objective:
To create a computer software and train a model using CNN which takes an image of hand gesture of American Sign Language and shows the output of the particular sign language in text format converts it into audio format.

## Scope:
This System will be beneficial for both dumb/deaf people and people who do not understand the sign language. They just need to use sign language gestures, and this system will identify what they are trying to say. After identification, it gives the output in the form of text as well as speech format.

## Modules:

### Data Acquisition:

The different approaches to acquire data about the hand gesture can be done in the following ways:

It uses electromechanical devices to provide exact hand configuration, and position. Different glove-based approaches can be used to extract information. But it is expensive and not user-friendly.

In vision-based methods, the computer webcam is the input device for observing the information of hands and/or fingers. The vision-based methods require only a camera, thus realizing a natural interaction between humans and computers without the use of any extra devices, thereby reducing costs. The main challenge of vision-based hand detection ranges from coping with the large variability of the human hand’s appearance due to a huge number of hand movements, to different skin-color possibilities as well as to the variations in viewpoints, scales, and speed of the camera capturing the scene.



### Data Pre-processing and Feature Extraction:

In this approach for hand detection, firstly we detect hand from image that is acquired by webcam and for detecting a hand we used MediaPipe library which is used for image processing. So, after finding the hand from the image we get the region of interest (ROI), then we crop that image and convert it to a grayscale image using OpenCV library. After that, we apply Gaussian blur. The filter can be easily applied using the OpenCV library. Then we convert the grayscale image to binary image using threshold and adaptive threshold methods.

We have collected images of different signs from different angles for sign letters A to Z.


In this method, there are many loopholes like your hand must be in front of a clean soft background and in proper lighting condition, then only this method will give good accurate results. But in the real world, we don’t get good background everywhere and we don’t get good lighting conditions too. So, to overcome this situation, we tried different approaches and reached an interesting solution. Firstly, we detect hand from frame using MediaPipe and get the hand landmarks of hand present in that image. Then we draw and connect those landmarks in a simple white image.

### Mediapipe Landmark System:


Now we get these landmark points and draw them on a plain white background using OpenCV library.

By doing this, we tackle the situation of background and lighting conditions because the MediaPipe library will give us landmark points in any background and mostly in any lighting conditions.


We have collected 180 skeleton images of Alphabets from A to Z.

### Gesture Classification:

#### Convolutional Neural Network (CNN)

CNN is a class of neural networks that are highly useful in solving computer vision problems. They found inspiration from the actual perception of vision that takes place in the visual cortex of our brain. They make use of a filter/kernel to scan through the entire pixel values of the image and make computations by setting appropriate weights to enable detection of a specific feature. CNN is equipped with layers like convolution layer, max pooling layer, flatten layer, dense layer, dropout layer, and a fully connected neural network layer. These layers together make a very powerful tool that can identify features in an image. The starting layers detect low-level features that gradually begin to detect more complex higher-level features.

Unlike regular neural networks, in the layers of CNN, the neurons are arranged in 3 dimensions: width, height, depth.

The neurons in a layer will only be connected to a small region of the layer (window size) before it, instead of all of the neurons in a fully-connected manner.

Moreover, the final output layer would have dimensions (number of classes), because by the end of the CNN architecture we will reduce the full image into a single vector of class scores.

#### Convolutional Layer:

In the convolution layer, we have taken a small window size [typically of length 5*5] that extends to the depth of the input matrix.

The layer consists of learnable filters of window size. During every iteration, we slide the window by stride size [typically 1], and compute the dot product of filter entries and input values at a given position.

As we continue this process, we create a 2-Dimensional activation matrix that gives the response of that matrix at every spatial position.

That is, the network will learn filters that activate when they see some type of visual feature such as an edge of some orientation or a blotch of some color.

#### Pooling Layer:

We use the pooling layer to decrease the size of the activation matrix and ultimately reduce the learnable parameters.

There are two types of pooling:

1. **Max Pooling**: In max pooling, we take a window size [for example, window of size 2*2], and only take the maximum of 4 values. We slide this window and continue this process, so we finally get an activation matrix half of its original size.
2. **Average Pooling**: In average pooling, we take the average of all values in a window.


#### Fully Connected Layer:

In the convolution layer, neurons are connected only to a local region, while in a fully connected region, all the inputs are connected to neurons.



The preprocessed 180 images/alphabet will feed the Keras CNN model.

Because we got bad accuracy in 26 different classes, we divided the whole 26 different alphabets into 8 classes in which every class contains similar alphabets:
- [y, j]
- [c, o]
- [g, h]
- [b, d, f, i, u, v, k, r, w]
- [p, q, z]
- [a, e, m, n, s, t]

All the gesture labels will be assigned with a probability. The label with the highest probability will be treated as the predicted label.

So when the model classifies [aemnst] in one single class using mathematical operations on hand landmarks, we will classify further into a single alphabet a or e or m or n or s or t.

Finally, we got 97% accuracy (with and without clean background and proper lighting conditions) through our method. And if the background is clear and there is good lighting condition, then we got even 99% accurate results.



### Text To Speech Translation:

The model translates known gestures into words. We have used the pyttsx3 library to convert the recognized words into the appropriate speech. The text-to-speech output is a simple workaround, but it's a useful feature because it simulates a real-life dialogue.

## Project Requirements:

### Hardware Requirement:
- Webcam

### Software Requirement:
- Operating System: Windows 8 and Above
- IDE: PyCharm
- Programming Language: Python 3.9
- Python libraries: OpenCV, NumPy, Keras, MediaPipe, TensorFlow


