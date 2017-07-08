# Self-driving truck
## Applied Computer Vision project

The aim of this project is to simulate a simple self-driving vehicle in a video game. I chose 
[Euro Truck Simulator 2](https://eurotrucksimulator2.com/) because of its simplicity and customization options.  

Credits to [PyGTA5](https://github.com/Sentdex/pygta5) for inspiration and the general idea of the program, 
however, my solution is a bit different - it involves
more image processing and a better AI model than his first simple model. Since this project was made for a computer vision course, 
I decided not to follow his second approach of using a neural network and discarding lane finding.

This project is written in Python 3 (using OpenCV) and made to run on Windows (I use Windows APIs for simulating 
keypresses and capturing the screen, but these can be switched with the appropriate replacements for other operating systems).

The processing pipeline is as follows:
* capture a screenshot from the game
* process the image, detect lines and find road lanes
* if learning, append the data and detected pressed key
* if driving, use the model (a decision tree) to choose which key to press

The in-game camera should be set to a hood view. A window showing the processed image with the detected lanes can be shown for monitoring purposes.
 I also included an option to create a video showing the lanes on top of the captured images - note that this uses a lot of RAM since it stores
 all of the images in memory before creating the video.

An example of lanes found in image:
![Lanes](https://github.com/janhartman/SelfDrivingTruckCV/raw/master/screens/img_lanes.jpg)
