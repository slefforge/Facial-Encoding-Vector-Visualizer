# Facial-Encoding-Vector-Visualizer

This is the tool I created to try and visualize the components of the face_recognition library's facial encoding vector. It's really slow and inefficent, so if time is a concern there are plenty of optimizations to be made.
In this file is a pdf containing the results for the LFW face database (http://vis-www.cs.umass.edu/lfw/). If you see a pattern in these results, feel free to contribute to the theory.txt file.
The face_recognition library used to calculate the facial encoding vector can be found here: https://github.com/ageitgey/face_recognition

If you would like to run this yourself:
* Make sure the face_encoding library is installed
* Format your data folder like so:
```
data
|____bob
|    |____bob_0001.jpg
|
|____alice
     |____alice_0001.jpg
```
* You can have multiple images in a folder, but the visualizer will only consider the first one (the one named [foldername]_0001.jpg)
