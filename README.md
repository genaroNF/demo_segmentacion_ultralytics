# demo_segmentation_ultralytics
Small code that shows how to segment images using ultralytics

## Before running
Before anything change the setting of ultralytics to search for that datasets from the folder you are running this script or any script from, this can be done by running `vim ~/.config/Ultralytics/settings.yaml` and changing `datasets_dir` to `.`

## How to run the demo?
Remove the files in the folder `segmentation_ant_roads` and execute `python demo.py`, you will be prompted to select an ammount of epochs the default works and gets you some results.

## What can I do with this?
The idea behind this demo is just to show how to segment using ultralytics and how to show the results obtained by the model on an image, you are welcome to experiment with other datasets (you can look in roboflow) or to test different configurations for the YOLO model (the ultralytics documentation and github is really good).