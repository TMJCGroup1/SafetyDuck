# SafetyDuck
This project implements computer vision using PeekingDuck at construction sites to ensure all contruction works are done under safety condition.
## Features
### Helmet_detection
It can check whether all construction workers are wearing safety helmet in the construction site.

![Screenshot (38)](https://user-images.githubusercontent.com/124423670/216811595-99f6e432-b31e-4b99-a5cf-0f94dd256cff.png)
![Screenshot (37)](https://user-images.githubusercontent.com/124423670/216811617-aa0d1fa1-ab70-45bb-9327-954532ee6d84.png)

### People_detection
It can count the number of people at a site and determine their safety status. Example, at least one competent safety observer must always be there when workers are engaging in working at heights.

![ddsgdsafgs](https://user-images.githubusercontent.com/124423670/216811422-c60da4a5-e541-4afd-9983-2dec07e66d45.png)
![sfgbdgfh](https://user-images.githubusercontent.com/124423670/216811475-f5570d33-6e2c-4a80-8be0-ed3e70638632.png)

## Installation
Install [PeekingDuck](https://github.com/aisingapore/PeekingDuck#readme)

Clone the GitHub repository in the terminal using ```git```

```
git clone https://github.com/TMJCGroup1/SafetyDuck
```
Or download the repository using the download function in GitHub
## Usage
Go to the SafetyDuck directory.
```
cd SafetyDuck
```
### Helmet Detection
```
cd helmet_detection
```
In the [pipeline_config.yml](https://github.com/TMJCGroup1/SafetyDuck/blob/main/helmet_detection/pipeline_config.yml), the config is as such:
```
nodes:
- input.visual:
   source: trial.mp4  #change this (0 for webcam)
- model.mtcnn
- custom_nodes.draw.face_rectangle
- custom_nodes.model.helmet_detect
- output.screen
```
OPTIONAL:
By default, the AI will run our trial.mp4, to change the source, simply change the ```source``` config to the path of the file you want to run.\
For webcam detection,
```source: 0```  \
Run the program in terminal.
```
peekingduck run
```
### Observer Detection
``` 
cd observer_detection
```
In the [pipeline_config.yml](https://github.com/TMJCGroup1/SafetyDuck/blob/main/people_detection/pipeline_config.yml), the config is as such:
```
nodes:
- input.visual:
   source: true_positive.mp4  #change this to (true_negative.mp4) for unsafety condition
- model.mtcnn
- custom_nodes.draw.face_rectangle
- custom_nodes.model.helmet_detect
- output.screen
```
``` 
peekingduck run
```
