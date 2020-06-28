# Handwriting Recognition

This repository contains the code to process a page containing handwritten Hebrew characters. The program segments lines, segments characters from these lines and then classifies each handwritten character. Moreover, the program will determine the style in which the page was originally written (*Archaic*, *Hasmonean* or *Herodian*).

## Installation
The required installs are ```python3``` and ```pip3```. 

## Setup

Clone the repo:
```bash
git clone https://github.com/ThomasB94/Handwriting-Recognition.git
```
The required packages are provided in the ```requirements.txt```, to install them:
```bash
pip3 install -r requirements.txt
```
Download the classifiers from [drive](https://drive.google.com/drive/folders/1_r2A1dqMXkP0lREvzFmkZRiNyglhF6Ij?usp=sharing).
> Store both classifiers (.pickle files) in the folder Handwriting-Recognition/recognition
## Usage

To run the program:
```bash
python3 main.py path/to/images
```
**Execution on Windows is not recommended**

## Authors
**Daan Lambert, Ivar Mak, Paul Hofman, Thomas Bakker**