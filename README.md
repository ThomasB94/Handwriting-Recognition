# Handwriting Recognition

This repository contains the code to process a page containing handwritten Hebrew characters. The program segments lines, segments characters from these lines and then classifies each handwritten character. Moreover, the program will determine the style in which the page was originally written (*Archaic*, *Hasmonean* or *Herodian*).

## Installation
The required installs are ```python3``` and ```pip3```.

## Setup

The required packages are provided in the ```requirements.txt```, to install them:

```bash
pip3 install -r requirements.txt
```

## Usage

To run the program:
```bash
python3 main.py path/to/images
```
**Execution on Windows is not recommended**

## Authors
**Daan Lambert, Ivar Mak, Paul Hofman, Thomas Bakker**