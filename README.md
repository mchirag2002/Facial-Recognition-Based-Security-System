# Facial Recognition Based Security System
## Aim of the Project:
The project was made as a security solution using Facial Recognition to detect the presence of any intruder or unauthorized person in the proximity.

## Requirements and Dependencies:
- Arduino Board
- Arduino Drivers
- Python 3.9 or above
- Facial Recognition
- OpenCV
- Pyfirmata

## Features of the Project:
- Detects the face of every individual entering the proximity
- If the person is unauthorized, the Alarm is triggered and a photograph of the intruder is taken and saved

## How to install and run
- Clone the repository into your machine
- Install all the dependencies mentioned
- Connect your Arduino board to the computer and note the serial port
- Check and verify the serial port if it is COM3, if not then change the port in security.py file
- Place the high resolution pictures of all the authorized personel in the database folder
- Run the security.py file