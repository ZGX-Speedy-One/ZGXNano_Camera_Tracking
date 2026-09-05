Initial idea was to create a simple object detection on a HP ZGX Nano, based on an NVIDIA GB10, therefor ARM64 and Ubuntu 24.04 Linux OS.

Used Ultralytics YOLO26 to capture the video stream from an external camera.

Made the window the video is shown in resizable. 

instructions:


copy the live_yolo.py onto your system

> [!NOTE]
> ### How to run the tool locally
> run the following lines in CLI / Terminal. 2nd point optional after 1st run.
>
> # 1️⃣ Create & activate an isolated Python 3 env (optional but recommended)
> python3 -m venv venv && source venv/bin/activate
> 
> # 2️⃣ Install the two required libraries
> python3 -m pip install --upgrade pip
> python3 -m pip install ultralytics opencv-python   # installs both YOLO and OpenCV
> 
> # 3️⃣ Run the script
> python3 live_yolo.py        # <-- or whatever you named the file
>

exit object detection with CTRL-c 

exit venv with "exit" or closing the terminal 

have fun.
