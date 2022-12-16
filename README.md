
## Start

<details>
<summary>Installation</summary>

Step1. Install YOLOX from source.
```shell
git clone https://github.com/jimklee2/human-detection-with-socket.git
cd YOLOX
pip3 install -v -e .  # or  python3 setup.py develop
```

</details>

<details>
<summary>Download ckpt file</summary>

Download 90epoch_ckpt.zip from 
[YOLOX(Human-detection)](https://github.com/jimklee2/human-detection-with-socket/releases)


</details>


<details>
<summary>Run model</summary>

Step1. Set the path to the downloaded 90epoch_ckpt file in the server.py file


Step2. Run server.py in your Jetson board
```shell
python3 server.py
```

Step3. Run client.py in your Host PC
```shell
python3 client.py
```

</details>


<details>
<summary>Run model directly(without using Jetson)</summary>

Step1. Set the path to the downloaded 90epoch_ckpt file in the server.py file


Step2. Run direct_inference.py in PC
```shell
python3 direct_inference.py
```

</details>

