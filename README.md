<h1 align="center"> 🐍 pyFlowDetect </h1>

[![PyPI version](https://badge.fury.io/py/pyflow-detect.svg)](https://pypi.org/project/pyflow-detect/)


Detect port scans in your network with python | Detecte escaneos de puertos dentro de su red con python.

This project uses machine learning and network traffic analysis techniques for port scan detection. It combines data extraction, preprocessing and classification with algorithms such as Decision Trees and Random Forest. It is a tool designed to strengthen cybersecurity through a practical, technical and scalable approach.

## Dependencies External| Dependencias Externas
| python - pip       | 
|--------------|
| pyenv        | 
| open-argus   | 
| Wireshark    | 

## Install | Instalacion
### Linux
In the project clone folder run
```bash
pyenv virtualenv 3.10.12 py_flow_detect
```
```bash
pyenv activate py_flow_detect
```
```bash
pip install -r requirements.txt
```


## Usage | Uso
This project needs several `.argus files`, i.e. network flow information files, stored in `./trainData/netflows`. These files must have legitimate network flows and port scan network flows. You can generate those files converting existing .pcap files to a netflow version (.argus). Refer to [argus documentation](https://openargus.org/using-argus) on how to do that.

**NOTE:** For ethical reasons this repository does not provide the dataset, but the steps to generate it are indicated.<br>
* To generate the dataset you need to scan the ports of the victim machine, you can do it manually with nmap or run`pyflow/capture.sh` indicating the ip inside the script, then simultaneously capture the network traffic with wireshark.
* Do it in a fully controlled environment and under your responsibility.
* One condition to generete these files is to keep track of wich computers in the network are the attackers, and wich ones are innocents, i.e. we need their ips. Then `pyflow/variables.json` file needs these ips in scannerIps and targetIps properties respectively. Aditionally it needs the password for sudo privileges when running the trainer.

The `variables.json` file  looks like
```json
{
    "argusConfig": "./netflowConfFiles",
    "trainingData": "./trainData/netflows",
    "demoData": "./demoData",
    "scannerIps": ["scanner ip here", "scanner ip here"], 
    "targetIps": ["target ip here", "target ip here"] ,
    "password": "password here"
}
```
Finnally running the following
1. `train.py` file will generate a bagging trained model with the following steps:



* In case you want to see the procedure step by step or run it in jupyter notebook you can use `Entrenamiento.ipynb`

  
  At this point the dataframe is ready to be used in training. Once the training ends, two grapichs are displayed, the first decision tree of the Random Forest model
  [![mydecisiontree.png](https://i.postimg.cc/rpKw7dxR/mydecisiontree.png)](https://postimg.cc/gwbpZ2MG)
  <p align="center">Adding as a precision metric the confusion matrix </p>
  <p align="center">
  <img src="https://github.com/alialucas7/pyFlowDetect/blob/master/matrix_confusion.png" alt="confusion_matrix"/>
  </p>
  <p align= "center">Finally, the most relevant columns of the model are shown below. </p>
  <p align= "center">
    <img src="https://github.com/alialucas7/pyFlowDetect/blob/master/columnas_ponderantes.png" alt="colum_relevant"/>
  
  

2. `py_detect.py` To see the model in action run `py_detect.py` to view a real time netflow clasification. It will search for a model called
   `rFOrest.pkl` and it will use argus in daemon mode to fetch the network traffic on the machine.



![definitivo (1)](https://github.com/user-attachments/assets/f8871984-8d47-4a3d-9b02-7e76acc91e64)

[![Virtual-Box-Kali-20-12-2024-15-48-40.png](https://i.postimg.cc/vTwrsTG3/Virtual-Box-Kali-20-12-2024-15-48-40.png)](https://postimg.cc/0MZMG8Mw)

## License
This project is licensed under MIT. Contributions to this project are accepted under the same license.










