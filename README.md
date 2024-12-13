<h1 align="center"> 🐍 pyFlowDetect </h1>

Detect port scans in your network with python | Detecte escaneos de puertos dentro de su red con python.

This project uses machine learning and network traffic analysis techniques for port scan detection. It combines data extraction, preprocessing and classification with algorithms such as Decision Trees and Random Forest. It is a tool designed to strengthen cybersecurity through a practical, technical and scalable approach.

##Install | Instalacion
### Linux

```bash
pip install pyflow-detect
```


##Usage | Uso
This project needs several .argus files, i.e. network flow information files, stored in "./trainData/netflows" folder. These files must have legitimate network flows and port scan network flows. You can generate those files using argus and argus clients to record network activity, or converting existing .pcap files to a netflow version (.argus). Refer to [argus documentation](https://openargus.org/using-argus) on how to do that.

One condition to generete these files is to keep track of wich computers in the network are the attackers, and wich ones are innocents, i.e. we need their ips. Then variables.json file needs these ips in scannerIps and targetIps properties respectively. Aditionally it needs the password for sudo privileges when running the trainer.

variables.json
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
Finnally running 
1. `train.py` file will generate a bagging trained model with the following steps:

[![mydecisiontree.png](https://i.postimg.cc/rpKw7dxR/mydecisiontree.png)](https://postimg.cc/gwbpZ2MG)


















