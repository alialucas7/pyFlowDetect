<h1 align="center"> 🐍 pyFlowDetect </h1>

Detect port scans in your network with python | Detecte escaneos de puertos dentro de su red con python.

This project uses machine learning and network traffic analysis techniques for port scan detection. It combines data extraction, preprocessing and classification with algorithms such as Decision Trees and Random Forest. It is a tool designed to strengthen cybersecurity through a practical, technical and scalable approach.

## Install | Instalacion
### Linux

```bash
pip install pyflow-detect
```


## Usage | Uso
This project needs several `.argus files`, i.e. network flow information files, stored in `./trainData/netflows`. These files must have legitimate network flows and port scan network flows. You can generate those files using argus and argus clients to record network activity, or converting existing .pcap files to a netflow version (.argus). Refer to [argus documentation](https://openargus.org/using-argus) on how to do that.

One condition to generete these files is to keep track of wich computers in the network are the attackers, and wich ones are innocents, i.e. we need their ips. Then `variables.json` file needs these ips in scannerIps and targetIps properties respectively. Aditionally it needs the password for sudo privileges when running the trainer.

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

[![mydecisiontree.png](https://i.postimg.cc/rpKw7dxR/mydecisiontree.png)](https://postimg.cc/gwbpZ2MG)

* In case you want to see the procedure step by step or run it in jupyter notebook you can use `Entrenamiento.ipynb`

2. `pyDetect.py` To see the model in action run `pyDetect.py` to view a real time netflow clasification. It will search for a model called
   `rFOrest.pkl` and it will use argus in daemon mode to fetch the network traffic on the machine.





## License
This project is licensed under MIT. Contributions to this project are accepted under the same license.










