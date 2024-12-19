import subprocess
import time
import json
from subprocess import  PIPE, STDOUT

variables=json.load(open(f'variables.json',))

argusConfPath = variables["argusConfig"]
pwd = variables["password"]
demoDirPath = variables["demoData"]


def createArgusFilesOutput (dataDir):
    print("argus >> txt")
    subprocess.run(['bash', f'{argusConfPath}/argus_conversion.sh', f'{argusConfPath}', f'{dataDir}'])
    print("argus >> txt DONE")


def createArgusDaemonOutput (outputDir):
    subprocess.run(f'echo kali | sudo -S pkill argus', shell=True, capture_output=True)
    time.sleep(5)
    print("init argus daemon")
    subprocess.run(f'echo {pwd} | sudo -S argus -P 561 -d', shell=True, capture_output=True)
    time.sleep(5)
    print("start netflows capture")
    p = subprocess.Popen(f'exec ra -F {argusConfPath}/rarc -S 127.0.0.1:561 | grep -v "CON"', shell=True, bufsize=3, stdout=PIPE)
    return p

#exec ra -F {argusConfPath}/rarc -S 127.0.0.1 > "./demoData"/realData.txt


createArgusDaemonOutput(demoDirPath)
