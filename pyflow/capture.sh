#!/bin/bash
ip=192.168.x.x; #192.168.43.203;
maxTimeSinceMin=1;
minTimeBetweenScans=1;
outputDirName='./quickScanGenerationFiles';
SYNFile="${outputDirName}/syn_output.txt";
ConnectFile="${outputDirName}/connect_output.txt";
UDPFile="${outputDirName}/udp_output.txt";
FINFile="${outputDirName}/fin_output.txt";
XMASFile="${outputDirName}/xmas_output.txt";
OSFile="${outputDirName}/os_output.txt";
VersionFile="${outputDirName}/version_output.txt";

mkdir -p "${outputDirName}"
chmod ugo+rwx "${outputDirName}"
if [ ! -e "$SYNFile" ] ; then
 touch $SYNFile
 chmod ugo+rwx $SYNFile
fi
if [ ! -e "$ConnectFile" ] ; then
 touch $ConnectFile
 chmod ugo+rwx $ConnectFile
fi
if [ ! -e "$UDPFile" ] ; then
 touch $UDPFile
 chmod ugo+rwx $UDPFile
fi
if [ ! -e "$FINFile" ] ; then
 touch $FINFile
 chmod ugo+rwx $FINFile
fi
if [ ! -e "$XMASFile" ] ; then
 touch $XMASFile
 chmod ugo+rwx $XMASFile
fi
if [ ! -e "$OSFile" ] ; then
 touch $OSFile
 chmod ugo+rwx $OSFile
fi
if [ ! -e "$VersionFile" ] ; then
 touch $VersionFile
 chmod ugo+rwx $VersionFile
fi

today=`date '+%Y-%m-%d %H:%M:%S'`;
getDate() {
 today=`date '+%Y-%m-%d %H:%M:%S'`;
}
fast='';
addFastScan() {
 fast='';
 addParameter=$(($RANDOM%2))
 if [ "$addParameter" -eq 1 ] ; then
 fast=' -F';
 fi
}
delay='';
addScanDelay() {
 d=$(($RANDOM%500))
 delay=" --scan-delay ${d}ms";
}
payload=''
addRandomPayload() {
 bytes=$(($RANDOM%256))
 payload=" --data-length ${bytes}";
}

doSYNScan () {
 echo -e '####################################################'$"\n${today}\n" | tee -a $SYNFile
 echo -e "nmap -sS -Pn ${fast} ${delay} ${payload} ${ip}"$"\n"
 nmap -sS -Pn $fast $delay $payload $ip| tee -a $SYNFile
}
doCONNECTscan () {
 echo -e '####################################################'$"\n${today}\n" | tee -a $ConnectFile
 echo -e "nmap -sT -Pn ${fast} ${delay} ${payload} ${ip}"$"\n"
 nmap -sT -Pn $fast $delay $payload $ip | tee -a $ConnectFile
}
doUDPScan() {
 echo -e '####################################################'$"\n${today}\n" | tee -a $UDPFile
 echo -e "nmap -sU -Pn ${fast} ${delay} ${payload} ${ip}"$"\n"
 nmap -sU -Pn $fast $delay $payload $ip | tee -a $UDPFile
}

doFINScan() {
 echo -e '####################################################'$"\n${today}\n" | tee -a
 $FINFile echo -e "nmap -sF -Pn ${fast} ${delay} ${payload} ${ip} "$"\n"
 nmap -sF -Pn $fast $delay $payload $ip | tee -a $FINFile
}

doXMASScan() {
 echo -e '####################################################'$"\n${today}\n" | tee -a $XMASFile
 echo -e "nmap -sX -Pn ${fast} ${delay} ${payload} ${ip}"$"\n"
 nmap -sX -Pn $fast $delay $payload $ip | tee -a $XMASFile
}
doOSScan() {
 echo -e '####################################################'$"\n${today}\n" | tee -a $OSFile
 echo -e "nmap -O -Pn ${fast} ${delay} ${payload} ${ip}"$"\n"
 nmap -O -Pn $fast $delay $payload $ip | tee -a $OSFile
}
doVersionScan() {
 echo -e '####################################################'$"\n${today}\n" | tee -a $VersionFile
 echo -e "nmap -sV -Pn ${fast} ${delay} ${payload} ${ip}"$"\n"
 nmap -sV -Pn $fast $delay $payload $ip | tee -a $VersionFile
}




randomScanGenerator() {
 echo "Scanning host ${ip}"
 while :
 do
 scanToDo=$(($RANDOM%7))
 getDate
 addFastScan
 addScanDelay
 addRandomPayload
 case $scanToDo in
 0)
 doSYNScan
 ;;
 1)
 doCONNECTscan
 ;;
 2)
 doUDPScan
 ;;
 3)
 doFINScan
 ;;
 4)
 doXMASScan
 ;;
 5)
 doOSScan
 ;;
 6)
 doVersionScan
 ;;
 esac
 sleepTime=$(( ($RANDOM % $2) + $1))
 echo $'\n\n';
 echo "next scan will start in ${sleepTime} minutes :)"
 echo $'\n\n';
 sleep $(($sleepTime*60))
 done
}

randomScanGenerator




