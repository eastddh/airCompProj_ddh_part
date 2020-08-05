#!/bin/bash

echo -n "Enter the initialized learning rate"
read lr
echo -n "Enter the lr decay rate"
read decay
for ((iter =1; iter <=100; iter++))
do
	echo "-----------------epoch $iter------------------------"
	if ((iter == 1)) 
	then
		python server.py 1 $iter $lr $decay
	else
		python channelAverage.py 2
		python server.py 0 $iter $lr $decay
	fi

	for ((clients = 1; clients <=2; clients++))
	do
		if ((iter == 1))
		then
			python client.py 1 $clients
		else
			python client.py 0 $clients
		fi
	done
done 
	

