#!/bin/bash

echo -n "Enter number of iterations"
read iters
echo -n "Enter the initialized learning rate"
read lr
echo -n "Enter the lr decay rate"
read decay
#echo -n "Enter number of clients"
#read numClients

mkdir gradients
mkdir history
mkdir weights
mkdir results
mkdir clients_bn

for ((iter =1;iter <=$iters;iter++))
do
	if ((iter % 10 ==0))
	then
		echo "----------------- iteration # $iter------------------------"
	fi
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
	

