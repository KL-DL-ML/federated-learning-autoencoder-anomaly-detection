#!/bin/bash

echo "Starting server"
python3 ./fl_server.py --server_address localhost:11000 --dataset ENERGY &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 1 6`; do
    echo "Starting client $i"
    python3 ./fl_client.py --server_address localhost:11000 --cid="dev$i" --dataset ENERGY &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait