#!/bin/bash

echo "Starting Runny app..."

# Change directory to the app folder
cd "/Users/maurii/Desktop/Runny" || exit

# Wait a couple seconds to ensure server is up, then open Firefox
(sleep 2 && open -a Firefox "http://127.0.0.1:5050/") &

# Run the app 
# (If you are using a virtual environment, uncomment the line below)
# source venv/bin/activate

python3 app.py
