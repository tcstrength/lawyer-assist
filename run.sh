#!/bin/bash

export PYTHONPATH=/Users/tccuong1404/Documents/Project/lawyer-assist/src

if [ "$1" == "api" ]; then
    poetry run fastapi run src/api/main.py
elif [ "$1" == "gui" ]; then
    poetry run streamlit run src/gui/main.py
fi