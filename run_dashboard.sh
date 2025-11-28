#!/bin/bash
echo "Starting TFS Dashboard on network..."
echo ""
echo "Your dashboard will be available at:"
echo "http://YOUR_IP_ADDRESS:8501"
echo ""
echo "To find your IP address, run: ifconfig or ip addr"
echo ""
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501

