# Define the paths as variables
PYTHON_EXEC=/home/zixian/.conda/envs/semantic_graph_code_clone/bin/python
PROJECT_DIR=/home/zixian/PycharmProjects/semantic_graph_code_code_clone
SCRIPT=$PROJECT_DIR/main.py
CONFIG_DIR=$PROJECT_DIR/configs/benchmark

# Run the scripts
echo "Running first configuration: AST_GMN_BCB.json"
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_GMN_BCB.json &

# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 300
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_VALUE_GMN_BCB.json


wait
echo "Both processes have completed."