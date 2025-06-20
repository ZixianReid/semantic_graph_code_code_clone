# Define the paths as variables
PYTHON_EXEC=/home/zixian/.conda/envs/semantic_graph_code_clone/bin/python
PROJECT_DIR=/home/zixian/PycharmProjects/semantic_graph_code_code_clone
SCRIPT=$PROJECT_DIR/main.py
CONFIG_DIR=$PROJECT_DIR/configs/benchmark_without_value_GGNN/BCB

$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_CFG_DFG_FA_GGNN_BCB.json &


# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_CFG_DFG_GGNN_BCB.json &


# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_CFG_FA_GGNN_BCB.json &

# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_CFG_GGNN_BCB.json &


# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_DFG_FA_GGNN_BCB.json &

# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_DFG_GGNN_BCB.json &

# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_FA_GGNN_BCB.json &

# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_GGNN_BCB.json &


wait
echo "Both processes have completed."