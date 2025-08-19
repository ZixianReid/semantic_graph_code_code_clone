# Define the paths as variables
PYTHON_EXEC=$(which python)
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
SCRIPT=$PROJECT_DIR/main.py
CONFIG_DIR=$PROJECT_DIR/configs/benchmark_without_value_GAT/BCB


$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_CFG_DFG_FA_GAT_BCB.json &


# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_CFG_DFG_GAT_BCB.json &


# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_CFG_FA_GAT_BCB.json &

# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_CFG_GAT_BCB.json &


# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_DFG_FA_GAT_BCB.json &

# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_DFG_GAT_BCB.json &

# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_FA_GAT_BCB.json &

# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR/AST_GAT_BCB.json &


wait
echo "Both processes have completed."