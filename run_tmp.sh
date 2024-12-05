PYTHON_EXEC=/home/zixian/.conda/envs/semantic_graph_code_clone/bin/python
PROJECT_DIR=/home/zixian/PycharmProjects/semantic_graph_code_code_clone
SCRIPT=$PROJECT_DIR/main.py
CONFIG_DIR_BCB=$PROJECT_DIR/configs/benchmark_without_value/BCB
CONFIG_DIR_GCJ=$PROJECT_DIR/configs/benchmark_without_value/GCJ

# Run the scripts
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR_BCB/AST_GMN_BCB.json &

# Wait for 30 minutes (1800 seconds)
echo "Waiting for 5 minutes before executing the next command..."
sleep 1200
$PYTHON_EXEC $SCRIPT --config $CONFIG_DIR_GCJ/AST_GMN_GCJ.json &

wait
echo "Both processes have completed."