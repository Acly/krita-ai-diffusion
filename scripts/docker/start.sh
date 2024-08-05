#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value

# ---------------------------------------------------------------------------- #
#                          Function Definitions                                #
# ---------------------------------------------------------------------------- #

# Start nginx service
start_nginx() {
    echo "Starting Nginx service..."
    service nginx start
}

# Execute script if exists
execute_script() {
    local script_path=$1
    local script_msg=$2
    local script_args=${@:3}
    if [[ -f ${script_path} ]]; then
        echo "${script_msg}"
        bash ${script_path} ${script_args}
    fi
}

# Setup ssh
setup_ssh() {
    if [[ $PUBLIC_KEY ]]; then
        echo "Setting up SSH..."
        mkdir -p ~/.ssh
        echo -e "${PUBLIC_KEY}\n" >> ~/.ssh/authorized_keys
        chmod 700 -R ~/.ssh
        service ssh start
    fi
}

# Export env vars
export_env_vars() {
    echo "Exporting environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
    echo 'source /etc/rp_environment' >> ~/.bashrc
}

# Start jupyter lab
start_jupyter() {
    if [[ $JUPYTER_PASSWORD ]]; then
        echo "Starting Jupyter Lab..."
        mkdir -p /workspace && \
        cd / && \
        nohup jupyter lab --allow-root \
          --no-browser \
          --port=8888 \
          --ip=* \
          --FileContentsManager.delete_to_trash=False \
          --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
          --ServerApp.token=${JUPYTER_PASSWORD} \
          --ServerApp.allow_origin=* \
          --ServerApp.preferred_dir=/workspace &> /workspace/logs/jupyter.log &
        echo "Jupyter Lab started"
    fi
}

# ---------------------------------------------------------------------------- #
#                               Main Program                                   #
# ---------------------------------------------------------------------------- #

start_nginx

execute_script "./pre_start.sh" "Running pre-start script..." "${@}"

echo "Pod Started"

setup_ssh
start_jupyter
export_env_vars

execute_script "/post_start.sh" "Running post-start script..." "${@}"

echo "Container is READY!"

sleep infinity