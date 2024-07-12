function build() { echo \"Installing dependencies...\";rosdep install --from-paths /workspaces/ros_ws/src --ignore-src -ry;echo \"Calling catkin build...\";catkin build;echo \"Sourcing devel/setup.bash...\";source /workspaces/ros_ws/devel/setup.bash;echo \"Sourced.\"; }

rm -rf build devel logs
rosdep install --from-paths /workspaces/ros_ws/src --ignore-src -ry
pip install -r src/ultralytics_ros/requirements.txt && pip install -r src/vgn/requirements.txt & build
pip install --force-reinstall -v "numpy==1.23.5" # sorry for this hack :( - xavier