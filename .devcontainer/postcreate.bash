function build() { echo \"Installing dependencies...\";rosdep install --from-paths /workspaces/ros_ws/src --ignore-src -ry;echo \"Calling catkin build...\";catkin build;echo \"Sourcing devel/setup.bash...\";source /workspaces/ros_ws/devel/setup.bash;echo \"Sourced.\"; }

rm -rf build devel logs
rosdep install --from-paths /workspaces/ros_ws/src --ignore-src -ry