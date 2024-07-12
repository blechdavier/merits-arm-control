FROM osrf/ros:noetic-desktop-focal
# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=cvar
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************
RUN apt install python3-catkin-tools python3-osrf-pycommon python3-pip git -y
RUN apt update
RUN apt upgrade -y

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /workspaces/ros_ws/devel/setup.bash" >> ~/.bashrc
RUN echo "export PATH=\"/home/cvar/.local/bin:$PATH\"" >> ~/.bashrc
RUN echo "function build() { echo \"Installing dependencies...\";rosdep install --from-paths /workspaces/ros_ws/src --ignore-src -ry;echo \"Calling catkin build...\";catkin build;echo \"Sourcing devel/setup.bash...\";source /workspaces/ros_ws/devel/setup.bash;echo \"Sourced.\"; }" >> ~/.bashrc
RUN rosdep update

RUN pip install conan==1.59
RUN export PATH="/home/cvar/.local/bin:$PATH"
RUN /home/cvar/.local/bin/conan profile new default --detect > /dev/null
RUN /home/cvar/.local/bin/conan profile update settings.compiler.libcxx=libstdc++11 default
RUN /home/cvar/.local/bin/conan config set general.revisions_enabled=1