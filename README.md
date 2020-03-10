# moro_g12
#
# Clone under catkin_ws src folder
#
# Run catkin build
#
# Change ekf_localization.launch last rows to:
# <!-- Launch EKF localization-->
# <node  pkg="moro_g12" type="ekf_localization.py" name="ekf_localization" output="screen" />
#
# Launch the visualization:
# roslaunch moro_localization ekf_localization.launch
