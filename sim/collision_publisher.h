#ifndef COLLISION_PUBLISHER_H
#define COLLISION_PUBLISHER_H

#include "ros/ros.h"
#include <string>
#include <vdrilling_msgs/points.h>
#include <vdrilling_msgs/UInt8Stamped.h>
#include <vdrilling_msgs/VolumeProp.h>
//Added
//include <vdrilling_msgs/force.h>
#include <geometry_msgs/WrenchStamped.h>

class DrillingPublisher{
public:
    DrillingPublisher(std::string a_namespace, std::string a_plugin);
    ~DrillingPublisher();
    void init(std::string a_namespace, std::string a_plugin);
    ros::NodeHandle* m_rosNode;

    void voxelsRemoved(double ray[3], float vcolor[4], double time);
    void burrChange(int burrSize, double time);
    //Added
    void force(double f[3], double time);
    void volumeProp(float dimensions[3], int voxelCount[3]);
private:
    ros::Publisher m_voxelsRemovedPub;
    ros::Publisher m_burrChangePub;
    ros::Publisher m_volumePropPub;
    //Added
    ros::Publisher m_forcePub;
    vdrilling_msgs::points voxel_msg;
    vdrilling_msgs::UInt8Stamped burr_msg;
    vdrilling_msgs::VolumeProp volume_msg;
    geometry_msgs::WrenchStamped force_msg;
};

#endif //VOLUMETRIC_PLUGIN_COLLISION_PUBLISHER_H
