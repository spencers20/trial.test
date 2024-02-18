#include <iostream>
#include <signal.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/ros.h>
#include <string>
#include "utils/kitti_loader.hpp"
#include "travel/node.h"

using PointType = pcl::PointXYZI;
using namespace std;

ros::Publisher NodePublisher;

string data_dir;
string seq;

void callbackSignalHandler(int signum) {
    cout << "Caught Ctrl + c " << endl;
    // Terminate program
    exit(signum);
}

template<typename T>
sensor_msgs::PointCloud2 cloud2msg(pcl::PointCloud<T> cloud, std::string frame_id = "map") {
    sensor_msgs::PointCloud2 cloud_ROS;
    pcl::toROSMsg(cloud, cloud_ROS);
    cloud_ROS.header.frame_id = frame_id;
    return cloud_ROS;
}

void Segmentation(pcl::PointCloud<PointType>::Ptr cloud_in) {
    // Preprocessing
    // Downsample the point cloud
    pcl::VoxelGrid<PointType> vg;
    vg.setInputCloud(cloud_in);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);
    pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);
    vg.filter(*cloud_filtered);

    // Algorithm Selection & Parameter Tuning
    // Use region growing segmentation with custom parameters
    pcl::RegionGrowing<PointType> rg;
    rg.setInputCloud(cloud_filtered);
    rg.setMinClusterSize(100);
    rg.setMaxClusterSize(100000);
    rg.setNumberOfNeighbours(30);
    rg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
    rg.setCurvatureThreshold(1.0);

    // Perform segmentation
    std::vector<pcl::PointIndices> clusters;
    rg.extract(clusters);

    // Feature Extraction
    // Compute normals
    pcl::NormalEstimation<PointType, pcl::Normal> ne;
    ne.setInputCloud(cloud_filtered);
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    ne.setSearchMethod(tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    ne.setKSearch(50);
    ne.compute(*cloud_normals);

    // Post-processing
    // Remove outliers using statistical outlier removal
    pcl::StatisticalOutlierRemoval<PointType> sor;
    sor.setInputCloud(cloud_filtered);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud_filtered);
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "Ros-Kitti-Publisher");

    // Initialize ROS node handle and retrieve parameters
    ros::NodeHandle nh;
    int kitti_hz;
    std::string node_topic;
    nh.param<string>("/node_topic", node_topic, "/node");
    nh.param<string>("/data_dir", data_dir, "/");
    nh.param<string>("/seq", seq, "");
    nh.param<int>("/kitti_hz", kitti_hz, 10);

    cout << "\033[1;32m" << "Node topic: " << node_topic << "\033[0m" << endl;
    cout << "\033[1;32m" << "KITTI data directory: " << data_dir << "\033[0m" << endl;
    cout << "\033[1;32m" << "Sequence: " << seq << "\033[0m" << endl;

    // Set up ROS publisher
    NodePublisher = nh.advertise<travel::node>(node_topic, 100, true);

    // Load KITTI dataset
    std::string data_path = data_dir + "/" + seq;
    KittiLoader loader(data_path);
    int N = loader.size();

    // Register signal handler
    signal(SIGINT, callbackSignalHandler);

    cout << "\033[1;32m[Kitti Publisher] Total " << N << " clouds are loaded\033[0m" << endl;
    ros::Rate r(kitti_hz);
    for (int n = 0; n < N; ++n) {
        cout << n << "th node is published!" << endl;

        // Load current point cloud from dataset
        pcl::PointCloud<PointType>::Ptr pc_curr(new pcl::PointCloud<PointType>);
        *pc_curr = *loader.cloud(n);
        cout << "Complete load!" << endl;

        // Implement segmentation improvements
        Segmentation(pc_curr);

        // Convert point cloud to ROS message
        travel::node node;
        node.lidar = cloud2msg(*pc_curr);
        node.header = node.lidar.header;
        node.header.seq = n;

        // Publish ROS message
        NodePublisher.publish(node);

        // Sleep to control publishing frequency
        r.sleep();
    }

    return 0;
}
