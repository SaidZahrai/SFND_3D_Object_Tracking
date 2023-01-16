#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_set>

// This is implementation of Euclidean Clustering usiong KD Tree from the Point Cloud course.
// It has been adapted to the present data structures.

// Structure to represent node of kd tree
template<typename PointT>
struct Node
{
	PointT point;
	int id;
	Node<PointT>* left;
	Node<PointT>* right;

	Node(PointT arr, int setId)
	:	point(arr), id(setId), left(NULL), right(NULL)
	{}
};

template<typename PointT>
struct KdTree
{
	Node<PointT>* root;

	KdTree()
	: root(NULL)
	{}

	void insert(PointT point, int id)
	{
	    // Completed during the course according to helps and instructions
		// the function should create a new node and place correctly with in the root 
		insertHelper(root, 0, point, id);

	}

	void insertHelper(Node<PointT> *&node, uint depth, PointT point, int id)
	{
		if(node == NULL)
		{
			node = new Node<PointT>(point, id);
		}
		else
		{
			std::vector<double> p {point.x, point.y, point.z};
			std::vector<double> n {(*node).point.x, (*node).point.y, (*node).point.z};
			uint cd = depth % 3;
			if (p[cd] < n[cd])
				insertHelper((*node).left, depth+1, point, id);
			else
				insertHelper((*node).right, depth+1, point, id);
		}
	}

	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(PointT target, std::vector<double> distanceTol)
	{
		std::vector<int> ids;
		searchHelper(target, root, 0, distanceTol, ids);
		return ids;
	}

	void searchHelper(PointT target, Node<PointT>* node, uint depth, std::vector<double> distanceTol, std::vector<int>& ids)
	{
		// Here, we search for points in an anisotrobic rectangular box around the point of interest.
		if (node != NULL)
		{
			// Third diention added:
			if ((node->point.x>=(target.x-distanceTol[0]))&&(node->point.x<=(target.x+distanceTol[0]))&&
				(node->point.y>=(target.y-distanceTol[1]))&&(node->point.y<=(target.y+distanceTol[1]))&&
				(node->point.z>=(target.z-distanceTol[2]))&&(node->point.z<=(target.z+distanceTol[2])))
			{
				ids.push_back(node->id);
			}
			std::vector<double> t {target.x, target.y, target.z};
			std::vector<double> n {(*node).point.x, (*node).point.y, (*node).point.z};
			if ((t[depth % 3]-distanceTol[depth % 3]) < n[depth % 3])
				searchHelper(target, node->left, depth+1, distanceTol, ids);
			if ((t[depth % 3]+distanceTol[depth % 3]) > n[depth % 3])
				searchHelper(target, node->right, depth+1, distanceTol, ids);
		}
	}
};

template<typename PointT>
struct EuclideanClustering
{
	std::vector<PointT> inputCloud;
	KdTree<PointT> tree;
	std::vector<bool> processed;

	EuclideanClustering(std::vector<PointT> cloud)
	: inputCloud(cloud), processed(std::vector<bool>(cloud.size(), false))
	{
		// for (int i = 0; i < cloud->size(); i++)
		// 	tree.insert(cloud->points[i],i);
		int index = 0;
        for (auto itr = cloud.begin(); itr != cloud.end(); itr++)
		{
            tree.insert(*itr, index);
			index++;
		}
	}

	// Euclidean Cluster - Implementation from the course
	void clusterHelper(int i, std::vector<int>& cluster, std::vector<double> distanceTol)
	{
		processed[i] = true;
		cluster.push_back(i);

		std::vector<int> nearest = tree.search(inputCloud[i], distanceTol);
		for (int id : nearest)
		{
			if (!processed[id])
				clusterHelper(id, cluster, distanceTol);
		}
	}

	std::vector<std::vector<int>> euclideanCluster(std::vector<double> distanceTol, int minSize, int maxSize)
	{
		std::vector<std::vector<int>> clusters;

		for (int i = 0; i < inputCloud.size(); i++)
		{
			if (processed[i])
				continue;

			std::vector<int> cluster;
			clusterHelper  (i, cluster, distanceTol);
			if ((cluster.size() > minSize) && (cluster.size()<maxSize))
				clusters.push_back(cluster);
		}

		return clusters;
	}
};
