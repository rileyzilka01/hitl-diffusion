Make sure im using the correct data everywhere: point cloud, joint position, ee position, ee orientation
Shouldnt need to change the dicts anywhere can just use whats there
 - action would be ee orientation, the rest would be their own inputs
 - agent_state is the ee position and joint position combined
 - point_cloud is the point cloud

1. Install the dependencies and create environments
2. Try and train on the real robot data
3. Visualize point clouds and how their data is formatted
4. Collect real data
5. Try and train on real data
