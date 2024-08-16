SEARCH_DIST = 50.0
TRAJ_SEARCH_DIST = 20.0
MAX_CROSS_DIST = 10.0
MIN_INTERSECTION_POINTS = 5
INTERSECTION_CLASS_RADIUS_PLUS = 10.0
INTERSECTION_CLASS_ANGLE = 30.0
LANE_SEARCH_RADIUS = 20.0
MIN_MOVE_SPEED = 0.50
ACC_THRESHOLD = 0.50
LEFT_RIGHT_THRESHOLD = 2.00
WOMD_ROUTE = '/media/msc-auto/HDD/waymo_motion/tf_example/training'
import sys
import json
import os
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings

import pickle
import numpy as np
import itertools
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from sklearn.cluster import DBSCAN

from google.protobuf import text_format

def main(start_time = 0, des_period = 11, future_period = 30):
    original_stdout = sys.stdout
    num_map_samples = 30000

    roadgraph_features = {
            'scenario/id':
            tf.io.FixedLenFeature([], tf.string, default_value=None),
            'roadgraph_samples/dir': tf.io.FixedLenFeature(
                [num_map_samples, 3], tf.float32, default_value=None
            ),
            'roadgraph_samples/id': tf.io.FixedLenFeature(
                [num_map_samples, 1], tf.int64, default_value=None
            ),
            'roadgraph_samples/type': tf.io.FixedLenFeature(
                [num_map_samples, 1], tf.int64, default_value=None
            ),
            'roadgraph_samples/valid': tf.io.FixedLenFeature(
                [num_map_samples, 1], tf.int64, default_value=None
            ),
            'roadgraph_samples/xyz': tf.io.FixedLenFeature(
                [num_map_samples, 3], tf.float32, default_value=None
            ),
        }
    # Features of other agents.
    state_features = {
            'state/id':
                tf.io.FixedLenFeature([128], tf.float32, default_value=None),
            'state/type':
                tf.io.FixedLenFeature([128], tf.float32, default_value=None),
            'state/is_sdc':
                tf.io.FixedLenFeature([128], tf.int64, default_value=None),
            'state/tracks_to_predict':
                tf.io.FixedLenFeature([128], tf.int64, default_value=None),
            'state/current/bbox_yaw':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/height':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/length':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/timestamp_micros':
                tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
            'state/current/valid':
                tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
            'state/current/vel_yaw':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/velocity_x':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/velocity_y':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/width':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/x':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/y':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/z':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/future/bbox_yaw':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/height':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/length':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/timestamp_micros':
                tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
            'state/future/valid':
                tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
            'state/future/vel_yaw':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/velocity_x':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/velocity_y':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/width':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/x':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/y':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/z':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/past/bbox_yaw':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/height':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/length':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/timestamp_micros':
                tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
            'state/past/valid':
                tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
            'state/past/vel_yaw':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/velocity_x':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/velocity_y':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/width':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/x':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/y':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/z':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/speed':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/current/speed':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/future/speed':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/objects_of_interest':
                tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        }

    traffic_light_features = {
            'traffic_light_state/current/state':
                tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
            'traffic_light_state/current/valid':
                tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
            'traffic_light_state/current/x':
                tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
            'traffic_light_state/current/y':
                tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
            'traffic_light_state/current/z':
                tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
            'traffic_light_state/past/state':
                tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
            'traffic_light_state/past/valid':
                tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
            'traffic_light_state/past/x':
                tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
            'traffic_light_state/past/y':
                tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
            'traffic_light_state/past/z':
                tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
            'traffic_light_state/past/id':
                tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        }

    features_description = {}
    features_description.update(roadgraph_features)
    features_description.update(state_features)
    features_description.update(traffic_light_features)
    end_time = start_time + des_period + future_period - 1

    for file in range(150):
        file_num = str(file).zfill(5)
        filename = os.path.join(WOMD_ROUTE, 'training_tfexample.tfrecord-'+file_num+'-of-01000')
        
        # Part 1: Data Loading
        dataset = tf.data.TFRecordDataset(filename, compression_type='')
        np_dataset = dataset.as_numpy_iterator()
        output_num = 0
        while True:
            try:
                data = next(np_dataset)
                parsed = tf.io.parse_single_example(data, features_description)
            except StopIteration:
                break
            
            output_num += 1
            
            interaction = parsed['state/objects_of_interest']
            interaction = list(interaction.numpy())
            if 1 not in interaction:
                continue
            int_pair = []
            for i, inte_label in enumerate(interaction):
                if inte_label == 1:
                    int_pair.append(i)

            # We choose only those with Objects of Interest pairs
            if len(int_pair) < 2:
                continue
            int_pair_id = []
            for i in int_pair:
                int_pair_id.append(parsed['state/id'][i].numpy())

            
            # Step 1: Agent Finding
            interest = parsed['state/tracks_to_predict'].numpy()
            av = parsed['state/is_sdc'].numpy()
            # Temp We take only the 1st agent, and the agent must be in the object of interest list
            for i in range(interest.shape[0]):
                if av[i] == 1:
                    agent = i
                    agent_id = parsed['state/id'][agent].numpy()
                    break
            if agent_id not in int_pair_id:
                continue
            
            plt.figure(dpi=300)
            plt.xlim([-100,100])
            plt.ylim([-100,100])

            sys.stdout = open('./log/training-'+file_num+'-of-01000-'+str(output_num-1).zfill(3)+'.log', 'w')

            # Load the trajectory.
            traj_hist_length = parsed['state/past/x'][agent,:].numpy().shape[0]
            traj_cur_length = 1
            traj_future_length = parsed['state/future/x'][agent,:].numpy().shape[0]
            agent_hist_x = np.expand_dims(parsed['state/past/x'][agent,:].numpy(), axis=1)
            agent_hist_y = np.expand_dims(parsed['state/past/y'][agent,:].numpy(), axis=1)
            agent_hist_z = np.expand_dims(parsed['state/past/z'][agent,:].numpy(), axis=1)
            agent_cur_x = np.expand_dims(parsed['state/current/x'][agent,:].numpy(), axis=1)
            agent_cur_y = np.expand_dims(parsed['state/current/y'][agent,:].numpy(), axis=1)
            agent_cur_z = np.expand_dims(parsed['state/current/z'][agent,:].numpy(), axis=1)
            agent_fut_x = np.expand_dims(parsed['state/future/x'][agent,:].numpy(), axis=1)
            agent_fut_y = np.expand_dims(parsed['state/future/y'][agent,:].numpy(), axis=1)
            agent_fut_z = np.expand_dims(parsed['state/future/z'][agent,:].numpy(), axis=1)
            agent_hist_traj = np.concatenate((agent_hist_x, agent_hist_y), axis=1)
            agent_cur_traj = np.concatenate((agent_cur_x, agent_cur_y), axis=1)
            agent_fut_traj = np.concatenate((agent_fut_x, agent_fut_y), axis=1)
            agent_traj = np.concatenate((agent_hist_traj, agent_cur_traj, agent_fut_traj), axis=0)
            center_2D = agent_traj[[start_time + des_period],:]
            agent_traj = agent_traj - center_2D
            agent_z = np.concatenate((agent_hist_z, agent_cur_z, agent_fut_z), axis=0) 
            center_z = agent_z[start_time + des_period]
            agent_z = agent_z - center_z
            traj_length = traj_hist_length + traj_cur_length + traj_future_length
            # print('Agent Traj =', agent_traj)
            # print('Agent Z =', agent_z)
            # print('Length =', agent_traj.shape[0], agent_z.shape[0], traj_length)

            # Load the map.
            road_xyz = parsed['roadgraph_samples/xyz'].numpy()
            road_length = road_xyz.shape[0]
            road_3Ddir = parsed['roadgraph_samples/dir'].numpy()
            center_2D = (parsed['state/current/x'][agent,0].numpy(), parsed['state/current/y'][agent,0].numpy())
            center_z = parsed['state/current/z'][agent,0].numpy()
            road_raw_xy = road_xyz[:,:2]
            road_raw_z = road_xyz[:,2]
            road_dir = road_3Ddir[:,:2]
            road_xy = road_raw_xy - np.tile(np.expand_dims(center_2D,axis=0), (road_length,1))
            road_z = road_raw_z - center_z
            road_type = parsed['roadgraph_samples/type'].numpy()
            road_ids = parsed['roadgraph_samples/id'][:,0].numpy()
            max_id = max(road_ids)

            # agent_vel
            agent_hist_vx = np.expand_dims(parsed['state/past/velocity_x'][agent,:].numpy(), axis=1)
            agent_hist_vy = np.expand_dims(parsed['state/past/velocity_y'][agent,:].numpy(), axis=1)
            agent_hist_vel = np.concatenate((agent_hist_vx, agent_hist_vy), axis=1)
            agent_cur_vx = np.expand_dims(parsed['state/current/velocity_x'][agent,:].numpy(), axis=1)
            agent_cur_vy = np.expand_dims(parsed['state/current/velocity_y'][agent,:].numpy(), axis=1)
            agent_cur_vel = np.concatenate((agent_cur_vx, agent_cur_vy), axis=1)
            agent_fut_vx = np.expand_dims(parsed['state/future/velocity_x'][agent,:].numpy(), axis=1)
            agent_fut_vy = np.expand_dims(parsed['state/future/velocity_y'][agent,:].numpy(), axis=1)
            agent_fut_vel = np.concatenate((agent_fut_vx, agent_fut_vy), axis=1)
            agent_vel = np.concatenate((agent_hist_vel, agent_cur_vel, agent_fut_vel), axis=0)
            # print('Agent Vel =', agent_vel)
            # print('Agent Vel =', agent_vel.shape)

            agent_hist_heading_angle = parsed['state/past/bbox_yaw'][agent,:].numpy()
            agent_cur_heading_angle = parsed['state/current/bbox_yaw'][agent,0].numpy()
            agent_fut_heading_angle = parsed['state/future/bbox_yaw'][agent,:].numpy()
            agent_heading_angle = np.concatenate((agent_hist_heading_angle, [agent_cur_heading_angle], agent_fut_heading_angle), axis=0)
            agent_heading_angle = np.expand_dims(agent_heading_angle, axis=1)
            agent_heading_dir = np.concatenate((np.cos(agent_heading_angle), np.sin(agent_heading_angle)), axis=1)
            # print(agent_heading_angle.shape)
            # print(agent_heading_dir.shape)

            all_class = parsed['state/type'].numpy()
            all_id = parsed['state/id'].numpy()
            all_past_valid = parsed['state/past/valid'].numpy()
            all_cur_valid = parsed['state/current/valid'].numpy()
            all_future_valid = parsed['state/future/valid'].numpy()
            all_valid = np.concatenate((all_past_valid, all_cur_valid, all_future_valid), axis=1)

            all_hist_x = np.expand_dims(parsed['state/past/x'][:,:].numpy(), axis = 2)
            all_hist_y = np.expand_dims(parsed['state/past/y'][:,:].numpy(), axis = 2)
            all_cur_x = np.expand_dims(parsed['state/current/x'][:,:].numpy(), axis = 2)
            all_cur_y = np.expand_dims(parsed['state/current/y'][:,:].numpy(), axis = 2)
            all_fut_x = np.expand_dims(parsed['state/future/x'][:,:].numpy(), axis = 2)
            all_fut_y = np.expand_dims(parsed['state/future/y'][:,:].numpy(), axis = 2)
            traj_num = all_hist_x.shape[0]
            all_hist_traj = np.concatenate((all_hist_x, all_hist_y), axis=2) - np.tile(np.expand_dims(np.expand_dims(center_2D,axis=0), axis=0), (traj_num, traj_hist_length,1)) # [traj_num, traj_hist_length, 2]
            all_cur_traj = np.concatenate((all_cur_x, all_cur_y), axis=2) - np.tile(np.expand_dims(np.expand_dims(center_2D,axis=0), axis=0), (traj_num, traj_cur_length,1)) # [traj_num, 1, 2]
            all_fut_traj = np.concatenate((all_fut_x, all_fut_y), axis=2) - np.tile(np.expand_dims(np.expand_dims(center_2D,axis=0), axis=0), (traj_num, traj_future_length,1)) # [traj_num, traj_future_length, 2]
            all_traj = np.concatenate((all_hist_traj, all_cur_traj, all_fut_traj), axis=1) # [traj_num, traj_length, 2]
            traj_length = all_traj.shape[1]
            time_length = traj_length

            all_hist_z = parsed['state/past/z'].numpy() - center_z
            all_cur_z = parsed['state/current/z'].numpy() - center_z
            all_fut_z = parsed['state/future/z'].numpy() - center_z
            all_z = np.concatenate((all_hist_z, all_cur_z, all_fut_z), axis=1)
            
            all_hist_speed = parsed['state/past/speed'].numpy()
            all_cur_speed = parsed['state/current/speed'].numpy()
            all_fut_speed = parsed['state/future/speed'].numpy()
            all_speed = np.concatenate((all_hist_speed, all_cur_speed, all_fut_speed), axis=1)

            all_hist_vx = np.expand_dims(parsed['state/past/velocity_x'][:,:].numpy(), axis=2)
            all_hist_vy = np.expand_dims(parsed['state/past/velocity_y'][:,:].numpy(), axis=2)
            all_hist_vel = np.concatenate((all_hist_vx, all_hist_vy), axis=2)
            all_cur_vx = np.expand_dims(parsed['state/current/velocity_x'][:,:].numpy(), axis=2)
            all_cur_vy = np.expand_dims(parsed['state/current/velocity_y'][:,:].numpy(), axis=2)
            all_cur_vel = np.concatenate((all_cur_vx, all_cur_vy), axis=2)
            all_fut_vx = np.expand_dims(parsed['state/future/velocity_x'][:,:].numpy(), axis=2)
            all_fut_vy = np.expand_dims(parsed['state/future/velocity_y'][:,:].numpy(), axis=2)
            all_fut_vel = np.concatenate((all_fut_vx, all_fut_vy), axis=2)
            all_vel = np.concatenate((all_hist_vel, all_cur_vel, all_fut_vel), axis=1)
            
            related_agent = []
            for i in range(traj_num):
                for time in range(start_time, start_time + des_period + future_period):
                    if np.linalg.norm(all_traj[i,time] - agent_traj[time]) < TRAJ_SEARCH_DIST and (i not in related_agent) and all_valid[i,time] == 1 and all_valid[i,start_time+des_period] == 1:
                        related_agent.append(i)
                        break

            rel_traj = all_traj[related_agent,:,:]
            rel_valid = all_valid[related_agent,:]
            rel_class = all_class[related_agent]
            rel_speed = all_speed[related_agent,:]
            rel_vel = all_vel[related_agent,:,:]
            rel_z = all_z[related_agent,:]
            rel_id = parsed['state/id'].numpy()[related_agent].astype(int)
            if int_pair_id[0] not in rel_id and int_pair_id[1] not in rel_id:
                continue
            all_hist_heading_angle = parsed['state/past/bbox_yaw'].numpy()
            all_cur_heading_angle = parsed['state/current/bbox_yaw'].numpy()
            all_fut_heading_angle = parsed['state/future/bbox_yaw'].numpy()
            all_heading_angle = np.concatenate((all_hist_heading_angle, all_cur_heading_angle, all_fut_heading_angle), axis=1)
            rel_heading_angle = np.expand_dims(all_heading_angle[related_agent], axis=-1)
            rel_heading_dir = np.concatenate((np.cos(rel_heading_angle), np.sin(rel_heading_angle)), axis=2)
            veh_num = 0
            bik_num = 100
            ped_num = 200
            other_num = 300
            rel_qa_id = []
            for i in range(rel_traj.shape[0]):
                if rel_class[i] == 1:
                    rel_qa_id.append(veh_num)
                    veh_num += 1
                elif rel_class[i] == 2:
                    rel_qa_id.append(ped_num)
                    ped_num += 1
                elif rel_class[i] == 3:
                    rel_qa_id.append(bik_num)
                    bik_num += 1
                else:
                    rel_qa_id.append(other_num)
                    other_num += 1


            # Step 2: Area Searching
            # 2.1 Intersection Point Searching
            intersection = []
            road_dist = np.zeros_like(road_ids)
            for i in range(road_length):
                road_dist[i] = max(np.abs(road_xy[i,0]), np.abs(road_xy[i,1]))
            for i in range(1,road_length-1):
                if road_type[i] > 2 or road_dist[i] > SEARCH_DIST:
                    continue
                if road_ids[i] == road_ids[i+1] and road_ids[i] == road_ids[i-1]:
                    for j in range(i+1,road_length-1):
                        if road_type[j] > 2 or road_dist[i] > SEARCH_DIST:
                            # Exclude none-highway or too far lanes
                            continue    
                        if np.abs(road_z[i] - road_z[j]) > 1.0:
                            continue
                        if np.dot(road_dir[i], road_dir[j]) > 0.95:
                            # Exclude same-direction lanes
                            continue
                        if road_ids[j] == road_ids[j+1] and road_ids[j] == road_ids[j-1] and road_ids[i] != road_ids[j]:
                            if max(np.abs(road_xy[i,0] - road_xy[j,0]), np.abs(road_xy[i,1] - road_xy[j,1])) < 0.50 :
                                intersection.append([i,j])
            if intersection == []:
                print('No intersection involved in this scene.')
                no_int_label = True
                int_center_0 = [0.0, 0.0]
                num_intersection = 0
            else:
                no_int_label = False
                intersection = np.array(intersection)
                num_intersection = intersection.shape[0]
                intersection_xy = road_xy[intersection[:,0],:]

                # 2.2 Intersection Searching
                # print(num_intersection)
                clustering = DBSCAN(eps=MAX_CROSS_DIST, min_samples=MIN_INTERSECTION_POINTS).fit(intersection_xy)

                if np.max(clustering.labels_) == -1:
                    no_int_label = True
                    print('No intersection involved in this scene.')
                
                num_int = 1
                if road_xy[intersection[clustering.labels_ == 0,0],:].shape[0] > 0:
                    int_center_0 = np.mean(road_xy[intersection[clustering.labels_ == 0,0],:], axis=0)
                else:
                    int_center_0 = [9999.0, 9999.0]
                if road_xy[intersection[clustering.labels_ == 1,0],:].shape[0] > 0:
                    int_center_1 = np.mean(road_xy[intersection[clustering.labels_ == 1,0],:], axis=0)
                    num_int += 1
                if road_xy[intersection[clustering.labels_ == 2,0],:].shape[0] > 0:
                    int_center_2 = np.mean(road_xy[intersection[clustering.labels_ == 2,0],:], axis=0)
                    num_int += 1

                
                pair_centers = []
                for car_index, car in enumerate(all_id):
                    if car == int_pair_id[1] or car == int_pair_id[0]:
                        pair_centers.append(car_index)
                interaction_center = [sum(all_cur_traj[pair_centers,0,0]) / len(pair_centers), sum(all_cur_traj[pair_centers,0,1]) / len(pair_centers)]


                dis_int = np.zeros(num_int)
                dis_int[0] = np.linalg.norm([a - b for a, b in zip(int_center_0, interaction_center)])
                if num_int > 1:
                    dis_int[1] = np.linalg.norm([a - b for a, b in zip(int_center_1, interaction_center)])
                if num_int > 2:
                    dis_int[2] = np.linalg.norm([a - b for a, b in zip(int_center_2, interaction_center)])
                target_int = np.argmin(dis_int)
                if target_int == 1:
                    int_center_0 = int_center_1
                if target_int == 2:
                    int_center_0 = int_center_2

            int_center_0 = np.array(int_center_0)
            int_radius = 0.0
            for i in range(num_intersection):
                if clustering.labels_[i] == target_int:
                    cur_radius = np.linalg.norm(road_xy[intersection[i,0]] - int_center_0)
                    if int_radius < cur_radius:
                        int_radius = cur_radius
            circle = patches.Circle((int_center_0[0], int_center_0[1]), int_radius, fc='blue', alpha=0.10, ec='blue')  
            plt.gca().add_patch(circle)

            in_int_mark = False
            for i in range(start_time, start_time + des_period + future_period):
                if np.linalg.norm(agent_traj[i] - int_center_0) < int_radius:
                    in_int_mark = True
                    break
            if no_int_label:
                pass
            else:
                int_agent_front = np.float32(np.dot(int_center_0, agent_heading_dir[start_time + des_period]))
                int_agent_side = np.float32(np.cross(int_center_0, agent_heading_dir[start_time + des_period]))
                if in_int_mark:
                    print('The ego agent is in intersection.', end=' ')
                elif int_agent_front > 0:
                    print('The ego agent is heading towards intersection.', end=' ')
                else:
                    print('The ego agent is departing from intersection.', end=' ')

                if np.abs(int_agent_front) > np.abs(int_agent_side):
                    if int_agent_front > 0:
                        print('The intersection center is', round(int_agent_front,0), 'meters in front of the ego agent,', end=' ')
                    else:
                        print('The intersection center is', round(-int_agent_front,0), 'meters behind the ego agent,', end=' ')
                    if int_agent_side > 0:
                        print('and is', round(int_agent_side,0), 'meters on the right of the ego agent.', end=' ')
                    else:
                        print('and is', round(-int_agent_side,0) , 'meters on the left of the ego agent.', end=' ')
                else:
                    if int_agent_side > 0:
                        print('The intersection center is', round(int_agent_side,0), 'meters on the right of the ego agent,', end=' ')
                    else:
                        print('The intersection center is', round(-int_agent_side,0) , 'meters on the left of the ego agent,', end=' ')
                    if int_agent_front > 0:
                        print('and is', round(int_agent_front,0), 'meters in front of the ego agent.', end=' ')
                    else:
                        print('and is', round(-int_agent_front,0), 'meters behind the ego agent.', end=' ')



            # 2.3 Intersection Classification
            edge_0 = []
            int_class_radius = int_radius + INTERSECTION_CLASS_RADIUS_PLUS
            for i in range(road_length):
                if road_type[i] > 2 or np.linalg.norm(road_dir[i]) < 0.10:
                    continue
                road_intersection_dist = np.linalg.norm(road_xy[i] - int_center_0)
                if road_intersection_dist < int_class_radius and road_intersection_dist > int_class_radius - 0.50:
                        if np.abs(np.dot(road_xy[i]-int_center_0, road_dir[i])) > 0.50:
                            edge_0.append([road_xy[i,0]-int_center_0[0],road_xy[i,1]-int_center_0[1]])
                            
            edge_0 = np.array(edge_0)
            threshold = 2.0 * int_class_radius * np.pi / 360.0 * INTERSECTION_CLASS_ANGLE

            if not no_int_label and edge_0.reshape(-1,2).shape[0] > 1:
                edge_0_clustering = DBSCAN(eps=threshold, min_samples=1).fit(edge_0)
                int_0_style = np.max(edge_0_clustering.labels_)
                if int_0_style > 1:
                    print('The intersection is a', int_0_style + 1, 'way intersection.', end=' ')
                else:
                    print('The intersection is an entry or exit intersection.', end=' ')
            print('\n', end='')


            # Step 3: Find Agent Involved Lanes
            # 3.1/2 Agent Involved Lanes Searching
            lane_stat = np.zeros(max_id+1)
            for j in range(start_time, start_time + des_period):
                for i in range(road_length):
                    if road_type[i] > 2 or np.abs(road_z[i]) > 3.0:
                        continue
                    if np.dot(road_dir[i], agent_heading_dir[j]) < 0.0:
                        continue
                    cur_error_sqr = np.dot(agent_traj[j] - road_xy[i], agent_traj[j] - road_xy[i])
                    lane_stat[road_ids[i]] += 1.0 / (cur_error_sqr + 0.001)
            chosen_id = np.argmax(lane_stat)
            chosen_lane = road_xy[road_ids == chosen_id,:]
            chosen_dir = road_dir[road_ids == chosen_id,:]

            # 3.3 Lane Position / Intersection Position
            if in_int_mark:
                chosen_lane_final_dir = chosen_dir[-2]
                chosen_lane_init_dir = chosen_dir[1]
                chosen_lane_angle_sin = np.cross(chosen_lane_final_dir, chosen_lane_init_dir)
                chosen_lane_angle_cos = np.dot(chosen_lane_final_dir, chosen_lane_init_dir)
                if chosen_lane_angle_cos < -0.707:
                    print('It is making a U-turn.', end=' ')
                elif chosen_lane_angle_sin < -0.50:
                    print('It is turning left.', end=' ')
                elif chosen_lane_angle_sin > 0.50:
                    print('It is turning right.', end=' ')
                else:
                    print('It is going straight.', end=' ')

                if np.dot(agent_vel[-1,:], int_center_0 - agent_traj[-1,:]) > 0.0:
                    print('It is entering the intersection.', end=' ')
                else:
                    print('It is exiting the intersection.', end=' ')
            else:
                tot_lane = []
                right_lane = []
                lane_dist_set = []
                dist = 999999.0
                chosen_point = 0
                for i in range(1,chosen_lane.shape[0]-1):
                    cur_dist = np.dot(chosen_lane[i] - agent_traj[start_time + des_period], chosen_lane[i] - agent_traj[start_time + des_period])
                    if cur_dist < dist:
                        dist = cur_dist
                        chosen_point = i
                for i in range(road_length):
                    if road_type[i] > 2:
                        continue
                    if np.dot(road_dir[i], chosen_dir[chosen_point]) < 0.75:
                        continue
                    if np.abs(np.dot(road_xy[i] - chosen_lane[chosen_point], chosen_dir[chosen_point])) > 2.00:
                        continue
                    if road_ids[i] == chosen_id:
                        continue
                    if np.dot(road_xy[i] - chosen_lane[chosen_point], road_xy[i] - chosen_lane[chosen_point]) < LANE_SEARCH_RADIUS ** 2:
                        cross = np.int64(np.cross(chosen_lane[chosen_point] - road_xy[i], chosen_dir[chosen_point]))
                        if cross not in lane_dist_set and np.abs(cross) > 1:
                            lane_dist_set.append(cross)
                            if road_ids[i] not in tot_lane:
                                tot_lane.append(road_ids[i])
                                if np.cross(chosen_lane[chosen_point] - road_xy[i], chosen_dir[chosen_point]) < 0:
                                    right_lane.append(road_ids[i])
                if len(tot_lane) == 0:
                    print('The ego agent is on the only lane in this direction.', end=' ')
                elif len(right_lane) < 0.50 * len(tot_lane): 
                    print('The ego agent is on the', len(right_lane) + 1, 'lane from the right, out of', len(tot_lane) + 1, 'lanes.', end=' ')
                else:
                    print('The ego agent is on the', len(tot_lane) - len(right_lane) + 1, 'lane from the left, out of', len(tot_lane) + 1, 'lanes.', end=' ')

            # 3.4 Vel / Acc
            agent_max_vel = 0.0
            init_vel = np.sqrt(agent_vel[0,0] ** 2 + agent_vel[0,1] ** 2)
            cur_vel = np.sqrt(agent_vel[-1,0] ** 2 + agent_vel[-1,1] ** 2)
            formatted_vel = "{:.0f}".format(cur_vel)
            print('Its current speed is', formatted_vel, 'm/s.', end=' ')
            for i in range(traj_length):
                vel = np.sqrt(agent_vel[i,0] ** 2 + agent_vel[i,1] ** 2)
                if vel > agent_max_vel:
                    agent_max_vel = vel
            if agent_max_vel < MIN_MOVE_SPEED:
                print('It is not moving.', end=' ')
            elif init_vel - cur_vel < - ACC_THRESHOLD:
                print('It is accelerating.', end=' ')
            elif init_vel - cur_vel > ACC_THRESHOLD:
                print('It is decelerating.', end=' ')
            else:
                print('It is moving at a constant speed.', end=' ')

            # Step 6: Traffic Light Searching
            agent_cur_heading_dir = agent_heading_dir[start_time+des_period]
            traffic_past = parsed['traffic_light_state/past/state'].numpy()
            traffic_past_valid = parsed['traffic_light_state/past/valid'].numpy()
            traffic_lane = parsed['traffic_light_state/past/id'].numpy()
            traffic_light_num = traffic_past.shape[1]
            for j in range(road_length):
                if road_ids[j] in traffic_lane[-1,:]:
                    lane_list = list(traffic_lane[-1])
                    k = lane_list.index(road_ids[j])
                    if traffic_past[-1, k] == 1 or traffic_past[-1, k] == 4 or traffic_past[-1, k] == 7:
                        plt.quiver(road_xy[j,0],road_xy[j,1], road_dir[j,0], road_dir[j,1], angles='xy', scale_units='xy', scale=1, color='red', alpha=0.50)
                    elif traffic_past[-1, k] == 2 or traffic_past[-1, k] == 5 or traffic_past[-1, k] == 8:
                        plt.quiver(road_xy[j,0],road_xy[j,1], road_dir[j,0], road_dir[j,1], angles='xy', scale_units='xy', scale=1, color='yellow', alpha=0.50)
                    elif traffic_past[-1, k] == 3 or traffic_past[-1, k] == 6:
                        plt.quiver(road_xy[j,0],road_xy[j,1], road_dir[j,0], road_dir[j,1], angles='xy', scale_units='xy', scale=1, color='green', alpha=0.50)
            
            traffic_provided = False
            for j in range(traffic_light_num):
                for i in range(10):
                    if traffic_past_valid[-i,j] < 1:
                        continue
                    if traffic_lane[-i,j] == chosen_id:
                        if traffic_past[-i,j] == 1:
                            print('Traffic Light for the ego agent is red arrow.', end=' ')
                        elif traffic_past[-i,j] == 2:
                            print('Traffic Light for the ego agent is yellow arrow.', end=' ')
                        elif traffic_past[-i,j] == 3:
                            print('Traffic Light for the ego agent is green arrow.', end=' ')
                        elif traffic_past[-i,j] == 4:
                            print('Traffic Light for the ego agent is red.', end=' ')
                        elif traffic_past[-i,j] == 5:
                            print('Traffic Light for the ego agent is yellow.', end=' ')
                        elif traffic_past[-i,j] == 6:
                            print('Traffic Light for the ego agent is green.', end=' ')
                        elif traffic_past[-i,j] == 7:
                            print('Traffic Light for the ego agent is flashing red.', end=' ')
                        elif traffic_past[-i,j] == 8:
                            print('Traffic Light for the ego agent is flashing yellow.', end=' ')
                        traffic_provided = True
                        break

            if not traffic_provided:
                # Find the last point in the chosen lane
                chosen_lane_length = chosen_lane.shape[0]
                chosen_lane_index = []
                for length in range(chosen_lane_length):
                    chosen_lane_index.append(np.dot(chosen_lane[length] - chosen_lane[0], chosen_dir[0]))
                chosen_final_point = np.argmax(chosen_lane_index)
                chosen_final_xy = chosen_lane[chosen_final_point]

                

                traffic_provided = False
                for k in range(road_length):
                    if traffic_provided:
                        break
                    if road_type[k] > 2:
                        continue
                    if np.abs(np.dot(road_xy[k] - chosen_final_xy, chosen_dir[0])) > 3.00:
                        continue
                    if np.dot(road_dir[k], chosen_dir[0]) < 0.75:
                        continue
                    if np.linalg.norm(road_xy[k] - int_center_0) > int_radius + 20.0:
                        continue
                    if np.abs(np.cross(road_xy[k] - chosen_final_xy, chosen_dir[0])) < 1.00:
                        for j in range(traffic_light_num):
                            for i in range(10):
                                if traffic_past_valid[-i,j] < 1:
                                    continue
                                if traffic_lane[-i,j] == road_ids[k]:
                                    traffic_dist = max(0.0,np.dot(road_xy[k], agent_cur_heading_dir))
                                    if traffic_past[-i,j] == 1:
                                        print('Traffic Light for the ego agent is red arrow ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                    elif traffic_past[-i,j] == 2:
                                        print('Traffic Light for the ego agent is yellow arrow ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                    elif traffic_past[-i,j] == 3:
                                        print('Traffic Light for the ego agent is green arrow ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                    elif traffic_past[-i,j] == 4:
                                        print('Traffic Light for the ego agent is red ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                    elif traffic_past[-i,j] == 5:
                                        print('Traffic Light for the ego agent is yellow ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                    elif traffic_past[-i,j] == 6:
                                        print('Traffic Light for the ego agent is green ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                    elif traffic_past[-i,j] == 7:
                                        print('Traffic Light for the ego agent is flashing red ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                    elif traffic_past[-i,j] == 8:
                                        print('Traffic Light for the ego agent is flashing yellow ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                    traffic_provided = True
                                    break
                    


            # Step 7: Other Road Elements
            # StopSign = 17, Crosswalk = 18, SpeedBump = 19
            
            cur_cross = -1
            cur_bump = -1
            agent_cur_speed = np.linalg.norm(agent_vel[start_time + des_period])
            stop_sign_num = 0
            stop_sign_current = 0
            int_center_0 = np.array(int_center_0)
            cross_dist = 1000.0
            agent_cur_heading_dir = agent_heading_dir[start_time + des_period]
            for i in range(road_length):
                if road_type[i] == 17:
                    plt.scatter(road_xy[i,0], road_xy[i,1], marker=".", s=0.25, color='red')
                    if np.linalg.norm(road_xy[i] - int_center_0) < (int_radius + 10.0):
                        stop_sign_num += 1
                        if np.dot(road_xy[i], agent_cur_heading_dir) > - 15.0 and np.cross(road_xy[i], agent_cur_heading_dir) > 0.0 and np.dot(int_center_0, agent_cur_heading_dir) - np.dot(road_xy[i], agent_cur_heading_dir) > 0.0:
                            sign_ahead = np.dot(road_xy[i], agent_cur_heading_dir)
                            if sign_ahead > 0.0:
                                print('The ego agent is approaching a stop sign', "{:.0f}".format(np.dot(road_xy[i], agent_cur_heading_dir)), 'meters ahead.', end=' ')
                            else:
                                print('The ego agent is at a stop sign.', end=' ')
                            stop_sign_current = 1
                elif road_type[i] == 18:
                    plt.scatter(road_xy[i,0], road_xy[i,1], marker=".", s=0.05, color='blue')
                    if np.dot(road_xy[i], agent_cur_heading_dir) > -5.0 and np.dot(road_xy[i], agent_cur_heading_dir) < 20.0:
                        if np.dot(road_xy[i], agent_cur_heading_dir) > 0.966 * np.linalg.norm(road_xy[i]):
                            cross_dist = min(cross_dist, np.dot(road_xy[i], agent_cur_heading_dir))
                        
                elif road_type[i] == 19:
                    plt.scatter(road_xy[i,0], road_xy[i,1], marker=".", s=0.05, color='yellow')
                    if np.dot(road_xy[i], agent_cur_heading_dir) > -5.0 and np.dot(road_xy[i], agent_cur_heading_dir) < 20.0:
                        if np.dot(road_xy[i], agent_cur_heading_dir) > 0.966 * np.linalg.norm(road_xy[i]) and road_ids[i] != cur_bump:
                            dot_bump = np.dot(road_xy[i], agent_cur_heading_dir)
                            if dot_bump > 0:
                                print('The ego agent is approaching a speed bump', "{:.0f}".format(np.sqrt(np.dot(road_xy[i], agent_cur_heading_dir))), 'meters ahead.', end=' ')
                            else:
                                print('The ego agent is at a speed bump.', end=' ')
                            cur_bump = road_ids[i]

            
            if stop_sign_num > 0:
                print('There are', stop_sign_num, 'stop signs in the intersection.', end=' ')
                if stop_sign_current == 0:
                    print('But no stop sign is on the ego agent\'s side.', end=' ')
            
            if cross_dist < 50.0 and cross_dist > 0.0:
                print('The ego agent is approaching a crosswalk', "{:.0f}".format(np.sqrt(cross_dist)), 'meters ahead.', end=' ')
            elif cross_dist < 0.0:
                print('The ego agent is at a crosswalk.', end=' ')


            print('\n', end='')

            # Step 4: Related Agents Searching
            rel_traj_print = rel_traj[:,start_time+des_period:end_time]
            rel_valid_print = rel_valid[:,start_time+des_period:end_time]

            for i in range(rel_traj.shape[0]):
                if rel_id[i] == agent_id:
                    continue
                if rel_id[i] == int_pair_id[1] or rel_id[i] == int_pair_id[0]:
                    plt.text(rel_traj[i,end_time,0], rel_traj[i,end_time,1], str(np.int64(rel_qa_id[i])), color='black', fontsize=10)
                else:
                    plt.text(rel_traj[i,end_time,0], rel_traj[i,end_time,1], str(np.int64(rel_qa_id[i])), color='darkgreen', fontsize=10)
                if rel_class[i] == 1:
                    plt.plot(rel_traj_print[i,rel_valid_print[i,:] == 1,0], rel_traj_print[i,rel_valid_print[i,:] == 1,1], color='darkgreen', linewidth=2.0, alpha=1.0)
                    if rel_valid[i, end_time] == 1:
                        plt.scatter(rel_traj[i,end_time,0], rel_traj[i,end_time,1], marker="x", s=20.00, color='darkgreen')
                    print('Surrounding agent #', np.int64(rel_qa_id[i]), 'is a vehicle.', end=' ')
                elif rel_class[i] == 2:
                    plt.plot(rel_traj_print[i,rel_valid_print[i,:] == 1,0], rel_traj_print[i,rel_valid_print[i,:] == 1,1], color='darkgreen', linewidth=2.0, alpha=1.0)
                    if rel_valid[i, end_time] == 1:
                        plt.scatter(rel_traj[i,end_time,0], rel_traj[i,end_time,1], marker="x", s=20.00, color='darkgreen')
                    print('Surrounding agent #', np.int64(rel_qa_id[i]), 'is a pedestrian.', end=' ')
                elif rel_class[i] == 3:
                    plt.plot(rel_traj_print[i,rel_valid_print[i,:] == 1,0], rel_traj_print[i,rel_valid_print[i,:] == 1,1], color='darkgreen', linewidth=2.0, alpha=1.0)
                    if rel_valid[i, end_time] == 1:
                        plt.scatter(rel_traj[i,end_time,0], rel_traj[i,end_time,1], marker="x", s=20.00, color='darkgreen')
                    print('Surrounding agent #', np.int64(rel_qa_id[i]), 'is a cyclist.', end=' ')
                else:
                    assert False, 'Other Class Found!'
                rel_cur_heading_angle = rel_heading_angle[i,start_time+des_period]
                rel_cur_heading_dir = rel_heading_dir[i,start_time+des_period,:]
                rel_ego_front = np.dot(rel_traj[i,start_time+des_period], agent_heading_dir[start_time+des_period])
                rel_ego_side = np.cross(rel_traj[i,start_time+des_period], agent_heading_dir[start_time+des_period])
                related_ego_dir = (rel_cur_heading_angle - agent_cur_heading_angle) % (2 * np.pi)

                if np.abs(rel_ego_front) > np.abs(rel_ego_side):
                    if rel_ego_front > 0:
                        print('It is', "{:.0f}".format(rel_ego_front), 'meters in front of the ego agent,', end=' ')
                    else:
                        print('It is', "{:.0f}".format(-rel_ego_front), 'meters behind the ego agent,', end=' ')
                    
                    if rel_ego_side > LEFT_RIGHT_THRESHOLD:
                        print('and is', "{:.0f}".format(rel_ego_side), 'meters on the right of the ego agent.', end=' ')
                    elif rel_ego_side < -LEFT_RIGHT_THRESHOLD:
                        print('and is', "{:.0f}".format(-rel_ego_side), 'meters on the left of the ego agent.', end=' ')
                    else:
                        print('and is on the same lane as the ego agent.', end=' ')
                else:
                    if rel_ego_side > LEFT_RIGHT_THRESHOLD:
                        print('It is', "{:.0f}".format(rel_ego_side), 'meters on the right of the ego agent,', end=' ')
                    elif rel_ego_side < -LEFT_RIGHT_THRESHOLD:
                        print('It is', "{:.0f}".format(-rel_ego_side), 'meters on the left of the ego agent,', end=' ')
                    else:
                        print('It is on the same lane as the ego agent,', end=' ')

                    if rel_ego_front > 0:
                        print('and is', "{:.0f}".format(rel_ego_front), 'meters in front of the ego agent.', end=' ')
                    else:
                        print('and is', "{:.0f}".format(-rel_ego_front), 'meters behind the ego agent.', end=' ')

                if related_ego_dir < 0.16 * np.pi or related_ego_dir > 1.84 * np.pi:
                    print('It is heading in the same direction as the ego agent.', end=' ')
                elif related_ego_dir < 0.84 * np.pi:
                    print('It is heading left of the ego agent.', end=' ')
                elif related_ego_dir < 1.16 * np.pi:
                    print('It is heading the opposite direction as the ego agent.', end=' ')
                else:
                    print('It is heading right of the ego agent.', end=' ')

                # 4.2 Speed and Acc Check
                max_vel = 0.0
                init_speed = rel_speed[i,start_time]
                cur_speed = rel_speed[i,start_time+des_period]
                for k in range(start_time, start_time + des_period):
                    if rel_speed[i,k] > max_vel:
                        max_vel = rel_speed[i,k]

                if max_vel < MIN_MOVE_SPEED:
                    print('It is not moving.', end=' ')
                else:
                    print('Its current speed is', "{:.0f}".format(cur_speed), 'm/s.', end=' ')
                    if init_speed - cur_speed < - ACC_THRESHOLD:
                        print('It is accelerating.', end=' ')
                    elif init_speed - cur_speed > ACC_THRESHOLD:
                        print('It is decelerating.', end=' ')
                    else:
                        print('It is moving at a constant speed.', end=' ')

                # Step 5: Agent - Car Relation
                # 5.1 Relation to the Intersection
                agent_cur_pos = agent_traj[start_time+des_period,:]
                agent_cur_vel = agent_vel[start_time+des_period,:]
                rel_cur_pos = rel_traj[i,start_time+des_period,:]
                rel_cur_vel = rel_vel[i,start_time+des_period,:]
                rel_cur_z = rel_z[i,start_time+des_period]
                rel_in_int = False
                int_radius_for_related = int_radius - 5.0
                if no_int_label:
                    pass
                elif np.linalg.norm(rel_cur_pos - int_center_0) < int_radius_for_related:
                    print('It is in the intersection.', end=' ')
                    rel_in_int = True
                elif max_vel > 0.1:      
                    if np.dot(rel_cur_vel, rel_cur_pos - int_center_0) > 0:
                        print('It is departing from the intersection.', end=' ')
                    else:
                        print('It is heading towards the intersection.', end=' ')

                # 5.2 Relation to the Agent
                rel_int_dir = rel_cur_pos - int_center_0
                rel_int_dir = rel_int_dir / (np.linalg.norm(rel_int_dir) + 0.01)
            
                
                agent_int_dir = agent_cur_pos - int_center_0
                agent_int_dir = agent_int_dir / (np.linalg.norm(agent_int_dir) + 0.01)
                if no_int_label:
                    pass
                elif not rel_in_int:
                    if np.abs(np.dot(rel_int_dir, agent_int_dir)) > 0.707:
                        if np.dot(rel_int_dir, agent_int_dir) > 0.0:
                            print('It is on the same side of the intersection as the ego agent.', end=' ')
                        else:
                            print('It is at the opposite side of the intersection.', end=' ')
                    else:
                        if np.cross(rel_int_dir, agent_int_dir) > 0.0:
                            print('It is on the left of the intersection.', end=' ')
                        elif np.cross(rel_int_dir, agent_int_dir) < -0.0:
                            print('It is on the right of the intersection.', end=' ')
                       
                    # Lane Info for Related Agent
                    rel_lane_stat = np.zeros(max_id+1)
                    for j in range(start_time, start_time + des_period):
                        for k in range(road_length):
                            if road_type[k] > 2:
                                continue
                            if np.abs(road_z[k] - rel_cur_z) > 3.0:
                                continue
                            cur_error_sqr = np.linalg.norm(rel_traj[i,j] - road_xy[k]) ** 2
                            rel_lane_stat[road_ids[k]] += 1.0 / (cur_error_sqr + 0.001)
                    rel_chosen_id = np.argmax(rel_lane_stat)
                    rel_chosen_lane = road_xy[road_ids == rel_chosen_id,:]
                    rel_chosen_dir = road_dir[road_ids == rel_chosen_id,:]

                    # Related Agent Lane Position
                    tot_lane = []
                    right_lane = []
                    lane_dist_set = []
                    # Need to be changed to nearest point, not the start of the lane!
                    dist = 9999.0
                    chosen_point = 0
                    for k in range(1,rel_chosen_lane.shape[0]-1):
                        cur_dist = np.linalg.norm(rel_chosen_lane[k] - rel_traj[i,start_time+des_period]) ** 2
                        if cur_dist < dist:
                            dist = cur_dist
                            chosen_point = k
                    for k in range(road_length):
                        if road_type[k] > 2:
                            continue
                        if np.abs(np.dot(road_xy[k] - rel_chosen_lane[chosen_point], rel_chosen_dir[chosen_point])) > 2.00:
                            continue
                        if np.dot(road_dir[k], rel_chosen_dir[chosen_point]) < 0.75:
                            continue
                        if road_ids[k] == rel_chosen_id:
                            continue
                        if (road_xy[k,0] - rel_chosen_lane[chosen_point,0]) ** 2 + (road_xy[k,1] - rel_chosen_lane[chosen_point,1]) ** 2 < LANE_SEARCH_RADIUS ** 2:
                            cross = np.int64(np.cross(rel_chosen_lane[chosen_point] - road_xy[k], rel_chosen_dir[chosen_point]))
                            if cross not in lane_dist_set and np.abs(cross) > 1:
                                lane_dist_set.append(cross)
                                if road_ids[k] not in tot_lane:
                                    tot_lane.append(road_ids[k])
                                    if np.cross(rel_chosen_lane[chosen_point] - road_xy[k], rel_chosen_dir[chosen_point]) < 0:
                                        right_lane.append(road_ids[i])
                    traffic_provided = False
                    for j in range(traffic_light_num):
                        for k in range(10):
                            if traffic_past_valid[-k,j] < 1:
                                continue
                            if traffic_lane[-k,j] == rel_chosen_id:
                                if traffic_past[-k,j] == 1:
                                    print('Traffic Light for it is red arrow.', end=' ')
                                elif traffic_past[-k,j] == 2:
                                    print('Traffic Light for it is yellow arrow.', end=' ')
                                elif traffic_past[-k,j] == 3:
                                    print('Traffic Light for it is green arrow.', end=' ')
                                elif traffic_past[-k,j] == 4:
                                    print('Traffic Light for it is red.', end=' ')
                                elif traffic_past[-k,j] == 5:
                                    print('Traffic Light for it is yellow.', end=' ')
                                elif traffic_past[-k,j] == 6:
                                    print('Traffic Light for it is green.', end=' ')
                                elif traffic_past[-k,j] == 7:
                                    print('Traffic Light for it is flashing red.', end=' ')
                                elif traffic_past[-k,j] == 8:
                                    print('Traffic Light for it is flashing yellow.', end=' ')
                                traffic_provided = True
                                break
                    if not traffic_provided:
                        # Find the last point in the chosen lane
                        rel_chosen_lane_length = rel_chosen_lane.shape[0]
                        rel_chosen_lane_index = []
                        for length in range(rel_chosen_lane_length):
                            rel_chosen_lane_index.append(np.dot(rel_chosen_lane[length] - rel_chosen_lane[0], rel_chosen_dir[0]))
                        rel_chosen_final_point = np.argmax(rel_chosen_lane_index)
                        rel_chosen_final_xy = rel_chosen_lane[rel_chosen_final_point]

                        

                        traffic_provided = False
                        for k in range(road_length):
                            if traffic_provided:
                                break
                            if road_type[k] > 2:
                                continue
                            if np.linalg.norm(road_xy[k] - int_center_0) > int_radius + 20.0:
                                continue
                            if np.abs(np.dot(road_xy[k] - rel_chosen_final_xy, rel_chosen_dir[0])) > 3.00:
                                continue
                            if np.dot(road_dir[k], rel_chosen_dir[0]) < 0.75:
                                continue
                            if np.abs(np.cross(road_xy[k] - rel_chosen_final_xy, rel_chosen_dir[0])) < 1.00:
                                for j in range(traffic_light_num):
                                    for m in range(10):
                                        traffic_dist = max(0.0,np.dot(road_xy[k] - rel_cur_pos, rel_cur_heading_dir))
                                        if traffic_past_valid[-m,j] < 1:
                                            continue
                                        if traffic_lane[-m,j] == road_ids[k]:
                                            if traffic_past[-m,j] == 1:
                                                print('Traffic Light for it is red arrow ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                            elif traffic_past[-m,j] == 2:
                                                print('Traffic Light for it is yellow arrow ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                            elif traffic_past[-m,j] == 3:
                                                print('Traffic Light for it is green arrow ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                            elif traffic_past[-m,j] == 4:
                                                print('Traffic Light for it is red ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                            elif traffic_past[-m,j] == 5:
                                                print('Traffic Light for it is yellow ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                            elif traffic_past[-m,j] == 6:
                                                print('Traffic Light for it is green ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                            elif traffic_past[-m,j] == 7:
                                                print('Traffic Light for it is flashing red ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                            elif traffic_past[-m,j] == 8:
                                                print('Traffic Light for it is flashing yellow ' + str(round(traffic_dist,1)) + ' meters ahead.', end=' ')
                                            traffic_provided = True
                                            break

                else:
                    pass
                
                if (not rel_in_int) and (not no_int_label):
                    related_int_dis = np.linalg.norm(rel_cur_pos - int_center_0)
                    print('It is', "{:.0f}".format(related_int_dis), 'meters away from the intersection center.', end=' ')
                else:
                    rel_agent_dis = np.linalg.norm(rel_cur_pos)
                    print('It is', "{:.0f}".format(rel_agent_dis), 'meters away from the ego agent.', end=' ')
                
                cross_dist = 1000.0
                for k in range(road_length):
                    if road_type[k] == 17 and np.linalg.norm(road_xy[k] - int_center_0) < (int_radius + 10.0):
                        if np.dot(road_xy[k] - rel_cur_pos, rel_cur_heading_dir) > - 15.0 and np.cross(road_xy[k] - rel_cur_pos, rel_cur_heading_dir) > 0.0 and np.dot(int_center_0 - rel_cur_pos, rel_cur_heading_dir) - np.dot(road_xy[k] - rel_cur_pos, rel_cur_heading_dir) > 0.0:
                            sign_ahead = np.dot(road_xy[k] - rel_cur_pos, rel_cur_heading_dir)
                            if sign_ahead > 0.0:
                                print('It is approaching a stop sign', "{:.0f}".format(np.dot(road_xy[k] - rel_cur_pos, rel_cur_heading_dir)), 'meters ahead.', end=' ')
                            else:
                                print('It is at a stop sign.', end=' ')
                    elif road_type[k] == 18:
                        if np.dot(road_xy[k] - rel_cur_pos, rel_cur_heading_dir) > -5.0 and np.dot(road_xy[k] - rel_cur_pos, rel_cur_heading_dir) < 20.0:
                            if np.dot(road_xy[k] - rel_cur_pos, rel_cur_heading_dir) > 0.966 * np.linalg.norm(road_xy[k] - rel_cur_pos):
                                cross_dist = min(cross_dist, np.dot(road_xy[k] - rel_cur_pos, rel_cur_heading_dir))
                            
                    elif road_type[k] == 19:
                        if np.dot(road_xy[k] - rel_cur_pos, rel_cur_heading_dir) > -5.0 and np.dot(road_xy[k] - rel_cur_pos, rel_cur_heading_dir) < 20.0:
                            if np.dot(road_xy[k] - rel_cur_pos, rel_cur_heading_dir) > 0.966 * np.linalg.norm(road_xy[k] - rel_cur_pos) and road_ids[k] != cur_bump:
                                dot_bump = np.dot(road_xy[k] - rel_cur_pos, rel_cur_heading_dir)
                                if dot_bump > 0:
                                    print('The ego agent is approaching a speed bump', "{:.0f}".format(np.sqrt(np.dot(road_xy[k] - rel_cur_pos, rel_cur_heading_dir))), 'meters ahead.', end=' ')
                                else:
                                    print('The ego agent is at a speed bump.', end=' ')
                                cur_bump = road_ids[k]
                if cross_dist < 50.0 and cross_dist > 0.0:
                    print('It is approaching a crosswalk', "{:.0f}".format(np.sqrt(cross_dist)), 'meters ahead.', end=' ')
                elif cross_dist < 0.0:
                    print('It is at a crosswalk.', end=' ')
                print('\n', end='')
                    
            

            # Step -1: Future Description
            print('The following is the description of the ego and surrounding agents after {s} seconds:'.format(s = future_period / 10.0))
            # Step 4: Related Agents Searching
            
            for i in range(rel_traj.shape[0]):
                
                if rel_valid[i,end_time] == 0:
                    continue
                rel_cur_heading_angle = rel_heading_angle[i,end_time]
                rel_cur_heading_dir = rel_heading_dir[i,end_time,:]
                agent_cur_heading_angle = agent_heading_angle[end_time]
                rel_ego_front = np.dot(rel_traj[i,end_time]-agent_traj[end_time], agent_heading_dir[end_time])
                rel_ego_side = np.cross(rel_traj[i,end_time]-agent_traj[end_time], agent_heading_dir[end_time])
                related_ego_dir = (rel_cur_heading_angle - agent_cur_heading_angle) % (2 * np.pi)


                if rel_id[i] == agent_id:
                    rel_ego_front = np.dot(rel_traj[i,end_time], agent_heading_dir[start_time + des_period])
                    rel_ego_side = np.cross(rel_traj[i,end_time], agent_heading_dir[start_time + des_period])
                    if rel_ego_front > 0:
                        print('The ego agent will be', "{:.0f}".format(rel_ego_front), 'meters in front of the current place,', end=' ')
                    else:
                        print('The ego agent will be', "{:.0f}".format(-rel_ego_front), 'meters behind the current place,', end=' ')
                    
                    if rel_ego_side > LEFT_RIGHT_THRESHOLD:
                        print('and will be', "{:.0f}".format(rel_ego_side), 'meters on the right of the current place.', end=' ')
                    elif rel_ego_side < -LEFT_RIGHT_THRESHOLD:
                        print('and will be', "{:.0f}".format(-rel_ego_side), 'meters on the left of the current place.', end=' ')
                    else:
                        print('and will be on the same lane as the current place.', end=' ')
                elif np.abs(rel_ego_front) > np.abs(rel_ego_side):
                    if rel_ego_front > 0:
                        print('Surrounding agent #', np.int64(rel_qa_id[i]), 'will be', "{:.0f}".format(rel_ego_front), 'meters in front of the ego agent,', end=' ')
                    else:
                        print('Surrounding agent #', np.int64(rel_qa_id[i]), 'will be', "{:.0f}".format(-rel_ego_front), 'meters behind the ego agent,', end=' ')
                    
                    if rel_ego_side > LEFT_RIGHT_THRESHOLD:
                        print('and will be', "{:.0f}".format(rel_ego_side), 'meters on the right of the ego agent.', end=' ')
                    elif rel_ego_side < -LEFT_RIGHT_THRESHOLD:
                        print('and will be', "{:.0f}".format(-rel_ego_side), 'meters on the left of the ego agent.', end=' ')
                    else:
                        print('and will be on the same lane as the ego agent.', end=' ')
                else:
                    if rel_ego_side > LEFT_RIGHT_THRESHOLD:
                        print('Surrounding agent #', np.int64(rel_qa_id[i]), 'will be', "{:.0f}".format(rel_ego_side), 'meters on the right of the ego agent,', end=' ')
                    elif rel_ego_side < -LEFT_RIGHT_THRESHOLD:
                        print('Surrounding agent #', np.int64(rel_qa_id[i]), 'will be', "{:.0f}".format(-rel_ego_side), 'meters on the left of the ego agent,', end=' ')
                    else:
                        print('Surrounding agent #', np.int64(rel_qa_id[i]), 'will be on the same lane as the ego agent,', end=' ')

                    if rel_ego_front > 0:
                        print('and will be', "{:.0f}".format(rel_ego_front), 'meters in front of the ego agent.', end=' ')
                    else:
                        print('and will be', "{:.0f}".format(-rel_ego_front), 'meters behind the ego agent.', end=' ')
                
                if rel_id[i] == agent_id:
                    related_ego_dir = (rel_cur_heading_angle - agent_heading_angle[start_time + des_period]) % (2 * np.pi)
                    if related_ego_dir < 0.16 * np.pi or related_ego_dir > 1.84 * np.pi:
                        print('It will be heading in the same direction as the current moment.', end=' ')
                    elif related_ego_dir < 0.84 * np.pi:
                        print('It will be heading left of the current moment.', end=' ')
                    elif related_ego_dir < 1.16 * np.pi:
                        print('It will be heading the opposite direction as the current moment.', end=' ')
                    else:
                        print('It will be heading right of the current moment.', end=' ')
                elif related_ego_dir < 0.16 * np.pi or related_ego_dir > 1.84 * np.pi:
                    print('It will be heading in the same direction as the ego agent.', end=' ')
                elif related_ego_dir < 0.84 * np.pi:
                    print('It will be heading left of the ego agent.', end=' ')
                elif related_ego_dir < 1.16 * np.pi:
                    print('It will be heading the opposite direction as the ego agent.', end=' ')
                else:
                    print('It will be heading right of the ego agent.', end=' ')

                # 4.2 Speed and Acc Check
                max_vel = 0.0
                cur_speed = rel_speed[i,end_time]
                for k in range(end_time-des_period, end_time):
                    if rel_speed[i,k] > max_vel:
                        max_vel = rel_speed[i,k]

                if max_vel < MIN_MOVE_SPEED:
                    print('It will not be moving.', end=' ')
                else:
                    print('Its speed will be', "{:.0f}".format(cur_speed), 'm/s.', end=' ')

                # Step 5: Agent - Car Relation
                # 5.1 Relation to the Intersection
                agent_cur_pos = agent_traj[end_time,:]
                agent_cur_vel = agent_vel[end_time,:]
                rel_cur_pos = rel_traj[i,end_time,:]
                rel_cur_vel = rel_vel[i,end_time,:]
                rel_cur_z = rel_z[i,end_time]
                rel_in_int = False
                int_radius_for_related = int_radius - 5.0
                if no_int_label:
                    pass
                elif np.linalg.norm(rel_cur_pos - int_center_0) < int_radius_for_related:
                    print('It will be in the intersection.', end=' ')
                    rel_in_int = True
                elif max_vel > 0.1:      
                    if np.dot(rel_cur_vel, rel_cur_pos - int_center_0) > 0:
                        print('It will be departing from the intersection.', end=' ')
                    else:
                        print('It will be heading towards the intersection.', end=' ')

                # 5.2 Relation to the Agent
                rel_int_dir = rel_cur_pos - int_center_0
                rel_int_dir = rel_int_dir / (np.linalg.norm(rel_int_dir) + 0.01)

                agent_cur_pos = agent_traj[end_time,:]
                agent_int_dir = agent_cur_pos - int_center_0
                agent_int_dir = agent_int_dir / (np.linalg.norm(agent_int_dir) + 0.01)
                if no_int_label:
                    pass
                elif not rel_in_int:
                    print('Looking from the agent\'s current angle, ', end='')
                    if np.abs(np.dot(rel_int_dir, agent_int_dir)) > 0.707:
                        if np.dot(rel_int_dir, agent_int_dir) > 0.0:
                            print('it will be on the same side of the intersection.', end=' ')
                        else:
                            print('it will be at the opposite side of the intersection.', end=' ')
                    else:
                        if np.cross(rel_int_dir, agent_int_dir) > 0.0:
                            print('it will be on the left of the intersection.', end=' ')
                        elif np.cross(rel_int_dir, agent_int_dir) < -0.0:
                            print('it will be on the right of the intersection.', end=' ')
                       
                    # Lane Info for Related Agent
                    rel_lane_stat = np.zeros(max_id+1)
                    for j in range(end_time-des_period, end_time):
                        for k in range(road_length):
                            if road_type[k] > 2:
                                continue
                            if np.abs(road_z[k] - rel_cur_z) > 3.0:
                                continue
                            cur_error_sqr = np.linalg.norm(rel_traj[i,j] - road_xy[k]) ** 2
                            rel_lane_stat[road_ids[k]] += 1.0 / (cur_error_sqr + 0.001)
                    # print(lane_stat)
                    rel_chosen_id = np.argmax(rel_lane_stat)
                    rel_chosen_lane = road_xy[road_ids == rel_chosen_id,:]
                    rel_chosen_dir = road_dir[road_ids == rel_chosen_id,:]

                    # Related Agent Lane Position
                    tot_lane = []
                    right_lane = []
                    lane_dist_set = []
                    # Need to be changed to nearest point, not the start of the lane!
                    dist = 9999.0
                    chosen_point = 0
                    for k in range(1,rel_chosen_lane.shape[0]-1):
                        cur_dist = np.linalg.norm(rel_chosen_lane[k] - rel_traj[i,end_time]) ** 2
                        if cur_dist < dist:
                            dist = cur_dist
                            chosen_point = k
                    for k in range(road_length):
                        if road_type[k] > 2:
                            continue
                        if np.abs(np.dot(road_xy[k] - rel_chosen_lane[chosen_point], rel_chosen_dir[chosen_point])) > 2.00:
                            continue
                        if np.dot(road_dir[k], rel_chosen_dir[chosen_point]) < 0.75:
                            continue
                        if road_ids[k] == rel_chosen_id:
                            continue
                        if (road_xy[k,0] - rel_chosen_lane[chosen_point,0]) ** 2 + (road_xy[k,1] - rel_chosen_lane[chosen_point,1]) ** 2 < LANE_SEARCH_RADIUS ** 2:
                            cross = np.int64(np.cross(rel_chosen_lane[chosen_point] - road_xy[k], rel_chosen_dir[chosen_point]))
                            if cross not in lane_dist_set and np.abs(cross) > 1:
                                lane_dist_set.append(cross)
                                if road_ids[k] not in tot_lane:
                                    tot_lane.append(road_ids[k])
                                    if np.cross(rel_chosen_lane[chosen_point] - road_xy[k], rel_chosen_dir[chosen_point]) < 0:
                                        right_lane.append(road_ids[i])

                else:
                    pass
                
                if (not rel_in_int) and (not no_int_label):
                    related_int_dis = np.linalg.norm(rel_cur_pos - int_center_0)
                    print('It will be', "{:.0f}".format(related_int_dis), 'meters away from the intersection center.', end=' ')
                else:
                    pass
                print('\n', end='') 


            road_xy_print = road_xy[road_type[:,0] < 3,:]
            road_dir_print = road_dir[road_type[:,0] < 3,:]

            plt.quiver(road_xy_print[:,0],road_xy_print[:,1], road_dir_print[:,0], road_dir_print[:,1], angles='xy', scale_units='xy', scale=1, color='black', alpha=0.50)

            plt.plot(agent_traj[start_time+des_period:end_time,0], agent_traj[start_time+des_period:end_time,1], color='red', linewidth=2.0, alpha=1.0)
            plt.scatter(agent_traj[end_time,0], agent_traj[end_time,1], marker="x", s=20.00, color='red')


            plt.xlabel('X')
            plt.ylabel('Y')
            plt.xlim([-100,100])
            plt.ylim([-100,100])
            # plt.title('Map and Traj')
            plt.savefig('./fig/training-'+file_num+'-of-01000-'+str(output_num-1).zfill(3)+'.png')
            plt.show()
            plt.close()


            scene_id = parsed["scenario/id"].numpy().decode('utf8')

            head = {
                "sid": scene_id,
                "ego": int(agent_id),
                "cur_time": float((start_time+des_period)/10.0-0.10),
                "future_time": float(future_period/10.0),
                "rel_id": rel_id.tolist(),
                "rel_qa_id": rel_qa_id,
            }

            with open('./head/training-'+file_num+'-of-01000-'+str(output_num-1).zfill(3)+'.json', "w") as json_file:
                json.dump(head, json_file)
            

    sys.stdout.close()
    sys.stdout = original_stdout

if __name__ == '__main__':
    main()

