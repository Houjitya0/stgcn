import sys

sys.path.extend(['../'])
from utils.rotation import *
from tqdm import tqdm


def pre_normalization(data, zaxis=[6, 13], xaxis=[0, 1]):
    N, M, T, V, C = data.shape
    s = data[:, :, :, :, :3].copy()
    
    print('pad the null frames with the previous frames')
    for i_s, skeleton in enumerate(tqdm(s)):  # pad
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break

    print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 1:2, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    print(
        'parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)
    data[:, :, :, :, :3] = s
    return data

# def create_transfomer_matrix()

# input : (n, m, t, v, c)
def transform_normarize(data, meta):
    print("start all_node_to_node_transforme_normarization")
    train_walk_num = meta["train_walk_num"]
    test_walk_num = meta["test_walk_num"]
    input = data[624//4*train_walk_num : 624//4*(train_walk_num+1), :, :3]
    V = input.shape[3]
    filename = ["up", "upper_right", "down", "lower_left"]
    transformer_matrix_folder = meta["transformer_matrix_folder"]
    for n, N in enumerate(tqdm(input)):
        for m, M in enumerate(N): 
            for t, T in enumerate(M):
                kp = np.zeros([V, 3])

                for i, x in enumerate(T):
                    # A_s = create_transfomer_matrix(data, meta)
                    A_s = np.load(f"utils/{transformer_matrix_folder}/{filename[train_walk_num]}_to_{filename[test_walk_num]}.npy")
                    print(i)
                    tmp = (A_s[meta["node_index"][0][i]] @ np.array([x]).T)
                    kp[i] = tmp.T
                input[n, m, t] = kp
                
    data[624//4*train_walk_num : 624//4*(train_walk_num+1)] = input[:3]
    return data

def point_transform(data, meta):
    train_walk_num = meta["train_walk_num"]
    test_walk_num = meta["test_walk_num"]
    input = data[624//4*train_walk_num : 624//4*(train_walk_num+1)]
    filename = ["up", "upper_right", "down", "lower_left"]
    transformer_matrix_folder = meta["transformer_matrix_folder"]
    
    A_s = np.load(f"utils/{transformer_matrix_folder}/{filename[train_walk_num]}_to_{filename[test_walk_num]}.npy")
    print("start points transforme normarization")
    for n, N in enumerate(tqdm(input)):
        for m, M in enumerate(N): 
            for t, T in enumerate(M):
                kp = np.zeros([45, 3])
                for i, x in enumerate(T):
                    
                    kp = input[n, m, t] @ A_s

                input[n, m, t] = kp
                
    data[624//4*train_walk_num : 624//4*(train_walk_num+1)] = input
    return data

    


if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalization(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)