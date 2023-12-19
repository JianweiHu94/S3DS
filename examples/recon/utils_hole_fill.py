import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as p
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

label_font = {
    'color': 'c',
    'size': 15,
    'weight': 'bold'
}
x_angle = -90 / float(180) * np.pi

rot_matrix = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(x_angle), -np.sin(x_angle)],
                       [0.0, np.sin(x_angle), np.cos(x_angle)],
                       ])

r_rot_matrix = np.array([[1.0, 0.0, 0.0],
                         [0.0, np.cos(-x_angle), -np.sin(-x_angle)],
                         [0.0, np.sin(-x_angle), np.cos(-x_angle)],
                         ])


def get_filename(dataset_file):
    '''
    input:
    dataset_list : train or test file
    return src_list tgt_list
    '''
    f = open(dataset_file, 'r')
    lines = f.readlines()
    src_list = []
    tgt_list = []
    for i in range(len(lines)):
        line = lines[i].split(' ')
        src_path = line[0]
        tgt_path = line[1]
        _, nameEXT = os.path.split(src_path)
        name_src, ext = os.path.splitext(nameEXT)
        src_list.append(name_src)
        _, nameEXT = os.path.split(tgt_path)
        name_tgt, ext = os.path.splitext(nameEXT)
        tgt_list.append(name_tgt)
    return src_list, tgt_list


def plot_sample(spheres, edges, errors, filename, thred=0.02):
    verts = np.array(spheres)[:, 0:3]
    verts_rot = verts.dot(rot_matrix)

    fig = p.figure()
    ax = fig.add_subplot(111, projection='3d')
    lx = fig.gca(projection='3d')
    # print(len(spheres))

    for i in range(len(spheres)):
        ax.scatter(verts_rot[i][0], verts_rot[i][1], verts_rot[i][2], s=10, c="g")
    count = 0
    # index = np.arg_sort(errors)
    for edge in edges:
        c = 'y'
        if errors[count] > thred:
            c = 'r'
        i = edge[0]
        j = edge[1]
        X = [verts_rot[i][0], verts_rot[j][0]]
        Y = [verts_rot[i][1], verts_rot[j][1]]
        Z = [verts_rot[i][2], verts_rot[j][2]]
        lx.plot(X, Y, Z, c)
        ax.set_xlabel("X axis", fontdict=label_font)
        ax.set_ylabel("Y axis", fontdict=label_font)
        ax.set_zlabel("Z axis", fontdict=label_font)

        # if errors[count] > thred:
        #     x_t = (X[0] + X[1]) / 2
        #     y_t = (Y[0] + Y[1]) / 2
        #     z_t = (Z[0] + Z[1]) / 2

        #     ax.text(x_t,y_t,z_t,str(round(errors[count],3)))
        count += 1
        # ax.set_title("Scatter plot", alpha=0.6, color="b", size=25, weight='bold', backgroundcolor="y")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.view_init(45, 60)
    p.savefig(filename)
    p.close(fig)


def plot_ma(spheres, edges, filename, rot=True):
    fig = p.figure()
    ax = fig.add_subplot(111, projection='3d')
    lx = fig.gca(projection='3d')
    verts = np.array(spheres)[:, 0:3]
    # sample_pc = np.array(sample_pc)
    if rot == True:
        verts = verts.dot(rot_matrix)

    for i in range(len(spheres)):
        ax.scatter(verts[i][0], verts[i][1], verts[i][2], s=10, c="g")

    for edge in edges:
        c = 'y'
        i = edge[0]
        j = edge[1]
        X = [verts[i][0], verts[j][0]]
        Y = [verts[i][1], verts[j][1]]
        Z = [verts[i][2], verts[j][2]]
        lx.plot(X, Y, Z, c)
        ax.set_xlabel("X axis", fontdict=label_font)
        ax.set_ylabel("Y axis", fontdict=label_font)
        ax.set_zlabel("Z axis", fontdict=label_font)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.view_init(45, 60)
    p.savefig(filename)
    p.close(fig)


def plot_sample_face(spheres, faces, errors, filename, thred=0.01):
    verts_rot = np.array(spheres)[:, 0:3]
    verts_rot = verts_rot.dot(rot_matrix)
    verts = []
    for i in range(len(spheres)):
        verts.append([verts_rot[i][0], verts_rot[i][1], verts_rot[i][2]])

    fig = p.figure()
    ax = fig.add_subplot(111, projection='3d')
    lx = fig.gca(projection='3d')
    # print(len(spheres))
    for i in range(len(spheres)):
        ax.scatter(verts[i][0], verts[i][1], verts[i][2], s=10, c="g")
    count = 0

    # print(faces)
    for i in range(len(faces)):
        face_ = faces[i]
        face = [int(t) for t in face_]
        poly3d = [[verts[vert_id] for vert_id in face]]
        # x = [verts[face[0]][0],verts[face[1]][0],verts[face[2]][0]]
        # y = [verts[face[0]][1],verts[face[1]][1],verts[face[2]][1]]
        # z = [verts[face[0]][2],verts[face[1]][2],verts[face[2]][2]]
        if errors[i] > thred:
            # ax.plot_trisurf(x,y,z,linewidth=0.2,edgecolor='b',facecolor='red')
            ax.add_collection3d(Poly3DCollection(poly3d, facecolors='red', linewidths=1, alpha=0.1))
            ax.add_collection3d(Line3DCollection(poly3d, colors='k', linewidths=0.5, linestyles=':'))
        else:
            ax.add_collection3d(Poly3DCollection(poly3d, facecolors='blue', linewidths=1, alpha=0.1))
            ax.add_collection3d(Line3DCollection(poly3d, colors='k', linewidths=0.5, linestyles=':'))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    p.savefig(filename)
    p.close(fig)


def plot_sample_face(spheres, faces, filename):
    verts_rot = np.array(spheres)[:, 0:3]
    verts_rot = verts_rot.dot(rot_matrix)
    verts = []
    for i in range(len(spheres)):
        verts.append([verts_rot[i][0], verts_rot[i][1], verts_rot[i][2]])

    fig = p.figure()
    ax = fig.add_subplot(111, projection='3d')
    lx = fig.gca(projection='3d')

    for i in range(len(spheres)):
        ax.scatter(verts[i][0], verts[i][1], verts[i][2], s=10, c="g")
    count = 0

    # print(faces)
    poly3d = []
    for i in range(len(faces)):
        face_ = faces[i]
        face = [int(t) for t in face_]
        poly3d.append([verts[vert_id] for vert_id in face])
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='blue', linewidths=1, alpha=0.9))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    p.savefig(filename)
    p.close(fig)


def load_MAT(filename_mat):
    spheres = []
    edges = []
    faces = []
    lines = open(filename_mat, "r").read().replace("\\n", " ").splitlines()
    sphere_num = int(lines[0].split()[0])
    edge_num = int(lines[0].split()[1])
    face_num = int(lines[0].split()[2])
    for i in range(1, sphere_num + 1):
        spheres.append([float(s) for s in lines[i].split()[1:5]])
    for i in range(sphere_num + 1, sphere_num + edge_num + 1):
        edges.append([int(e) for e in lines[i].split()[1:3]])
    for i in range(sphere_num + edge_num + 1, sphere_num + edge_num + face_num + 1):
        faces.append([int(f) for f in lines[i].split()[1:4]])
    #print(len(spheres), len(edges), len(faces))
    return spheres, edges, faces

def load_surf_MAT(filename_mat):
    spheres = []
    edges = []
    faces = []
    lines = open(filename_mat, "r").read().replace("\\n", " ").splitlines()
    sphere_num = int(lines[0].split()[0])
    edge_num = int(lines[0].split()[1])
    face_num = int(lines[0].split()[2])
    fine_num = sphere_num // 2
    for i in range(1, sphere_num + 1):
        spheres.append([float(s) for s in lines[i].split()[1:5]])
    fine_edges = []
    for i in range(sphere_num + 1, sphere_num + edge_num + 1):
        edge = [int(e) for e in lines[i].split()[1:3]]
        if edge[0] < fine_num or edge[1] < fine_num:
            fine_edges.append(edge)
            continue
        edges.append(edge)
    fine_faces = []
    for i in range(sphere_num + edge_num + 1, sphere_num + edge_num + face_num + 1):
        face = [int(f) for f in lines[i].split()[1:4]]
        if face[0] > fine_num and face[1] > fine_num and face[2] > fine_num:
            faces.append(face)
        else:
            fine_faces.append(face)
    #print(len(spheres), len(edges), len(faces))
    return spheres, edges, faces, fine_edges, fine_faces


def plot_loop(spheres, loops, filename, rot=True):
    fig = p.figure()
    ax = fig.add_subplot(111, projection='3d')
    lx = fig.gca(projection='3d')
    verts = np.array(spheres)[:, 0:3]

    if rot:
        verts = verts.dot(rot_matrix)

    for i in range(len(spheres)):
        ax.scatter(verts[i][0], verts[i][1], verts[i][2], s=10, c="g")

    for loop in loops:
        rgbd = np.random.uniform(0, 1.0, size=(4))
        for i in range(len(loop)):
            e1 = loop[i]
            if i + 1 == len(loop):
                e2 = loop[0]
            else:
                e2 = loop[i + 1]

            # color = np.zeros([1, 4], 'float32')
            # color[0:3] = rgb
            # rgbd[3] = 0

            X = [verts[e1][0], verts[e2][0]]
            Y = [verts[e1][1], verts[e2][1]]
            Z = [verts[e1][2], verts[e2][2]]
            lx.plot(X, Y, Z, c=rgbd)
        # break
        # ax.set_xlabel("X axis", fontdict=label_font)
        # ax.set_ylabel("Y axis", fontdict=label_font)
        # ax.set_zlabel("Z axis", fontdict=label_font)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.view_init(45, 60)
    p.savefig(filename)
    p.close(fig)


def write_ma(filename_ma, spheres, edges, faces):
    f = open(filename_ma, 'w')
    sphere_num = len(spheres)
    edge_num = len(edges)
    face_num = len(faces)
    f.write('%d %d %d\n' % (sphere_num, edge_num, face_num))
    for i in range(sphere_num):
        f.write('v %f %f %f %f\n' % (spheres[i][0], spheres[i][1], spheres[i][2], spheres[i][3]))
    for i in range(edge_num):
        f.write('e %d %d\n' % (edges[i][0], edges[i][1]))
    for i in range(face_num):
        f.write('f %d %d %d\n' % (faces[i][0], faces[i][1], faces[i][2]))
    f.close()

def check_ma(spheres,edges,faces):
    valid_sphere_ids = []
    
    init_sphere_num = len(spheres)
    valid_sphere_mask = np.ones((init_sphere_num),'int')
    per_id_has_num_invalid_ids = np.zeros((init_sphere_num),'int')
    new_ids = np.zeros((init_sphere_num),'int')
    for i in range(init_sphere_num):
        xyzr = spheres[i]
        if abs(xyzr[0]) > 0.7 or abs(xyzr[1]) > 0.7 or abs(xyzr[2]) > 0.7:
            valid_sphere_ids.append(i)
            valid_sphere_mask[i] = 0
    
    if len(valid_sphere_ids) == 0:
        return spheres,edges,faces

    # start=0
    # valid_sphere_ids.append(init_sphere_num-1)
    # for i in range(len(valid_sphere_ids)):
    #     for j in range(start,valid_sphere_ids[i])
    #         if j < valid_sphere_ids[i]:
    #             per_id_has_num_invalid_ids[i] = i 
    #     start = valid_sphere_ids[i]
    valid_spheres = []
    count = 0
    for i in range(init_sphere_num):
        if valid_sphere_mask[i]:
            valid_spheres.append(spheres[i])
            new_ids[i] = count
            count+=1
        else:
            new_ids[i] = -1
    #print(count)
    valid_edges = []
    for edge in edges:
        if new_ids[edge[0]] != -1 and new_ids[edge[1]] != -1:
            new_edge = [new_ids[edge[0]],new_ids[edge[1]]]
            new_edge.sort()
            if new_edge not in valid_edges:
                valid_edges.append(new_edge)
    valid_faces = []
    for face in faces:
        if new_ids[face[0]] != -1 and new_ids[face[1]] != -1 and new_ids[face[2]] != -1:
            new_face = [new_ids[face[0]],new_ids[face[1]],new_ids[face[2]]]
            new_face.sort()
            if new_face not in valid_faces:
                valid_faces.append(new_face)
    return valid_spheres,valid_edges,valid_faces
    


def write_ma_check(filename_ma, spheres, edges, faces):
    spheres, edges, faces = check_ma(spheres, edges, faces)
    f = open(filename_ma, 'w')
    sphere_num = len(spheres)
    edge_num = len(edges)
    face_num = len(faces)
    f.write('%d %d %d\n' % (sphere_num, edge_num, face_num))
    for i in range(sphere_num):
        f.write('v %f %f %f %f\n' % (spheres[i][0], spheres[i][1], spheres[i][2], spheres[i][3]))
    for i in range(edge_num):
        f.write('e %d %d\n' % (edges[i][0], edges[i][1]))
    for i in range(face_num):
        f.write('f %d %d %d\n' % (faces[i][0], faces[i][1], faces[i][2]))
    f.close()




def write_off(filename_ma, spheres, edges, faces):
    f = open(filename_ma, 'w')
    sphere_num = len(spheres)
    edge_num = len(edges)
    face_num = len(faces)
    f.write('OFF\n')
    f.write('%d %d %d\n' % (sphere_num, face_num, edge_num))
    for i in range(sphere_num):
        f.write('%f %f %f\n' % (spheres[i][0], spheres[i][1], spheres[i][2]))
    for i in range(face_num):
        f.write('3 %d %d %d\n' % (faces[i][0], faces[i][1], faces[i][2]))
    for i in range(edge_num):
        f.write('2 %d %d\n' % (edges[i][0], edges[i][1]))

    f.close()


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


def save_graph_ma(ma_file,spheres,graph_mask):
    # graph_mask 328x328x1 numpy
    edges = np.where(graph_mask == 1.0)
    #print(len(edges[0]))
    edge_list = []
    for i in range(len(edges[0])):
        e = [edges[0][i],edges[1][i]]
        e.sort()
        if e not in edge_list:
            edge_list.append(e)
    
    faces = edge2face(graph_mask,graph_mask.shape[0])
    write_ma(ma_file,spheres,edge_list,faces)
    write_off(ma_file[:-2]+'off', spheres, edge_list, faces)

def edge2face(edge_matrix,num_point):
    edge_mask_1 = np.expand_dims(edge_matrix,-1) #  N N 1
    edge_mask_2 = np.transpose(edge_mask_1,(0,2,1)) # 1 N N
    edge_mask_3 = np.transpose(edge_mask_1,(2,1,0)) # N 1 N 

    edge_mask_1 = np.tile(edge_mask_1,(1,1,num_point))
    edge_mask_2 = np.tile(edge_mask_2,(1,num_point,1))
    edge_mask_3 = np.tile(edge_mask_3,(num_point,1,1))

    tmp = edge_mask_1 * edge_mask_2
    face_mask_matrix = tmp * edge_mask_3
    faces = np.where(face_mask_matrix == 1)
    face_list = []
    #print(faces[2])
    for i in range(len(faces[0])):
        f = [faces[0][i],faces[1][i],faces[2][i]]
        f.sort()
        if f not in face_list:
            face_list.append(f)
    return face_list
    