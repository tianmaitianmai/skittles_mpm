import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

N = 6
dt = 5e-5
dx = 1 / N
NV = (N + 1)**3  #number of vertices
NT = 6 * N**3  #number of tetrahedras
NF = 4 * NT  #number of faces(triangles)
s_0 = 7500.0
s_1 = 5000.0
mu_N = 0.5
Tet = ti.Vector.field(4, int, NT)
Force = ti.Vector.field(3, float, NV)
X = ti.Vector.field(3, float, NV)
V = ti.Vector.field(3, float, NV)
inv_Dm = ti.Matrix.field(3, 3, float, NT)
det_Dm = ti.field(float, NT)
damp = 0.99
gravity = ti.Vector([0, -40, 0])
ball_center = ti.Vector.field(3, float, (2, ))


@ti.kernel
def init_mesh():
    for i, j, k in ti.ndrange(N, N, N):
        l = (i * N * N + j * N + k) * 6
        p_0 = i * (N + 1) * (N + 1) + j * (N + 1) + k
        p_1 = p_0 + 1
        p_2 = p_0 + N + 1 + 1
        p_3 = p_0 + N + 1
        p_4 = p_0 + (N + 1)**2
        p_5 = p_0 + (N + 1)**2 + 1
        p_6 = p_0 + (N + 1)**2 + N + 1 + 1
        p_7 = p_0 + (N + 1)**2 + N + 1
        Tet[l + 0] = [p_0, p_6, p_1, p_2]
        Tet[l + 1] = [p_0, p_6, p_3, p_2]
        Tet[l + 2] = [p_0, p_6, p_1, p_5]
        Tet[l + 3] = [p_0, p_6, p_4, p_5]
        Tet[l + 4] = [p_0, p_6, p_3, p_7]
        Tet[l + 5] = [p_0, p_6, p_4, p_7]


@ti.kernel
def init_pos():
    for i, j, k in ti.ndrange(N + 1, N + 1, N + 1):
        l = i * (N + 1)**2 + j * (N + 1) + k
        X[l] = ti.Vector([i, j, k]) / 32 + ti.Vector([0.45, 0.45, 0.45])
        V[k] = ti.Vector([0, 0, 0])
    for i in range(NT):
        i_0, i_1, i_2, i_3 = Tet[i]
        x_0, x_1, x_2, x_3 = X[i_0], X[i_1], X[i_2], X[i_3]
        Dm_i = ti.Matrix.cols([x_1 - x_0, x_2 - x_0, x_3 - x_0])
        inv_Dm[i] = Dm_i.inverse()
        det_Dm[i] = Dm_i.determinant()


@ti.kernel
def advance():
    for i in range(NV):
        Force[i] = gravity
    for i in range(NT):
        # get deformation gradient
        i_0, i_1, i_2, i_3 = Tet[i]
        x_0, x_1, x_2, x_3 = X[i_0], X[i_1], X[i_2], X[i_3]
        edge_matrix = ti.Matrix.cols([x_1 - x_0, x_2 - x_0, x_3 - x_0])
        F = edge_matrix @ inv_Dm[i]

        G = 0.5 * (F.transpose() @ F - ti.Matrix.identity(float, 3))
        S = s_0 * G.trace() * ti.Matrix.identity(float, 3) + 2 * s_1 * G

        P = F @ S

        f_123 = -(P @ (inv_Dm[i].transpose())) * det_Dm[i]/ 6
        f_1 = ti.Vector([f_123[0, 0], f_123[1, 0], f_123[2, 0]])
        f_2 = ti.Vector([f_123[0, 1], f_123[1, 1], f_123[2, 1]])
        f_3 = ti.Vector([f_123[0, 2], f_123[1, 2], f_123[2, 2]])
        f_0 = -f_1 - f_2 - f_3

        Force[i_0] += f_0
        Force[i_1] += f_1
        Force[i_2] += f_2
        Force[i_3] += f_3
    for i in range(NV):
        V[i] *= damp
        V[i] += Force[i] * dt
        cond = X[i] < 0 and V[i] < 0 or X[i] > 1 and V[i] > 0
        for j in ti.static(range(X.n)):
            if cond[j]:
                V[i][j] *= -mu_N
        X[i] += dt * V[i]

window = ti.ui.Window('FEM_StVK_3D', (512, 512), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
init_mesh()
init_pos()

while window.running:
    for e in window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.ESCAPE:
            window.running = False
    for i in range(30):
        advance()
    camera.position(0.5, 0.8, 2)
    camera.lookat(0.5, 0.5, 0)
    scene.set_camera(camera)
    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
    scene.particles(X, radius=0.01, color=(0.5, 0, 0))
    canvas.scene(scene)
    window.show()