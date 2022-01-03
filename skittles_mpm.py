# based on MPM-MLS in 88 lines of Taichi code, which is originally created by @yuanming-hu
import taichi as ti
import numpy as np
import os

ti.init(arch=ti.gpu)

n_particles = 8192 * 2
n_grid = 128
dx = 1 / n_grid
dt = 2e-4

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

_x = ti.Vector.field(2, float, n_particles)
x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
J = ti.field(float, n_particles)
eps = 1e-6
_v = ti.Vector.field(2, float, n_particles)
err = ti.field(float, n_particles)

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))

L_1 = ti.Vector.field(2, float, 10)
L_2 = ti.Vector.field(2, float, 10)
B_1 = ti.Vector.field(2, float, 10)
B_2 = ti.Vector.field(2, float, 10)
R_1 = ti.Vector.field(2, float, 10)
R_2 = ti.Vector.field(2, float, 10)
BB_1 = ti.Vector.field(2, float, 1)
BB_2 = ti.Vector.field(2, float, 1)
I_1 = ti.Vector.field(2, float, 6)
I_2 = ti.Vector.field(2, float, 6)
bins = ti.field(int, 6)
color = ti.field(int, n_particles)
toner = [0xFF0000, 0xFF7F00, 0xFFFF00, 0x00FF00, 0x0000FF, 0x4B0082]


@ti.func
def collision_handle(i, j, p, q, n_type, bound_, mu):
    # n_type : 1. e_n = (1,0); 2. e_n = (0,1);
    if n_type == 1 and i < p.x + bound_ and i > p.x and j < ti.max(
            p.y, q.y) and j > ti.min(p.y, q.y) and grid_v[i, j].x < 0:
        grid_v[i, j].x *= -mu
    if n_type == 1 and i < p.x and i > p.x - bound_ and j < ti.max(
            p.y, q.y) and j > ti.min(p.y, q.y) and grid_v[i, j].x > 0:
        grid_v[i, j].x *= -mu
    if n_type == 2 and j < p.y + bound_ and j > p.y and i < ti.max(
            p.x, q.x) and i > ti.min(p.x, q.x) and grid_v[i, j].y < 0:
        grid_v[i, j].y *= -mu
    if n_type == 2 and j < p.y and j > p.y - bound_ and i < ti.max(
            p.x, q.x) and i > ti.min(p.x, q.x) and grid_v[i, j].y < 0:
        grid_v[i, j].y *= -mu


@ti.kernel
def color_by_bins():
    for j in range(n_particles):
        for i in range(5):
            if x[j].x > bins[i] / 128 and x[j].x < bins[i + 1] / 128:
                color[j] = i


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0
        for k in range(10):
            collision_handle(i, j, B_1[k], B_2[k], 2, 2, 0.0)
            collision_handle(i, j, L_1[k], L_2[k], 1, 2, 0.0)
            collision_handle(i, j, R_1[k], R_2[k], 1, 2, 0.0)
            collision_handle(i, j, BB_1[k], BB_2[k], 1, 2, 0.0)
            collision_handle(i, j, I_1[k], I_2[k], 1, 2, 0.0)

    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C


@ti.kernel
def init():
    bins[0] = 0
    bins[5] = 128
    for i in range(1, 5):
        bins[i] = 34 + (i - 1) * 20
    for i in range(n_particles):
        _x[i] = [ti.random() * 0.08 + 0.46, ti.random() * 0.08 + 0.9]
        v[i] = [0, -1]
        J[i] = 1
        color[i] = 5
    for i in range(4):
        for j in range(i + 1):
            B_1[(i * (i + 1)) // 2 +
                j] = [57 - 10 * i + 20 * j, 110 - 20 * i + 0 * j]
            B_2[(i * (i + 1)) // 2 +
                j] = [71 - 10 * i + 20 * j, 110 - 20 * i + 0 * j]
            L_1[(i * (i + 1)) // 2 +
                j] = [57 - 10 * i + 20 * j, 114 - 20 * i + 0 * j]
            L_2[(i * (i + 1)) // 2 +
                j] = [57 - 10 * i + 20 * j, 124 - 20 * i + 0 * j]
            R_1[(i * (i + 1)) // 2 +
                j] = [71 - 10 * i + 20 * j, 114 - 20 * i + 0 * j]
            R_2[(i * (i + 1)) // 2 +
                j] = [71 - 10 * i + 20 * j, 124 - 20 * i + 0 * j]
    #L_2[1].y=L_2[3].y=L_2[6].y=R_2[2].y=R_2[5].y=R_2[9].y=114
    L_2[1].y = L_2[1].y + 20
    L_2[3].y = L_2[3].y + 20
    L_2[6].y = L_2[6].y + 20
    R_2[2].y = R_2[2].y + 20
    R_2[5].y = R_2[5].y + 20
    R_2[9].y = R_2[9].y + 20
    BB_1[0] = [0, 0]
    BB_2[0] = [128, 0]
    for i in range(6):
        I_1[i].y = 0
        I_2[i].y = 48
    for i in range(6):
        I_1[i].x = I_2[i].x = bins[i]


@ti.kernel
def reset():
    for i in range(n_particles):
        x[i] = _x[i]
        v[i] = [0, -1]
        J[i] = 1


@ti.kernel
def get_error_list():
    for i in range(n_particles):
        err[i] = (v[i] - _v[i]).norm_sqr() / (_v[i].norm_sqr() + eps)


@ti.kernel
def get__v():
    for i in range(n_particles):
        _v[i] = v[i]


init()
reset()
gui = ti.GUI('MPM88')
t = 0
Err = np.zeros((10, n_particles))
ct = 0
T = 0
while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 'r':
            t = 300
            color_by_bins()
            reset()
        elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            break
    for s in range(50):
        substep()
    t += 1
    T += 1
    if t == 300:
        get__v()
        reset()
    if t == 600 and ct < 10:
        print("get error list ", ct)
        get_error_list()
        Err[ct, :] = err.to_numpy()
        ct += 1
    if ct == 10:
        print("error lists saved ")
        np.save('Err.npy', Err)
        ct += 1
    if not os.path.exists(f'./skittles_results'):
        os.mkdir('./skittles_results')
    gui.clear(0xFFFFFF)
    #gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.circles(x.to_numpy(), radius=1.5, palette=toner, palette_indices=color)
    gui.lines(B_1.to_numpy() / 128,
              B_2.to_numpy() / 128,
              radius=2,
              color=0x000000)
    gui.lines(L_1.to_numpy() / 128,
              L_2.to_numpy() / 128,
              radius=2,
              color=0x000000)
    gui.lines(R_1.to_numpy() / 128,
              R_2.to_numpy() / 128,
              radius=2,
              color=0x000000)
    gui.lines(I_1.to_numpy() / 128,
              I_2.to_numpy() / 128,
              radius=2,
              color=0x000000)
    gui.lines(BB_1.to_numpy() / 128,
              BB_2.to_numpy() / 128,
              radius=2,
              color=0x000000)
    #print(x.to_numpy)
    if T % 20 == 0:
        filename = f'./skittles_results/frame_{T:05d}.png'
        print(f'Frame {T} is recorded in {filename}')
        gui.show(filename)
    else:
        gui.show()