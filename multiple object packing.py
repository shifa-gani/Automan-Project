import numpy as np
import matplotlib.pyplot as plt
from itertools import product, permutations
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

# Base class
class Shape3D:
    def __init__(self):
        self.position = None

    def get_center(self):
        raise NotImplementedError

    def collides_with(self, other):
        raise NotImplementedError

    def get_bounding_box(self):
        raise NotImplementedError

    def is_inside_box(self, box_size):
        min_corner, max_corner = self.get_bounding_box()
        return np.all(min_corner >= 0) and np.all(max_corner <= box_size)


# Sphere class
class Sphere(Shape3D):
    def __init__(self, diameter):
        super().__init__()
        self.radius = diameter / 2

    def get_center(self):
        return self.position

    def get_bounding_box(self):
        r = self.radius
        return self.position - r, self.position + r

    def collides_with(self, other):
        if isinstance(other, Sphere):
            return np.linalg.norm(self.position - other.position) < (self.radius + other.radius)
        elif isinstance(other, Cuboid) or isinstance(other, Cylinder):
            return other.collides_with(self)  # Delegate to other
        return False


# Cuboid class
class Cuboid(Shape3D):
    def __init__(self, size):
        super().__init__()
        self.size = np.array(size)

    def get_orientations(self):
        return [np.array(p) for p in set(permutations(self.size))]

    def get_center(self):
        return self.position + self.size / 2

    def get_bounding_box(self):
        return self.position, self.position + self.size

    def collides_with(self, other):
        if isinstance(other, Cuboid):
            for i in range(3):
                if self.position[i] + self.size[i] <= other.position[i] or \
                   other.position[i] + other.size[i] <= self.position[i]:
                    return False
            return True
        elif isinstance(other, Sphere):
            closest = np.maximum(self.position, np.minimum(other.position, self.position + self.size))
            dist = np.linalg.norm(other.position - closest)
            return dist < other.radius
        elif isinstance(other, Cylinder):
            return other.collides_with(self)
        return False


# Cylinder class
class Cylinder(Shape3D):
    def __init__(self, diameter, height):
        super().__init__()
        self.radius = diameter / 2
        self.height = height

    def get_orientations(self):
        return ["z", "x", "y"]

    def get_center(self):
        return self.position + np.array([0, 0, self.height / 2])

    def get_bounding_box(self):
        return self.position - [self.radius, self.radius, 0], self.position + [self.radius, self.radius, self.height]

    def collides_with(self, other):
        if isinstance(other, Cylinder):
            dz = abs(self.get_center()[2] - other.get_center()[2])
            vertical_clearance = (self.height + other.height) / 2
            dxdy = np.linalg.norm(self.get_center()[:2] - other.get_center()[:2])
            return dxdy < (self.radius + other.radius) and dz < vertical_clearance
        elif isinstance(other, Sphere):
            dxdy = np.linalg.norm(self.get_center()[:2] - other.get_center()[:2])
            dz = abs(self.get_center()[2] - other.get_center()[2])
            return dxdy < (self.radius + other.radius) and dz < (self.height / 2 + other.radius)
        elif isinstance(other, Cuboid):
            closest = np.maximum(other.position, np.minimum(self.get_center(), other.position + other.size))
            return np.linalg.norm(self.get_center() - closest) < self.radius
        return False


# General packer
class FlexiblePacker:
    def __init__(self, box_size, objects, K=1, K2=1, grid_step=0.05):
        self.box_size = np.array(box_size)
        self.objects = objects
        self.placed_objects = []
        self.K = K
        self.K2 = K2
        self.grid_step = grid_step

    def is_position_valid(self, obj, position):
        obj.position = position
        if not obj.is_inside_box(self.box_size):
            return False
        for other in self.placed_objects:
            if obj.collides_with(other):
                return False
        return True

    def compute_cost(self, obj):
        center = obj.get_center()
        h_thres = center[2]
        dist_sum = sum(np.linalg.norm(center - o.get_center()) for o in self.placed_objects)
        return self.K * h_thres + self.K2 * dist_sum

    def find_best_position(self, obj):
        best_cost = float('inf')
        best_position = None
        best_config = None

        if isinstance(obj, Sphere):
            radius = obj.radius
            r = [np.arange(radius, self.box_size[i] - radius + self.grid_step, self.grid_step) for i in range(3)]
            for pos in product(*r):
                pos = np.array(pos)
                if self.is_position_valid(obj, pos):
                    cost = self.compute_cost(obj)
                    if cost < best_cost:
                        best_cost = cost
                        best_position = pos
        elif isinstance(obj, Cuboid):
            for orientation in obj.get_orientations():
                obj.size = orientation
                r = [np.arange(0, self.box_size[i] - orientation[i] + self.grid_step, self.grid_step) for i in range(3)]
                for pos in product(*r):
                    pos = np.array(pos)
                    if self.is_position_valid(obj, pos):
                        cost = self.compute_cost(obj)
                        if cost < best_cost:
                            best_cost = cost
                            best_position = pos
                            best_config = orientation
            if best_config is not None:
                obj.size = best_config
        elif isinstance(obj, Cylinder):
            for axis in obj.get_orientations():
                if axis == "z":
                    r = [np.arange(obj.radius, self.box_size[0] - obj.radius + self.grid_step, self.grid_step),
                         np.arange(obj.radius, self.box_size[1] - obj.radius + self.grid_step, self.grid_step),
                         np.arange(0, self.box_size[2] - obj.height + self.grid_step, self.grid_step)]
                    for pos in product(*r):
                        pos = np.array(pos)
                        if self.is_position_valid(obj, pos):
                            cost = self.compute_cost(obj)
                            if cost < best_cost:
                                best_cost = cost
                                best_position = pos
        return best_position

    def pack_objects(self):
        for obj in self.objects:
            pos = self.find_best_position(obj)
            if pos is not None:
                obj.position = pos
                self.placed_objects.append(obj)
            else:
                print(f"Could not place object: {type(obj).__name__}")
        return self.placed_objects

    def compute_packing_summary(self):
        total_volume = np.prod(self.box_size)
        used_volume = 0
        max_z = 0

        for obj in self.placed_objects:
            if isinstance(obj, Sphere):
                used_volume += 4/3 * np.pi * obj.radius ** 3
                max_z = max(max_z, obj.position[2] + obj.radius)
            elif isinstance(obj, Cuboid):
                used_volume += np.prod(obj.size)
                max_z = max(max_z, obj.position[2] + obj.size[2])
            elif isinstance(obj, Cylinder):
                used_volume += np.pi * obj.radius**2 * obj.height
                max_z = max(max_z, obj.position[2] + obj.height)

        density = used_volume / total_volume * 100
        wasted = 100 - density

        return {
            "Packing Density (%)": round(density, 2),
            "Max Z Height": round(max_z, 2),
            "Wasted Volume (%)": round(wasted, 2)
        }

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([0, self.box_size[0]])
        ax.set_ylim([0, self.box_size[1]])
        ax.set_zlim([0, self.box_size[2]])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        for obj in self.placed_objects:
            if isinstance(obj, Sphere):
                self.plot_sphere(ax, obj)
            elif isinstance(obj, Cuboid):
                self.plot_cuboid(ax, obj)
            elif isinstance(obj, Cylinder):
                self.plot_cylinder(ax, obj)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_sphere(ax, sphere):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        r = sphere.radius
        center = sphere.position
        x = r * np.cos(u) * np.sin(v) + center[0]
        y = r * np.sin(u) * np.sin(v) + center[1]
        z = r * np.cos(v) + center[2]
        ax.plot_surface(x, y, z, color=np.random.rand(3,), alpha=0.6)

    @staticmethod
    def plot_cuboid(ax, cuboid):
        p = cuboid.position
        dx, dy, dz = cuboid.size
        corners = np.array([
            [p[0], p[1], p[2]],
            [p[0] + dx, p[1], p[2]],
            [p[0] + dx, p[1] + dy, p[2]],
            [p[0], p[1] + dy, p[2]],
            [p[0], p[1], p[2] + dz],
            [p[0] + dx, p[1], p[2] + dz],
            [p[0] + dx, p[1] + dy, p[2] + dz],
            [p[0], p[1] + dy, p[2] + dz]
        ])
        faces = [[corners[i] for i in f] for f in [[0, 1, 2, 3], [4, 5, 6, 7],
                                                   [0, 1, 5, 4], [2, 3, 7, 6],
                                                   [1, 2, 6, 5], [0, 3, 7, 4]]]
        color = np.random.rand(3,)
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='k', linewidths=0.5, alpha=0.7))

    @staticmethod
    def plot_cylinder(ax, cylinder):
        z = np.linspace(0, cylinder.height, 10) + cylinder.position[2]
        theta = np.linspace(0, 2 * np.pi, 30)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = cylinder.radius * np.cos(theta_grid) + cylinder.position[0]
        y_grid = cylinder.radius * np.sin(theta_grid) + cylinder.position[1]
        ax.plot_surface(x_grid, y_grid, z_grid, color=np.random.rand(3,), alpha=0.6)


# Example use
if __name__ == "__main__":
    box_size = [0.4, 0.5, 0.5]

    objects = [
        Sphere(0.3),
        Cuboid([0.2, 0.2, 0.3]),
        Cylinder(0.25, 0.2),
        Cuboid([0.15, 0.1, 0.4]),
        Sphere(0.1),
        Cylinder(0.1, 0.3),
        Sphere(0.08),
        Sphere(0.1),
        Cuboid([0.1, 0.1, 0.1]),
        Cuboid([0.15, 0.15, 0.1]),
    ]

    packer = FlexiblePacker(box_size, objects, K=2, K2=1, grid_step=0.05)
    packer.pack_objects()
    summary = packer.compute_packing_summary()
    print("Packing Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    packer.visualize()
