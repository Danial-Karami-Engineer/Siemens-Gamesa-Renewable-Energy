import numpy as np

# Rotation matrices (angles must be given in degrees)

def rot_x(deg):
    """Returns a 3x3 rotation matrix for rotation about the X-axis."""
    rad = np.radians(deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad),  np.cos(rad)]])

def rot_y(deg):
    """Returns a 3x3 rotation matrix for rotation about the Y-axis."""
    rad = np.radians(deg)
    return np.array([
        [ np.cos(rad), 0, np.sin(rad)],
        [0,            1,           0],
        [-np.sin(rad), 0, np.cos(rad)]])

def rot_z(deg):
    """Returns a 3x3 rotation matrix for rotation about the Z-axis."""
    rad = np.radians(deg)
    return np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad),  np.cos(rad), 0],
        [0,            0,           1]])


# Coordinate system (origin + orientation relative to global frame)

class CoordinateSystem:
    """
    Defines a coordinate system by:
        origin  : [Ox, Oy, Oz] in global coordinates
        x_deg   : rotation about global X-axis (degrees)
        y_deg   : rotation about global Y-axis (degrees)
        z_deg   : rotation about global Z-axis (degrees)
    Rotation order applied: X → Y → Z
    """
    def __init__(self, origin, x_deg=0, y_deg=0, z_deg=0):
        self.origin = np.array(origin, float)

        # Build orientation from X, Y, Z rotations
        Rx = rot_x(x_deg)
        Ry = rot_y(y_deg)
        Rz = rot_z(z_deg)

        # Final orientation of this coordinate system relative to global
        self.R = Rz @ Ry @ Rx  


# Transform a point from CSYS1 → CSYS2

def transform_point(point, cs1, cs2):
    """
    Converts a point [x1, y1, z1] defined in CSYS1 into
    the equivalent coordinates [x2, y2, z2] in CSYS2.
    """
    point = np.array(point, float)

    # point expressed in global coordinates
    point_global = cs1.origin + cs1.R @ point

    # convert global point into CSYS2 coordinates
    return cs2.R.T @ (point_global - cs2.origin)


# A simple 3D object having position + orientation

class Object:
    """
    Represents an object in 3D with:
        position : [x, y, z]
        x_deg, y_deg, z_deg : orientation angles (degrees)
    Rotation order applied: X → Y → Z
    """
    def __init__(self, position, x_deg=0, y_deg=0, z_deg=0):
        self.position = np.array(position, float)

        Rx = rot_x(x_deg)
        Ry = rot_y(y_deg)
        Rz = rot_z(z_deg)

        # object's internal rotation matrix
        self.rotation = Rz @ Ry @ Rx


# Transform a 3D object between coordinate systems

def transform_object(obj, cs1, cs2):
    """
    Converts both the location and orientation of an object
    from CSYS1 into CSYS2.
    """
    # new object position
    new_position = transform_point(obj.position, cs1, cs2)

    # object's orientation expressed in global
    rotation_global = cs1.R @ obj.rotation

    # orientation expressed in CSYS2
    new_rotation = cs2.R.T @ rotation_global

    # create new object and assign the new rotation
    out = Object(new_position)
    out.rotation = new_rotation
    return out


# Simple True/False test

if __name__ == "__main__":

    # define CSYS1 as identity
    cs1 = CoordinateSystem([0,0,0], 0, 0, 0)

    # CSYS2 from assignment: rotated 6° about Z, translated to [10,5,3]
    cs2 = CoordinateSystem([10,5,3], 0, 0, 6)

    # test point defined in CSYS1
    p = [1,0,0]

    # function result
    result = transform_point(p, cs1, cs2)

    # expected answer computed manually from formula
    expected = cs2.R.T @ (cs1.origin + cs1.R @ np.array(p) - cs2.origin)

    # print readable output
    print("Point transformation is correct:", np.allclose(result, expected))