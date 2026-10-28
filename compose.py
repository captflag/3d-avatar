"""Assemble a dressed avatar: fitted face, body, clothes, hair and eyes.

The body is not generated from the photo and never will be. It is a catalogue
asset — a base mesh we dress — exactly as the product we are competing with does
it (../avatarsdk-re/REPORT.md section 5: their export is a body, a head and a
pile of separately authored wearables, fitted with visibility masks).

Base mesh and every wearable cut from it: MakeHuman, CC0 1.0, so commercial use
is unrestricted.

ponytail: the face is placed by bounding box and feathered at the edge, not
wrapped onto the skull. Flat garment colours, no fabric. Both are v1 cuts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

import appearance
import skinned_glb
import delight
import photo_check
import facetexture
import garments
import morph
import skin
import views
from catalogue import (BODY_OBJ, build_item, catalogue, hair_colour, items_by_id,
                       load_base, load_base_uv, srgb_to_linear, submesh, submesh_uv,
                       coverage_alpha)
from generate import NoFaceFound, build_mesh, expression_of, fit_face, load_canonical_faces
from rig import load_skeleton, load_weights, rigid_binding, weight_table

HERE = pathlib.Path(__file__).resolve().parent

# Landmarks that sit on cheek skin: away from eyes, lips, brows and hairline, so
# the median of them is a fair read of skin tone whatever the expression.
SKIN_LANDMARKS = [50, 280, 205, 425, 123, 352, 101, 330, 36, 266]

# How far the fitted face patch sits proud of the skull, in mesh units (about
# 10 cm each), so the two surfaces do not z-fight.
FACE_FORWARD_BIAS = 0.05

# How far the eyeballs sit behind the eyelid surface, in mesh units (~10 cm each).
EYE_RECESS = 0.06

# How strongly the generated pore map bumps the skin, per texture. The head's
# own map puts a texel at a fraction of a millimetre; the body's runs neck to
# toe, a texel is millimetres across, and the same grain at full strength read
# as bark on the neck.
PORE_STRENGTH = {"head": 0.5, "body": 0.25}
# Head and body skin textures, texels a side.
SKIN_SIZE = 2048
# How much a full crease darkens generated skin (appearance.skin_cavity).
CAVITY_DEPTH = 0.35
# Skin's index of refraction: about 1.4, a reflectance of 2.8% head on.
SKIN_IOR = 1.4
# How far the normals under the brows are smoothed (neighbour-averaging
# rounds), so the 3D light adds the face's form there but not the socket's
# hollow, which the photo already shows.
PHOTO_NORMAL_ROUNDS = 120
# Skin's stand-in for subsurface glow: sheen colour (linear RGB) and roughness.
SKIN_SHEEN = ((0.20, 0.07, 0.04), 0.55)
# How far teeth and tongue sit back from where the model puts them (mesh units).
TEETH_RECESS = 0.05
# The upper front teeth's crowns lengthened: from the biting edge, the first
# TEETH_CROWN of the model's teeth (mesh units, ~4 mm: up to where its gum dips
# between the crowns) is stretched by TEETH_RAISE (~3 mm) and everything above
# moved up by it. Only those 4 mm of crown showed between gum and edge, where a
# photo's laugh shows about 9 mm up to the lip (samples/portrait.jpg), and the
# gum showed under the upper lip as pink arches.
TEETH_CROWN = 0.04
TEETH_RAISE = 0.03
# How far the mouth's lining is set back into the mouth (mesh units, ~1.5 mm).
MOUTH_LINING = 0.015
# The ARKit shapes that part the lips, scaled together so the avatar's mouth
# opens as far as the photo's (open_as_photo).
OPENING = ("jawOpen", "mouthLowerDown", "mouthUpperUp")
# Cloth sheen: colour (linear RGB) and roughness. At 0.35 it washed navy out
# to slate grey.
CLOTH_SHEEN = ((0.12, 0.12, 0.12), 0.8)
# Hair: anisotropy strength, and its direction (radians from the texture's u
# axis, which the strands run across).
HAIR_ANISOTROPY = (0.7, np.pi / 2)

# Surface finish, written into the file itself so Blender, Unity and Unreal show
# what the studio shows. Everything used to go out at 0.85, which is right for
# cotton and wrong for everything else: skin read as dry plaster and the eyes
# had no wet highlight. Our own viewer hid it by overriding the values.
# A catalogue item can set its own `roughness`; these are the defaults.
# Eyes 0.4, with the studio's cornea coat at 0.3 (scene-setup SHADING.cornea):
# at 0.18 under a full coat the highlight over a light iris rendered 86-88 L*
# against the photos' 55, and the irises 2-9 L* dark. Measured on 1800-pixel
# renders of four faces, summed: highlight 77 -> 53 L* off, iris 20 -> 8.
ROUGHNESS = {"skin": 0.5, "eyes": 0.4, "hair": 0.45,
             "top": 0.85, "bottom": 0.85, "shoes": 0.4}
# The cornea as a clear coat over the eye (strength, roughness), written into
# the file so other viewers show its wet highlight too; the studio reads it
# from there. Without it, Blender, Unity or a web viewer drew a matte eye.
CORNEA_COAT = (0.3, 0.03)


def body_mesh(vertices: np.ndarray | None = None) -> trimesh.Trimesh:
    """The visible body, without the helper and joint groups."""
    base, groups = load_base()
    if vertices is None:
        vertices = base
    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=groups["body"], process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def skin_tone(photo: Image.Image, landmarks) -> np.ndarray:
    """Median skin colour of the face, as 0-255 RGB."""
    rgb = photo.convert("RGB")
    w, h = rgb.size
    samples = []
    for i in SKIN_LANDMARKS:
        if i >= len(landmarks):
            continue
        x = min(max(int(landmarks[i].x * w), 0), w - 1)
        y = min(max(int(landmarks[i].y * h), 0), h - 1)
        samples.append(rgb.getpixel((x, y)))
    if not samples:
        return np.array([200, 170, 150], dtype=np.uint8)
    return np.median(np.array(samples, dtype=np.float64), axis=0).astype(np.uint8)


def trim_face_border(face: trimesh.Trimesh, rings: int = 2) -> trimesh.Trimesh:
    """Peel the outermost ring of triangles off the fitted face.

    The rim of the patch is where the photo's background smears onto the side of
    the head, so dropping it loses nothing and lets the skull's own skin tone
    carry the jaw and hairline.

    A transparent falloff was tried first and was worse: a double-sided
    transparent mesh sorts badly and the face came out ghostly and streaked.
    ponytail: a hard edge one ring in, not a real wrap-and-blend.
    """
    mesh = face.copy()
    for _ in range(max(rings, 0)):
        edges = mesh.edges_sorted
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        border = set(np.unique(unique[counts == 1]).tolist())
        if not border:
            break
        keep = [not (set(tri.tolist()) & border) for tri in mesh.faces]
        if not any(keep):
            break
        mesh.update_faces(np.array(keep, dtype=bool))
    mesh.remove_unreferenced_vertices()
    return mesh


def face_anchor_points(face: trimesh.Trimesh) -> np.ndarray:
    """Eyes, nose tip, mouth and chin on a fitted (or canonical) face mesh.

    Eye-contour midpoints rather than iris landmarks, so MediaPipe's canonical
    face, which has no iris points, is anchored the same way as a real one. +X
    is the subject's left on both meshes, so the eyes are paired by x.
    """
    v = face.vertices
    a, b = (v[33] + v[133]) / 2.0, (v[362] + v[263]) / 2.0
    left, right = (a, b) if a[0] > b[0] else (b, a)
    return np.array([left, right, v[1], (v[13] + v[14]) / 2.0, v[152]])


# The hair as rendered came out about a fifth more saturated than the photo's
# at the same lightness (chroma 34 against 28, and 24 against 21): the dyed
# texture's strands and highlights add it. The dye is given that much less.
HAIR_DYE_CHROMA = 0.84


def dye_colour(rgb) -> np.ndarray:
    """The colour to dye hair so it renders as `rgb` was read from the photo."""
    import cv2
    lab = cv2.cvtColor(np.uint8([[np.clip(rgb, 0, 255)]]), cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[..., 1:] = 128.0 + (lab[..., 1:] - 128.0) * HAIR_DYE_CHROMA
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)[0, 0].astype(np.float64)


# How far round the mouth the photo is carried with it onto the mesh's
# mouth, as a share of the mouth's width (Gaussian reach): the skin round the
# lips, and then the lips themselves exactly.
MOUTH_CARRY = (0.45, 0.15)
# The hair cap under a haircut's cards: how far off the skin it sits (mesh
# units; the cards lie closer than 0.05 at the back of the head) and the
# tint weight it is cut to (see hairline_tint).
CAP_OFFSET = 0.012
CAP_CUT = 0.05
# How far below the hair's edge the cap fades out (mesh units): opaque
# under the cards, gone a centimetre below their lowest strands, where the
# photo shows skin between sparse hairs.
CAP_FADE = 0.06
# And how far above the lowest strands it starts to fade: the lowest few
# strands per slice set the edge, and a cap opaque down to them put solid
# hair 0.15 eye spacings lower than the photo's.
CAP_RAISE = 0.1
# How far round each eye the photo is carried onto the mesh's eye, in eye
# spacings (Gaussian reach): far enough that the brow comes with it.
# 0.7, faded out at the nose (see eyes_onto_mesh): brow landmark error
# 3.29 -> 3.03, 4.00 -> 3.85, 4.23 -> 4.00, 5.18 -> 4.67 against 0.5.
EYE_CARRY = 0.7


# The pitch, in degrees, MediaPipe reads on our own level head renders
# (1.5-4.5 across the five test faces, yaw and roll under 1). A photo's head
# pose is taken relative to it, so a photo read as level gives a level head.
# ponytail: one constant for every avatar; measure per avatar if it matters.
LEVEL_READ_PITCH = 3.3
# Landmarks an expression hardly moves: forehead, brows, eye corners, the
# nose's bridge and the temples; the pitch a rigid fit to them alone reads on
# our level renders (-2.2 to 2.8, mean 1.0); and how far it may disagree with
# MediaPipe's before MediaPipe's is taken to be bent by the expression.
POSE_LANDMARKS = [10, 151, 9, 8, 168, 6, 197, 195, 33, 133, 263, 362, 70, 63, 105, 66, 107,
                  300, 293, 334, 296, 336, 127, 356, 162, 389, 21, 251, 54, 284]
STABLE_LEVEL_PITCH = 1.0
POSE_DISAGREE = 5.0


def _pitch(rotation) -> float:
    return float(np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2])))


def photo_head_pose(result, landmarks, photo) -> list[float] | None:
    """How the head was turned in the photo, as a quaternion (x y z w) in the
    scene's axes, for a viewer that renders the head as photographed. Test
    photos are shot 4-12 degrees from level; compared with a level render, that
    pitch was charged to the nose and jaw, 1-1.5% of the eye spacing of
    landmark error.

    MediaPipe's head transform, unless an expression has bent it: it is fitted
    to every landmark, and on a laughing photo the raised cheeks and dropped
    jaw read as 11 degrees of pitch where the forehead and eyes alone show 3.
    Where the two readings part by more than POSE_DISAGREE, the rigid fit of
    MediaPipe's average face to POSE_LANDMARKS is taken instead. Not always:
    its own level reading varies more from face to face (-2.2 to 2.8 against
    MediaPipe's 1.5-4.5), and taken on every face it cost ref-man 0.33."""
    from scipy.spatial.transform import Rotation
    if not result.facial_transformation_matrixes:
        return None
    u, _, vt = np.linalg.svd(np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)[:3, :3])
    turn, level = u @ vt, LEVEL_READ_PITCH
    points = np.asarray(build_mesh(landmarks, load_canonical_faces(), photo, None).vertices)[POSE_LANDMARKS]
    average = np.asarray(morph.canonical_face().vertices)[POSE_LANDMARKS]
    a, b = average - average.mean(axis=0), points - points.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    stable = (u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt).T
    if abs(_pitch(turn) - _pitch(stable)) > POSE_DISAGREE:
        turn, level = stable, STABLE_LEVEL_PITCH
    return [float(c) for c in (Rotation.from_matrix(turn)
                               * Rotation.from_euler("x", level, degrees=True).inv()).as_quat()]


def level_upper_face(sheet: np.ndarray) -> np.ndarray:
    """The placed sheet above the eyes turned by LEVEL_READ_PITCH.

    The sheet is the photo's landmarks turned upright by MediaPipe's whole head
    rotation, but the render turns the head back by that rotation less
    LEVEL_READ_PITCH. Paint laid on the one head and drawn on the other moves
    the brow ridge (forward of the eyes) and the lid (behind them) apart: the
    brows rendered 1-3.6% of the eye spacing high and a fifth thicker. Turned
    in full from the eye line up, and not at all from the nose tip down, so the
    nose and mouth stay where they were placed."""
    from scipy.spatial.transform import Rotation
    turn = Rotation.from_euler("x", LEVEL_READ_PITCH, degrees=True).as_matrix()
    eyes = sheet[[33, 133, 263, 362]].mean(axis=0)
    turned = (sheet - eyes) @ turn.T + eyes
    share = np.clip((sheet[:, 1] - sheet[1, 1]) / (eyes[1] - sheet[1, 1]), 0.0, 1.0)[:, None]
    return sheet + share * (turned - sheet)


def mouth_onto_mesh(sheet: np.ndarray, vertices: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """The placed landmark sheet with its mouth moved onto the mesh's own.

    The photo is baked through this sheet, so wherever its mouth sits is where
    the photo's lips are painted. The fitted expression takes most of a smile
    off it, but MakeHuman's units cannot reach a broad laugh: on one photo the
    sheet's mouth stayed 15% wider than the avatar's closed one, the grin's
    corners were painted out on the cheeks, and the neutral face read as
    smiling (MediaPipe 0.64; its shape alone 0.03). The outer lip line is
    moved in the picture's plane onto the mesh's, and the skin round it with a
    falloff, so the photo's lips land on the avatar's.
    """
    lips = [61, 291] + morph.OUTER_UPPER + morph.OUTER_LOWER
    width = float(np.linalg.norm(sheet[61, :2] - sheet[291, :2]))
    kernel = lambda a, b, reach: np.exp(-(np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=2) / reach) ** 2)
    pull = vertices[corr[lips], :2] - sheet[lips, :2]
    # The skin round the mouth: the lips' pull averaged, fading with distance.
    w = kernel(sheet, sheet[lips], MOUTH_CARRY[0] * width)
    field = (w @ pull) / np.maximum(w.sum(axis=1, keepdims=True), 1e-12) * w.max(axis=1, keepdims=True)
    # The lips exactly: an average dilutes the corners' pull with the lip's
    # middle, which barely moves, and left a third of a laugh's width. What is
    # left is fitted by a narrow Gaussian radial basis, damped so it does not
    # ring between the closely spaced lip points.
    near = kernel(sheet[lips], sheet[lips], MOUTH_CARRY[1] * width)
    alpha = np.linalg.solve(near + 1e-2 * np.eye(len(lips)), pull - field[lips])
    moved = sheet.copy()
    moved[:, :2] += field + kernel(sheet, sheet[lips], MOUTH_CARRY[1] * width) @ alpha
    return moved


def eyes_onto_mesh(sheet: np.ndarray, vertices: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """The placed landmark sheet with its eyes moved onto the mesh's own, and
    the brows with them.

    The eyelids the avatar shows are the mesh's, and the brows are painted
    and carded from the sheet. Placed by one scale and shift across eyes,
    nose, mouth and chin, the sheet's eyes do not land on the mesh's lids,
    and on the test photos the brows came out 4-8% of the eye spacing higher
    above the lids than the photo's. Each eye's contour is moved in the
    picture's plane onto the mesh's, and the skin round it, brows included,
    by the same pull fading with distance.
    """
    marks = list(appearance.EYE_CONTOURS[0]) + list(appearance.EYE_CONTOURS[1])
    spacing = float(np.linalg.norm(sheet[33, :2] - sheet[263, :2]))
    pull = vertices[corr[marks], :2] - sheet[marks, :2]
    w = np.exp(-(np.linalg.norm(sheet[:, None, :2] - sheet[None, marks, :2], axis=2)
                 / (EYE_CARRY * spacing)) ** 2)
    field = (w @ pull) / np.maximum(w.sum(axis=1, keepdims=True), 1e-12) * w.max(axis=1, keepdims=True)
    # Faded out from the eyes down to the nose tip: the mouth and chin are
    # placed by mouth_onto_mesh and the anatomy, not by the eyes.
    eye_y, nose_y = float(sheet[marks, 1].mean()), float(sheet[1, 1])
    field *= np.clip((sheet[:, 1] - nose_y) / max(eye_y - nose_y, 1e-9), 0.0, 1.0)[:, None]
    moved = sheet.copy()
    moved[:, :2] += field
    return moved


def place_face(face: trimesh.Trimesh, vertices: np.ndarray, groups: dict) -> trimesh.Trimesh:
    """Put the fitted face on the skull by anatomy.

    Its eyes, nose tip, mouth and chin are lined up with where the skull keeps
    its own: the eyeball and teeth helpers, and the skull's furthest-forward and
    lowest facial points. A uniform scale and an offset are solved by least
    squares across the face plane, and depth is set by the nose tip.

    This replaces placement by bounding box with hand-tuned ratios, which sat
    the face 3 to 5 cm too high: the photo's mouth landed on the skull's nose and
    its chin on the skull's mouth. It also matters beyond the patch, because the
    shape fit finds its landmark correspondences from where the average face is
    placed.
    """
    skull = facetexture.head_features(vertices, groups)
    target = np.array([skull["left_eye"], skull["right_eye"], skull["nose"],
                       skull["mouth"], skull["chin"]])
    source = face_anchor_points(face)

    src, dst = source[:, :2], target[:, :2]
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    scale = float(((src - src_mean) * (dst - dst_mean)).sum() / ((src - src_mean) ** 2).sum())

    placed = face.copy()
    placed.apply_scale(scale)
    shift = np.zeros(3)
    shift[:2] = dst_mean - scale * src_mean
    # Sit the nose tip just proud of the skull's, so the patch does not z-fight
    # with the skull's own face underneath.
    shift[2] = skull["nose"][2] - scale * source[2, 2] + FACE_FORWARD_BIAS
    placed.apply_translation(shift)
    return placed


def rest_pose() -> dict:
    """A relaxed stance instead of the base mesh's modelling pose.

    Arms out and fingers splayed is how a mesh is modelled, not how a person
    stands, and it is the first thing that reads as mannequin. Bones here are
    translation-only with world-aligned axes, so these are plain world-axis
    rotations.
    """
    pose = finger_curls()
    # Bring the arms down towards the body. Mirrored, hence the sign flip.
    for side, sign in (("L", 1.0), ("R", -1.0)):
        pose[f"upperarm01.{side}"] = skinned_glb.axis_angle([0, 0, 1], -22.0 * sign)
        pose[f"lowerarm01.{side}"] = skinned_glb.axis_angle([0, 0, 1], -10.0 * sign)
        # The model's upper lids sit at the top of the iris: a stare. At rest
        # they cover its top edge. (orbicularis03 is the upper lid, pivoting
        # at the eye's centre; a turn about +x lowers it.)
        pose[f"orbicularis03.{side}"] = skinned_glb.axis_angle([1, 0, 0], UPPER_LID_DROP)
    return pose


# Degrees the upper lids come down from the model's: 4 covers the iris's top
# edge, judged on renders against the stare it replaces (7 hid nearly half of
# it: sleepy).
UPPER_LID_DROP = 4.0


# Degrees each finger joint bends towards the palm at rest, knuckle to tip,
# thumb to little finger. A relaxed hand curls more towards the little finger.
# (Half as much again, 52 to 84 degrees down a finger, turned the hanging
# hands' fingertips and thumbs in across the thighs towards the fly.)
FINGER_CURL = {1: (0, 5, 5), 2: (8, 18, 8), 3: (10, 22, 10), 4: (12, 26, 12), 5: (15, 30, 14)}


def finger_curls() -> dict:
    """Each finger bone bent towards its palm, from the base mesh's bind pose.

    A hand at rest curls; straight fingers read as a shop dummy's paddle. No
    world axis does it: the fingers hang down and forward and the palm faces in
    and down, so the first version's turn about y swung them sideways instead.
    The palm faces across the knuckles and along the middle finger, on the
    side the thumb sits.
    """
    skeleton = load_skeleton(load_base()[0])
    head = lambda name: skeleton.heads[skeleton.index[name]]
    tail = lambda name: skeleton.tails[skeleton.index[name]]
    pose = {}
    for side in ("L", "R"):
        knuckles = (head(f"finger2-1.{side}"), head(f"finger5-1.{side}"))
        palm = np.cross(knuckles[1] - knuckles[0], tail(f"finger3-3.{side}") - head(f"finger3-1.{side}"))
        palm *= np.sign(palm @ (tail(f"finger1-3.{side}") - (knuckles[0] + knuckles[1]) / 2))
        for finger, angles in FINGER_CURL.items():
            for joint, degrees in enumerate(angles, 1):
                bone = f"finger{finger}-{joint}.{side}"
                pose[bone] = skinned_glb.axis_angle(np.cross(tail(bone) - head(bone), palm), degrees)
    return pose


def relaxed_pose() -> dict:
    """Standing at ease: weight on the right leg, the left knee soft.

    Everything a person does standing still and a mannequin does not: the
    hip over the weight leg rises and the shoulders tilt the other way, the
    free knee bends and its foot turns out a little, the arms hang closer and
    the elbows bend. Built on the rest pose, with the same world-axis
    rotations; where a bone turns about two axes they are composed.

    Angles in degrees. +x is the body's left, +y up, +z forward; a positive
    turn about z swings a hanging limb towards +x.
    """
    from scipy.spatial.transform import Rotation

    def turn(z=0.0, x=0.0, y=0.0):
        # Applied in the order z, then x, then y, as one glTF quaternion.
        return [float(c) for c in Rotation.from_euler("zxy", [z, x, y], degrees=True).as_quat()]

    pose = rest_pose()
    hip = 5.0
    pose["root"] = turn(z=-hip)                       # right hip up
    pose["spine05"] = turn(z=hip + 2.0)               # torso back up, shoulders tilt left-down
    # Head level and square to the camera, as a portrait photo has it. A 4
    # degree turn here put every rendered face's nose line 9% of the eye
    # spacing off-centre: one cheek 17-20% wider than the other, whatever the
    # photo, read as a crooked face.
    pose["neck01"] = turn(z=-2.0)
    # The base mesh stands with its ankles twice its hip width apart; at ease
    # they are about hip width (measured: 2.7 against hips of 2.2).
    # Weight leg: in under the body's centre.
    pose["upperleg01.R"] = turn(z=hip + 8.0)
    pose["foot.R"] = turn(z=-(hip + 8.0))
    # Free leg: in a little too, knee forward, foot flat and turned out.
    pose["upperleg01.L"] = turn(z=hip - 4.0, x=-10.0)
    pose["lowerleg01.L"] = turn(x=17.0)
    pose["foot.L"] = turn(z=-(hip - 4.0), x=-7.0, y=12.0)
    # Arms closer to the body than the rest pose, elbows soft. The base mesh
    # is modelled with its forearms reaching forward, about 46 degrees off
    # vertical once the upper arm hangs; +44 brings them down to ~14, hands at
    # the sides of the thighs rather than in front of them (measured: the
    # upper arm leans a little forward too).
    # (This was -16 at first, which pushed them out to about 62: sleepwalking,
    # then +31, which still held the hands out in front: ~24.)
    # The forearm-twist bone then sets the palms on the thighs, thumbs a little
    # forward, twisted about the forearm's own length.
    for side, sign in (("L", 1.0), ("R", -1.0)):
        pose[f"upperarm01.{side}"] = turn(z=-33.0 * sign, x=-4.0)
        pose[f"lowerarm01.{side}"] = turn(z=-4.0 * sign, x=44.0)
        forearm = np.array([0.503 * sign, -0.481, 0.718])
        forearm /= np.linalg.norm(forearm)
        pose[f"lowerarm02.{side}"] = [float(c) for c in Rotation.from_rotvec(
            forearm * np.radians(PALM_TWIST * sign)).as_quat()]

    # The model's thumb sticks out forward and in; hanging, it pointed across
    # the thigh at the fly. Turned at its root towards straight down, as the
    # arm now hangs: worked out on the posed skeleton, since "down" depends on
    # everything above the hand.
    from garments import skinning_matrices
    skeleton = load_skeleton(load_base()[0])
    matrices = skinning_matrices(skeleton, pose)
    for side in ("L", "R"):
        root, tip = skeleton.index[f"finger1-1.{side}"], skeleton.index[f"finger1-3.{side}"]
        start = (matrices[root] @ np.append(skeleton.heads[root], 1.0))[:3]
        end = (matrices[tip] @ np.append(skeleton.tails[tip], 1.0))[:3]
        thumb = (end - start) / np.linalg.norm(end - start)
        axis = np.cross(thumb, [0.0, -1.0, 0.0])
        # Bones are world-aligned at rest, so the parent's skinning rotation
        # takes the posed frame back to the thumb's own.
        local = matrices[skeleton.parents[root]][:3, :3].T @ (axis / np.linalg.norm(axis))
        turn_down = Rotation.from_rotvec(local * np.radians(THUMB_DROP))
        pose[f"finger1-1.{side}"] = [float(c) for c in (
            turn_down * Rotation.from_quat(pose[f"finger1-1.{side}"])).as_quat()]
    return pose


# Degrees the thumb is turned down from where the model holds it: from 28 to
# about 50 below horizontal, still 24 from the index finger.
THUMB_DROP = 25.0


def open_as_photo(expression: dict, vertices: np.ndarray, shapes: np.ndarray, marks: np.ndarray,
                  landmarks, photo_size) -> dict:
    """`expression` with the shapes that part the lips (OPENING) scaled
    together until the mesh's lips stand as far apart at their middle,
    front-on and in eye spacings, as the photo's (landmarks 13 and 14).

    MediaPipe's scores through the response curves overshot a broad laugh:
    on samples/portrait.jpg the render's lips stood 0.256 eye spacings apart
    against the photo's 0.149, both rows of teeth and the tongue showing
    under a thick rolled upper lip. The expression the fit took off instead
    undershoots one (REALISM-GAP.md, route 7a). Measured on the mesh, so no
    render is needed; `marks` are the landmarks' vertices (morph.lip_corr)."""
    w, h = photo_size
    at = lambda i: np.array([landmarks[i].x * w, landmarks[i].y * h])
    wanted = np.linalg.norm(at(13) - at(14)) / np.linalg.norm(at(33) - at(263))
    names = list(morph.ARKIT_NAMES)

    def gap(scale):
        moved = vertices[marks[[13, 14, 33, 263]]].astype(np.float64)
        for name, weight in expression.items():
            k = min(1.0, weight * scale) if name.startswith(OPENING) else weight
            moved = moved + k * shapes[names.index(name)][marks[[13, 14, 33, 263]]]
        return np.linalg.norm((moved[0] - moved[1])[:2]) / np.linalg.norm((moved[2] - moved[3])[:2])

    lo, hi = 0.0, 3.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) < wanted else (lo, mid)
    scale = (lo + hi) / 2
    return {name: (min(1.0, weight * scale) if name.startswith(OPENING) else weight)
            for name, weight in expression.items()}


def expression_morphs(mesh, shapes: np.ndarray):
    """Blendshapes on one part: (shapes, vertices, 3) offsets, or None if none
    of them moves it.

    `shapes` are offsets on the base mesh (morph.arkit_shapes: ARKit's 52,
    built from MakeHuman's CC0 expression units). A part cut from the base mesh
    takes them by its base index; a part hung from it by MHCLO (brows, lashes,
    teeth, garments) by the same three vertices and weights it hangs from, so
    brows follow a raised brow and the tongue a tongue put out.
    """
    if "base_index" in mesh.metadata:
        moved = shapes[:, mesh.metadata["base_index"]]
    elif "mhclo_idx" in mesh.metadata:
        idx, blend = mesh.metadata["mhclo_idx"], mesh.metadata["mhclo_blend"]
        moved = (shapes[:, idx] * blend[None, :, :, None]).sum(axis=2)
    else:
        return None
    return moved if np.abs(moved).max() > 1e-6 else None


# Below this many pixels across, a photo's face cannot carry a close likeness:
# MediaPipe reads the face at a fixed resolution, so enlarging a small photo
# adds nothing (measured: landmarks wander as much at 4x as at 1x).
GOOD_FACE_PX = 250


def photo_advice(landmarks, photo_size, expression: dict) -> list[str]:
    """What would make this photo give a closer avatar, in plain words."""
    w, h = photo_size
    advice = []
    face_px = abs(landmarks[454].x - landmarks[234].x) * w
    if face_px < GOOD_FACE_PX:
        advice.append(f"The face is only {face_px:.0f} pixels across; a closer or sharper photo "
                      f"(face {GOOD_FACE_PX}+ pixels wide) gives a much closer likeness.")
    smile = max((expression.get(k, 0.0) for k in morph.MOUTH_SCORES), default=0.0)
    if smile > 0.5:
        advice.append("The photo is smiling broadly, so the resting mouth is estimated; "
                      "a relaxed, closed-mouth photo shows the lips' real shape.")
    return advice


ARM_CHAIN = ("clavicle", "shoulder", "upperarm", "lowerarm", "wrist", "metacarpal", "finger")


def _skin(part, rows, matrices) -> np.ndarray:
    """Linear-blend-skinned positions of some of a part's vertices."""
    v = np.c_[part.vertices[rows], np.ones(int(np.count_nonzero(rows)) if rows.dtype == bool else len(rows))]
    out = np.zeros((len(v), 3))
    for k in range(part.joints.shape[1]):
        m = matrices[part.joints[rows, k]]
        out += part.weights[rows, k:k + 1] * np.einsum("nij,nj->ni", m, v)[:, :3]
    return out


def hand_clearance(pose: dict, skeleton, body) -> dict:
    """Per side, how far the hand stands off the hip and thigh skin beside it:
    the median of each hand vertex's distance along the nearest skin's outward
    normal, everything skinned into the pose. Negative is inside the body.

    Measured against the skin, not the clothes: garment normals are not
    reliably outward, the body's are. Clothes sit about 0.08 off the skin.
    """
    from scipy.spatial import cKDTree
    from garments import skinning_matrices
    matrices = skinning_matrices(skeleton, pose)
    dominant = body.joints[np.arange(len(body.joints)), body.weights.argmax(axis=1)]
    on_hip = np.array([n.startswith(("root", "spine01", "pelvis", "upperleg")) for n in skeleton.names])[dominant]
    skin = _skin(body, on_hip, matrices)
    outward = np.einsum("nij,nj->ni", matrices[dominant[on_hip]][:, :3, :3], body.normals[on_hip])
    tree = cKDTree(skin)
    clear = {}
    for side in ("L", "R"):
        on_hand = np.array([n.endswith("." + side) and n.startswith(("wrist", "metacarpal", "finger"))
                            for n in skeleton.names])[dominant]
        h = _skin(body, on_hand, matrices)
        nearest = tree.query(h)[1]
        clear[side] = float(np.median(((h - skin[nearest]) * outward[nearest]).sum(axis=1)))
    return clear


def arms_clear(pose: dict, skeleton, body, need: float = 0.12, step: float = 2.0,
               most: float = 20.0) -> dict:
    """The pose with each arm swung out until its hand clears the hip.

    The arm pose is set on the male base body. A fuller figure, wider at the
    hips, put the woman's hands inside her trouser legs (median 0.25 inside the
    thigh). Each arm turns out from the shoulder a step at a time until its
    hand stands `need` off the skin, enough for the clothes over it.
    """
    from scipy.spatial.transform import Rotation
    pose = dict(pose)
    for side, sign in (("L", 1.0), ("R", -1.0)):
        rest = Rotation.from_quat(pose.get(f"upperarm01.{side}", [0.0, 0.0, 0.0, 1.0]))
        for extra in np.arange(0.0, most + step / 2, step):
            pose[f"upperarm01.{side}"] = [float(c) for c in (
                Rotation.from_euler("z", extra * sign, degrees=True) * rest).as_quat()]
            if hand_clearance(pose, skeleton, body)[side] >= need:
                break
    return pose


# Degrees the forearm twists, measured on the posed palm normal: at 0 the palm
# already faces the thigh, -20 brings the thumb a little forward as a hanging
# hand does. (It was -70, judged on renders of straight fingers where palm and
# back look alike, and it turned the palms up and forward: begging hands.)
PALM_TWIST = -20.0


POSES = {"relaxed": relaxed_pose, "rest": rest_pose}


def hairline_for(face: trimesh.Trimesh, vertices: np.ndarray, groups: dict, lift: float = 0.0):
    """Where the hair stops, as a height that varies around the head.

    A straight horizontal cut sat the hair on the crown like a swim cap and left
    the temples and the back of the skull bare. Real hair starts above the
    forehead, comes down at the temples to just above the ears, and drops
    towards the nape at the back.

    Driven by the angle around the head's vertical axis — front, temples, ears,
    back — rather than by depth: measuring "how far back" from the nose tip
    counted the curve of the forehead itself as the back of the head, and the
    first version brought hair down over the brows.

    ponytail: one hairline shape for everyone. Receding lines, fringes and long
    hair want authored pieces from the catalogue.
    """
    body = vertices[np.unique(groups["body"])]
    top = float(face.bounds[1][1])                       # upper forehead landmark
    head = body[body[:, 1] > float(face.bounds[0][1])]   # everything above the chin
    centre_x = float(np.median(face.vertices[:, 0]))
    centre_z = float((head[:, 2].max() + head[:, 2].min()) / 2.0)

    # `lift` raises the line all round: an underlayer worn beneath a haircut
    # must stop short of the haircut's own edge, or it shows as a dark band.
    forehead = top + 0.10 + lift      # a little above the highest landmark
    above_ears = top - 0.30 + lift    # temples and sides
    nape = top - 1.05          # back of the head

    def line(points: np.ndarray) -> np.ndarray:
        angle = np.degrees(np.abs(np.arctan2(points[:, 0] - centre_x, points[:, 2] - centre_z)))
        # 0 deg is straight ahead, 90 the ears, 180 the back of the head.
        to_temple = np.clip((angle - 30.0) / 45.0, 0.0, 1.0)
        to_back = np.clip((angle - 105.0) / 75.0, 0.0, 1.0)
        smooth = lambda t: t * t * (3.0 - 2.0 * t)
        height = forehead + (above_ears - forehead) * smooth(to_temple)
        return height + (nape - above_ears) * smooth(to_back)
    return line


def hairline_on_head(points, landmarks, photo_size, sheet: np.ndarray):
    """The photo's hairline (read_hair's `hairline_points`, photo pixels) on the
    head, as (xs, ys) in head units, or None.

    Through the placed landmark sheet: an affine from the photo to the sheet,
    fitted over the upper face, which the hairline sits just above.
    """
    if not points or len(points) < 4:
        return None
    pw, ph = photo_size
    pixels = np.array([[lm.x * pw, lm.y * ph] for lm in landmarks[:len(sheet)]])
    upper = sheet[:, 1] > np.percentile(sheet[:, 1], 66)
    fit = np.linalg.lstsq(np.c_[pixels[upper], np.ones(upper.sum())], sheet[upper, :2], rcond=None)[0]
    on_head = np.c_[np.asarray(points, dtype=np.float64), np.ones(len(points))] @ fit
    on_head = on_head[np.argsort(on_head[:, 0])]
    return on_head[:, 0], on_head[:, 1]


def brows_to_photo(card: trimesh.Trimesh, sheet: np.ndarray, depth: np.ndarray,
                   vertices: np.ndarray, groups: dict) -> trimesh.Trimesh:
    """Move the brow cards onto the photo's own brows, which the bake paints on.

    The cards hang from the template's brow, which MakeHuman sits lower than
    MediaPipe does: on the average face 3.6 mm below the middle of MediaPipe's
    brow for a woman, 9 mm for a man. The shape fit cannot close that, since
    it fits how a face differs from average, so the cards drew a second, thin
    brow under the photo's darker one. Each card's centre line is bent onto the
    line where the photo's brow is darkest, `depth` of the way down its
    landmark contours (facetexture.brow_depth), as quadratics in x so it takes
    the arch too. Then it is brought back in depth to its own height off the
    skin: the brow ridge bulges above the template's brow, and a card moved
    straight up sank into it.
    """
    from scipy.spatial import cKDTree

    # BROW_LANDMARKS is right then left, upper contour then lower, outer to inner.
    upper, lower = np.moveaxis(sheet[np.array(facetexture.BROW_LANDMARKS).reshape(2, 2, 5)], 1, 0)
    brows = upper + np.asarray(depth, dtype=np.float64)[:, None, None] * (lower - upper)
    skin = body_mesh(vertices)
    tree = cKDTree(skin.vertices)

    def off_skin(p):
        """Height above the skin, on the plane of the nearest skin vertex."""
        i = tree.query(p)[1]
        return ((p - skin.vertices[i]) * skin.vertex_normals[i]).sum(axis=1), skin.vertex_normals[i]

    v = card.vertices.copy()
    height, _ = off_skin(v)
    for line, own in zip(brows, (v[:, 0] < 0, v[:, 0] > 0)):   # +x is the subject's left
        x = v[own, 0]
        v[own, 1] += (np.polyval(np.polyfit(line[:, 0], line[:, 1], 2), x)
                      - np.polyval(np.polyfit(x, v[own, 1], 2), x))
    # Back onto the skin along the line of sight only, so the front view keeps
    # the placement. Where the tail wraps round the temple the skin faces
    # sideways and depth barely matters, hence the floor on the normal's z.
    for _ in range(4):
        now, normal = off_skin(v)
        v[:, 2] -= (now - height) / np.maximum(normal[:, 2], 0.3)
    card.vertices = v
    return card


def gltf_uv(uv):
    return np.column_stack([uv[:, 0], 1.0 - uv[:, 1]]).astype(np.float32)


def island_uv(uv_all: np.ndarray, faces_uv: np.ndarray) -> np.ndarray:
    """Stretch an island's texture coordinates to fill the unit square."""
    idx = np.unique(faces_uv)
    lo, hi = uv_all[idx].min(axis=0), uv_all[idx].max(axis=0)
    remapped = uv_all.copy()
    remapped[idx] = (uv_all[idx] - lo) / np.maximum(hi - lo, 1e-9)
    return remapped


def rim_colour(texture: Image.Image, faces: np.ndarray, uv: np.ndarray, uv_faces: np.ndarray,
               rims: list[int]) -> np.ndarray:
    """The texture's median colour (OpenCV LAB) where it meets the vertices
    `rims`: the lips' as the head was baked, for their inner side."""
    rgb = np.asarray(texture.convert("RGB"))
    size = rgb.shape[0]
    corners = np.isin(faces, rims)
    at = uv[uv_faces[corners]]
    px = np.clip(np.round(at[:, 0] * (size - 1)).astype(int), 0, size - 1)
    py = np.clip(np.round((1.0 - at[:, 1]) * (size - 1)).astype(int), 0, size - 1)
    return np.median(cv2.cvtColor(rgb[py, px][None], cv2.COLOR_RGB2LAB)[0].astype(np.float64), axis=0)


def head_island(vertices: np.ndarray, groups: dict, faces: np.ndarray,
                uv_faces: np.ndarray) -> np.ndarray:
    """Which body triangles make up the head's UV island, found from the nose
    by connectivity."""
    used = np.unique(faces)
    nose_vertex = used[np.linalg.norm(
        vertices[used] - facetexture.head_features(vertices, groups)["nose"], axis=1).argmin()]
    nose_uv = uv_faces.reshape(-1)[np.where(faces.reshape(-1) == nose_vertex)[0][0]]
    return facetexture.uv_island(uv_faces, nose_uv)


def build_eyes(vertices: np.ndarray, groups: dict, iris: np.ndarray):
    """Generated eyeballs with this iris colour: (mesh, uv, texture).

    Painting the photo's eye region onto the eyeball put eyelid and brow on a
    sphere, which reads as skin, not an eye.
    """
    uv_all, uv_groups = load_base_uv()
    eye_faces = np.vstack([groups["helper-l-eye"], groups["helper-r-eye"]])
    eye_uv_faces = np.vstack([uv_groups["helper-l-eye"], uv_groups["helper-r-eye"]])
    eye_uv_all = island_uv(uv_all, eye_uv_faces)
    eye_centres = np.array([vertices[np.unique(groups["helper-l-eye"])].mean(axis=0),
                            vertices[np.unique(groups["helper-r-eye"])].mean(axis=0)])
    texture = facetexture.bake_eyes(vertices, eye_faces, eye_uv_all, eye_uv_faces,
                                    eye_centres, iris)
    # Written darker by the renderer's lift, as the skin is (delight.RENDER_LIFT):
    # the whites rendered at 246 against the photos' 216, and the iris a
    # shade lighter than the colour read from the photo.
    texture = Image.fromarray(np.clip(delight.to_srgb(
        delight.to_linear(np.asarray(texture.convert("RGB"))) / delight.RENDER_LIFT), 0, 255).astype(np.uint8))
    eyes, eye_uv = submesh_uv(eye_faces, eye_uv_faces, vertices, uv=eye_uv_all)
    # Seat the eyeballs a few millimetres behind the lids. Where they sat level
    # with the lid surface they drew over it, the whole iris showed ringed by
    # white, and the avatar stared.
    eyes.vertices = eyes.vertices - np.array([0.0, 0.0, EYE_RECESS])
    return eyes, eye_uv, texture


def welded_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Vertex normals continuous across seams: a card mesh keeps a vertex per
    texture island, so two copies of one position sat on different cards with
    different normals, and the hair's shading stepped at every seam. The
    normals are taken from the mesh with coincident positions merged, and
    every copy gets its position's."""
    unique, inverse = np.unique(np.round(mesh.vertices, 5), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    welded = trimesh.Trimesh(unique, inverse[mesh.faces], process=False)
    return np.asarray(welded.vertex_normals)[inverse]        # copies share one normal


def rigged_part(name, mesh, roughness, vertices, groups, joints, weights, bone_count):
    """A catalogue piece, skinned to the rig.

    Cut garments look their weights up by the base-mesh vertex they came
    from. Imported ones have no such vertices and borrow weights from the
    body underneath them instead.
    """
    if "base_index" in mesh.metadata:
        source = mesh.metadata["base_index"]
        part_joints, part_weights = joints[source], weights[source]
    elif "mhclo_idx" in mesh.metadata:
        part_joints, part_weights = garments.mhclo_weights(
            mesh.metadata["mhclo_idx"], mesh.metadata["mhclo_blend"],
            joints, weights, bone_count)
    else:
        part_joints, part_weights = garments.weights_for(
            mesh, mesh.metadata.get("fit", "tight"), vertices, groups,
            joints, weights, bone_count,
        )
    if mesh.metadata.get("texture") is not None:
        # Authored asset: its own texture, normal map and alpha cut-out.
        # No vertex colour, which glTF would multiply into the texture.
        return skinned_glb.Part(
            name, mesh.vertices, mesh.faces, part_joints, part_weights,
            uv=gltf_uv(mesh.metadata["uv"]), texture=mesh.metadata["texture"],
            normal_texture=mesh.metadata.get("normal_texture"),
            alpha_mask=mesh.metadata.get("alpha_mask", False),
            normals=welded_normals(mesh), roughness=roughness,
        )
    return skinned_glb.Part(
        name, mesh.vertices, mesh.faces, part_joints, part_weights,
        colour=mesh.visual.vertex_colors.astype(np.float32) / 255.0,
        normals=mesh.vertex_normals, roughness=roughness,
    )


def outfit_for(gender: str, chosen: list[str] | None) -> list[dict]:
    """The wearables to put on, by id, falling back to the catalogue default."""
    index = items_by_id()
    ids = chosen or catalogue()["defaults"].get(gender, catalogue()["defaults"]["male"])
    missing = [i for i in ids if i not in index]
    if missing:
        raise KeyError(f"not in the catalogue: {', '.join(missing)}")
    return [index[i] for i in ids]


def compose(photo_path: pathlib.Path, out_path: pathlib.Path,
            gender: str = "male", wear: list[str] | None = None,
            face_patch: bool = False, pose: str = "relaxed",
            side_photos: list[pathlib.Path] | None = None,
            hair_hint: str | None = None, hair_colour_hint: str | None = None) -> dict:
    if not BODY_OBJ.exists():
        raise FileNotFoundError(
            f"{BODY_OBJ} is missing. See README.md 'Set up' for the body mesh download."
        )

    raw, groups = load_base()
    face, landmarks, result, photo = fit_face(photo_path)
    # Photos of the head turned to each side measure the depth one photo can
    # only guess: the face is fused from all of them (views.fuse), and the fit
    # trusts depth as far as it was measured.
    # A turned photo with no face found in it is left out, not the avatar.
    side_views = []
    for side in side_photos or []:
        try:
            side_views.append(views.view_of(side))
        except NoFaceFound:
            pass
    depth_trust = measured = None
    if side_views:
        fused, measured, cameras = views.fuse([views.view_of(photo_path)] + side_views)
        face = face.copy()
        face.vertices = fused
        # Not the lips: thin, moving, and read off a painted lip line, their
        # fused depth stood a centimetre proud on a rendered test head and
        # dragged mouth and chin forward with it.
        lips = morph.INNER_UPPER + morph.INNER_LOWER + morph.OUTER_UPPER + morph.OUTER_LOWER + [61, 291, 78, 308]
        measured[lips] = 0.0
        depth_trust = morph.DEPTH_TRUST + (1.0 - morph.DEPTH_TRUST) * measured

    # A man's or a woman's body first, then the face fitted on top of it.
    base = morph.body_for(raw, gender)

    # Shape the skull to this person before anything is built from it. The
    # person's face and MediaPipe's average face are placed on the head the same
    # way, and the targets are fitted to the difference, so placement error
    # cancels instead of being baked into the shape.
    average = place_face(morph.canonical_face(), base, groups).vertices
    person = place_face(face, base, groups).vertices
    if side_views:
        # How the fused face was put on the skull, to project the head back
        # into the turned photos for painting (views.paint_sides).
        to_fused = np.ptp(person[:, 0]) / np.ptp(fused[:, 0])
        to_fused = (to_fused, person.mean(axis=0) - to_fused * fused.mean(axis=0))
    corr = morph.correspondences(base, groups["body"], average[:468])
    # The photo's expression is fitted alongside, so a smile is not taken for
    # a wide mouth, and the mesh is built from identity alone: a resting face.
    front = morph.front_of(base, groups["body"], corr)
    pw, ph = photo.size
    photo_rgb = np.asarray(photo.convert("RGB"))
    cheeks = np.array([[landmarks[i].x * pw, landmarks[i].y * ph]
                       for i in SKIN_LANDMARKS if i < len(landmarks)])
    # A jaw outline drawn round a beard is not the jaw, and a laugh spreads the
    # mouth and nostrils: trust those landmarks less.
    outline_trust, jaw_beard = appearance.outline_trust(photo_rgb, landmarks, cheeks,
                                                        expression_of(result))
    shape_weights, shape_names, expression = morph.fit(base, corr, person, average,
                                                       expression_of(result), front,
                                                       outline_trust, depth=depth_trust)
    shape_weights = morph.lip_volume(shape_weights, shape_names, base, corr, person)
    vertices = morph.apply(base, shape_weights)
    shape_fit = morph.explained(base, corr, person, average, shape_weights, expression, front)
    # The lips' and eyelids' shape the targets could not reach, free-form.
    # Only those: corrected everywhere, the nose followed MediaPipe's own nose
    # points, which do not sit on the mesh's, and came out further from the
    # photo on every test face, and the outline's points slide with the
    # silhouette. The lids: the mesh's eyes rendered 1-4% of the eye spacing
    # more open than every test photo's.
    lip_marks = np.zeros(len(corr))
    lip_marks[list(appearance.EYE_CONTOURS[0]) + list(appearance.EYE_CONTOURS[1])] = 1.0
    lip_marks = np.asarray(outline_trust, dtype=np.float64)[:len(corr)] * lip_marks
    # The lips at full trust, on the vertices where the mesh's lips meet:
    # trusted as little as a laugh's other landmarks (0.4 on ref-man), the
    # mouth kept the skull's shape. The expression taken off them first is
    # still the photo's, so a laugh's stretch does not stay.
    lip_marks[morph.LIP_LANDMARKS] = 1.0
    # Towards the lips closed: the mesh's met 1.4% of the eye spacing apart on
    # a photo whose own were 0.4 apart, and a sliver of teeth showed.
    vertices = morph.correct(vertices, base, morph.lip_corr(base, corr), morph.close_lips(person), average,
                             shape_weights, expression, front, lip_marks)
    vertices = morph.seal_lips(vertices, groups["body"])
    # The jaw's edge as the camera saw it, with the photo's expression off.
    resting = person[:len(corr)] - morph.expression_offsets(base, front, expression)
    u, _, vt = np.linalg.svd(np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)[:3, :3])
    rotation = u @ vt
    # And as the photo-pose render turns it.
    head_pose = photo_head_pose(result, landmarks, photo)
    render_turn = rotation if head_pose is None else Rotation.from_quat(head_pose).as_matrix()
    head_only = morph.head_only(vertices, groups["body"], load_weights())
    vertices = morph.fit_outline(vertices, head_only, resting, rotation)

    # The same expression taken off the landmarks the photo is baked through,
    # and the lips closed, so the photo's smile is warped back onto a relaxed
    # mouth. Done in the face's own frame, before it is placed, so it is placed
    # by resting anatomy.
    scale = np.ptp(person[:, 0]) / np.ptp(face.vertices[:, 0])
    relax = np.zeros_like(face.vertices)
    relax[:len(corr)] = morph.expression_offsets(base, front, expression) / scale
    face = face.copy()
    face.vertices = morph.close_lips(face.vertices - relax)

    # The skin is read and baked from the photo with its lighting divided out
    # and its exposure capped; hair, brow and iris colour still come from the
    # photo as shot.
    # Everything measured from the face's skin is measured without its beard: a
    # beard is a third of some faces, read as shadow by the lighting fit and
    # as darker skin by the skin statistics.
    facial_hair = facetexture.facial_hair(photo_rgb, landmarks, cheeks)
    skin_rgb, lighting, face_mask = delight.delight(photo_rgb, landmarks, load_canonical_faces(), cheeks,
                                                    exclude=facial_hair)
    bare_skin = face_mask & (facial_hair < 0.2)
    skin_rgb = appearance.sharpen_creases(skin_rgb, landmarks, face_mask, keep_out=facial_hair)
    # The mouth is closed on the avatar; paint the photo's open mouth over in
    # lip colour so no teeth get baked across the lips.
    skin_rgb, mouth_filled = appearance.fill_mouth(skin_rgb, landmarks, cheeks)
    skin_rgb = appearance.fill_eyes(skin_rgb, landmarks)
    # The lids lighter than the brow's shade the photo shows over them.
    skin_rgb = appearance.lift_lids(skin_rgb, landmarks)
    # The nostrils are the mesh's own.
    skin_rgb = appearance.fill_nostrils(skin_rgb, landmarks, keep_out=facial_hair)
    # The lash line back on the lids. The brows are left as the photo has them:
    # darkened against the skin round them (tighten_brows, gain 1.6), they
    # rendered 14-34 L* darker than the photos' and the lid under them went
    # dark with them; without it they render within 9 L* of the photos' and
    # the identity score rose 0.445 -> 0.471 at the photo's pose, 0.410 ->
    # 0.446 level (side-session-notes/eye-region.md).
    skin_rgb = appearance.paint_lashes(skin_rgb, photo_rgb, landmarks)
    # A colour cast that puts skin outside the natural range is pulled back.
    skin_rgb, cast_shift = appearance.correct_cast(skin_rgb, bare_skin)
    # The lips darker and less yellow by what the render adds to them.
    skin_rgb = appearance.lip_tone(skin_rgb, landmarks)
    # The beard darker by what the studio's light lifts dark paint.
    skin_rgb = appearance.darken_beard(skin_rgb, facial_hair, landmarks)
    tone = facetexture.skin_tone_lab(skin_rgb, cheeks)
    full_face = place_face(face, vertices, groups)
    full_face.vertices = level_upper_face(full_face.vertices)
    # Onto the corners where the mesh's lips meet, not 3.6 mm inside them.
    full_face.vertices = mouth_onto_mesh(full_face.vertices, vertices, morph.lip_corr(base, corr))
    full_face.vertices = eyes_onto_mesh(full_face.vertices, vertices, corr)
    placed = trim_face_border(full_face)

    # The hair as the photo shows it, from a segmentation of the photo: its
    # colour, and — unless an outfit was chosen by hand — which haircut.
    parts = appearance.segment(photo_rgb)
    hair_reading = appearance.read_hair(photo_rgb, landmarks, parts)
    top_rgb = None
    outfit = outfit_for(gender, wear)
    if wear is None:
        default_cut = next((i["id"] for i in outfit if i["category"] == "hair" and "mhclo" in i), None)
        cut = appearance.hair_style(hair_reading, default_cut)
        # Where the segmentation read no hair to measure, the photo check's
        # haircut (photo_check.hair_hint: a catalogue id, or "bald") decides.
        if hair_hint and (hair_reading is None or hair_reading["top"] is None):
            cut = None if hair_hint == "bald" else hair_hint
        # Bald: no hair at all. Otherwise the chosen cut over the underlayer.
        rest = [i for i in outfit if i["category"] != "hair"]
        # No scalp layer under a real haircut: the head texture is tinted in
        # the hair colour under the cards (hairline_tint), and the layer,
        # 0.055 off the skin, stood in front of cards that lie closer than
        # that at the back of the head: a flat bald patch.
        outfit = rest if cut is None else rest + [items_by_id()[cut]]
        # The default top dyed the colour of the one in the photo.
        top_rgb = appearance.clothing_colour(photo_rgb, landmarks, parts)
        if top_rgb is not None:
            dye = delight.albedo_for(top_rgb)
            outfit = [{**i, "recolour": {**i["recolour"], "top": [int(c) for c in dye]}}
                      if i.get("recolour") else i for i in outfit]

    # Teeth and tongue go in every mouth, whatever is worn: hidden at rest,
    # and what an opened jaw shows.
    outfit = outfit + [items_by_id()[i] for i in ("teeth", "tongue") if i not in {o["id"] for o in outfit}]

    worn, hair_rgb, curve, hair_comb = [], None, None, None
    brow_depth = facetexture.brow_depth(photo_rgb, landmarks)
    for item in outfit:
        colour = None
        if item.get("colour") == "from_photo":
            if hair_rgb is None:
                read = hair_reading["colour"] if hair_reading else None
                hair_rgb = read if read is not None else hair_colour(photo, landmarks, cheeks)
                # The photo check's colour name (photo_check.hair_colour_hint),
                # where the page ran one, bounds how light the reading may be.
                if hair_colour_hint:
                    hair_rgb = photo_check.within_colour(hair_rgb, hair_colour_hint)
            colour = np.clip(dye_colour(hair_rgb) * float(item.get("colour_scale", 1.0)), 0, 255).astype(np.uint8)
        elif item.get("colour") == "from_brows":
            colour = facetexture.brow_colour(photo_rgb, landmarks)
        if item["category"] == "hair" and "group" in item:
            # The skull-cap cut: give it a hairline that follows the head.
            item = {**item, "hairline": hairline_for(placed, vertices, groups,
                                                     float(item.get("hairline_lift", 0.0)))}
            item.pop("y_min", None)
        mesh = build_item(item, colour, vertices)
        if item["category"] == "hair" and "mhclo" in item and parts is not None and hair_reading:
            # Combed the way the photo's hair runs, before it is shaped to its outline.
            hair_comb = garments.comb_hair(mesh, vertices, head_only, render_turn, person,
                                           np.array([[lm.x * pw, lm.y * ph] for lm in landmarks]),
                                           photo_rgb, parts == appearance.HAIR)
        if item["category"] == "hair" and "mhclo" in item and hair_reading and hair_reading.get("hairline"):
            # The fringe up to where the photo's hair ends, in eye spacings
            # above the brows on the shaped head.
            spacing = float(np.linalg.norm(vertices[corr[33], :2] - vertices[corr[263], :2]))
            hairline_y = float(vertices[corr[[105, 334]], 1].mean()) + hair_reading["hairline"] * spacing
            curve = hairline_on_head(hair_reading.get("hairline_points"), landmarks, photo.size, full_face.vertices)
            garments.sweep_back(mesh, vertices, groups, appearance.opaque_hair(mesh),
                                hairline_y if curve is None else curve, above=float(vertices[corr[[105, 334]], 1].mean()))
        if item["category"] == "hair" and "mhclo" in item and hair_reading and hair_reading.get("rise"):
            # And up to the photo's height over the forehead: the stock cuts lie flat.
            spacing = float(np.linalg.norm(vertices[corr[33], :2] - vertices[corr[263], :2]))
            rise_y = float(vertices[corr[[105, 334]], 1].mean()) + hair_reading["rise"] * spacing
            garments.volumise(mesh, vertices, groups, rise_y)
            # And out to the photo's outline at the sides.
            garments.widen(mesh, vertices, groups, float(vertices[corr[[105, 334]], 1].mean()), spacing,
                           hair_reading.get("widths"))
        if item["category"] == "hair" and "mhclo" in item and parts is not None and hair_reading:
            # And its outline to the photo's, seen as the photo saw the head.
            garments.fit_hair(mesh, vertices, head_only, render_turn, person,
                              np.array([[lm.x * pw, lm.y * ph] for lm in landmarks]),
                              parts == appearance.HAIR, appearance.opaque_hair)
        if item["category"] == "brows":
            mesh = brows_to_photo(mesh, full_face.vertices, brow_depth, vertices, groups)
        if item["category"] in ("teeth", "tongue"):
            # Set back from the lips: the model's resting lips leave a slit,
            # and teeth 1 to 3 mm behind it showed as a white line across a
            # closed mouth. An opened jaw still shows them.
            mesh.vertices = mesh.vertices - np.array([0.0, 0.0, TEETH_RECESS])
        if item["category"] == "teeth":
            # The upper teeth: those hung from the upper teeth helper, so an
            # overbite does not split a tooth.
            upper = np.isin(mesh.metadata["mhclo_idx"], np.unique(groups["helper-upper-teeth"])).all(axis=1)
            v = np.asarray(mesh.vertices, dtype=np.float64).copy()
            top = v[upper]
            # Each tooth's own biting edge: the lowest of the upper teeth within
            # 2 mm across; one edge for all lifted the incisors' with the canines'.
            across = np.abs(top[:, None, 0] - top[None, :, 0]) < 0.02
            edge = np.where(across, top[None, :, 1], np.inf).min(axis=1)
            v[upper, 1] += TEETH_RAISE * np.clip((top[:, 1] - edge) / TEETH_CROWN, 0.0, 1.0)
            mesh.vertices = v
        worn.append((item, mesh))

    # Trousers down to the ankle make the shoes' socks redundant, and posed they
    # poke through the fabric; drop them.
    ankle = float(vertices[np.unique(groups["joint-l-ankle"])][:, 1].mean())
    if any(item["category"] in ("outfit", "bottom") and m.vertices[:, 1].min() < ankle + 0.3
           for item, m in worn):
        for item, m in worn:
            if item["category"] == "shoes" and "mhclo" in item:
                garments.drop_socks(m, ankle)

    # Only clothing hides skin; hair sits on top of it.
    covering = [(item, m) for item, m in worn if item["category"] in ("top", "bottom", "shoes", "outfit")]
    # Every body triangle is kept; skin under clothes is hidden by the texture's
    # alpha instead, with a band left under each garment edge (coverage_alpha).
    body_faces = groups["body"]
    skin_alpha = np.ones(len(vertices))
    body_vertices_used = np.unique(body_faces)
    skin_alpha[body_vertices_used] = coverage_alpha(vertices[body_vertices_used],
                                                    [m for _, m in covering])
    for _, m in covering:
        # MakeHuman garments list the body vertices their author made sure
        # are under cloth; hide those too — but not within 3 cm of the
        # garment's edge. On a body shaped to a person and posed, some of the
        # listed neck sits just in front of the collar, and hiding it cut
        # windows in the neck through to the inside of the collar: green
        # patches below the jaw. Near an edge, coverage_alpha decides.
        hides = m.metadata.get("hides")
        if hides is not None and len(hides):
            from scipy.spatial import cKDTree
            hides = hides[hides < len(skin_alpha)]
            rim = m.vertices[garments.open_rim(m)]
            if len(rim):
                hides = hides[cKDTree(rim).query(vertices[hides])[0] > 0.3]
            skin_alpha[hides] = 0.0
    # The neck is never cut away. A collar wraps it but never covers it from
    # the front, and cutting it where a collar passes close behind left holes
    # under the ears through which the collar's inside showed. Skin poking
    # through a collar on a hard head turn is the smaller fault.
    collarbone = float(vertices[np.unique(groups["joint-l-clavicle"])][:, 1].mean())
    neck = (vertices[:, 1] > collarbone + 0.05) & (np.abs(vertices[:, 0]) < 0.75)
    skin_alpha[neck] = 1.0
    # A foot inside a shoe is never seen, and where it is fuller than the shoe
    # it came through: a toe under the sole. Below the collar's lowest point
    # the shoe encloses the foot, so the skin there is hidden outright; the
    # distance test above kept most of it, being near the sole's open edges.
    for item, m in worn:
        if item["category"] == "shoes":
            skin_alpha[garments.enclosed_by_shoe(m, vertices)] = 0.0
    uv_all, uv_groups = load_base_uv()
    uv_faces = uv_groups["body"]

    # Dense landmark sampler: every point on the head is looked up through the
    # anatomically placed landmark sheet, so the photo's mouth lands on the
    # skull's mouth and its chin on the skull's chin.
    landmark_pixels = np.array([[lm.x * pw, lm.y * ph] for lm in landmarks])
    rim_edges, rim_counts = np.unique(full_face.edges_sorted, axis=0, return_counts=True)
    rim = np.unique(rim_edges[rim_counts == 1])
    beard_map = facetexture.facial_hair(skin_rgb, landmarks, cheeks)
    # The forehead up to the hairline, which the landmarks stop short of.
    forehead = appearance.forehead_skin(parts, landmarks)
    sampler = facetexture.FaceSampler(full_face.vertices, landmark_pixels, rim,
                                      keep=beard_map,
                                      lips=([78] + morph.INNER_UPPER + [308],
                                            [78] + morph.INNER_LOWER + [308]),
                                      beyond=forehead, lip_outline=appearance.OUTER_LIPS)
    face_centre = full_face.vertices.mean(axis=0)
    face_radius = float(np.linalg.norm(full_face.vertices - face_centre, axis=1).max())

    # The head gets a texture of its own. In the shared body atlas the whole face
    # is about 130 pixels across, which is why fine features came out blurred
    # however carefully they were placed. Its UV island is found by connectivity
    # and stretched to fill a 1024 map, around eight times the texels.
    in_head = head_island(vertices, groups, body_faces, uv_faces)
    head_faces, head_uv_faces = body_faces[in_head], uv_faces[in_head]
    rest_faces, rest_uv_faces = body_faces[~in_head], uv_faces[~in_head]
    head_uv_all = island_uv(uv_all, head_uv_faces)
    # The mouth's inside, set back into the mouth from the lips it lines,
    # fully from MOUTH_DEPTH in from the rims: sealed, the lining met the
    # lips' skin at the seam, and a smile pressed on ref-man's turned head
    # showed it through his lower lip.
    mouth = appearance.mouth_inside(vertices, rest_faces, morph.UPPER_RIM + morph.LOWER_RIM)
    lining = trimesh.Trimesh(vertices, rest_faces, process=False).vertex_normals
    vertices = vertices + MOUTH_LINING * np.clip(2.0 * mouth - 1.0, 0.0, 1.0)[:, None] * lining

    # A skin texture built to the person's own cheek statistics, so skin is
    # mottled rather than one flat colour.
    stats = skin.face_stats(skin_rgb, bare_skin)
    skin_mean, skin_spread = stats if stats is not None else skin.cheek_stats(skin_rgb, cheeks)
    # The pore map stays 1024: it is noise, and at 2048 it cost 885 KB.
    skin_normals = Image.fromarray(skin.pore_normals(1024))

    def finish(texture_rgb, base, faces_, uv_all_, uv_faces_):
        """Cut skin under clothes away with alpha, flattening what is hidden.

        A part nothing covers keeps a plain RGB texture, which goes out as JPEG
        at a fraction of the size of the PNG an alpha channel forces.
        """
        alpha = facetexture.rasterize(skin_alpha, faces_, uv_all_, uv_faces_, texture_rgb.size[0])
        if alpha.min() >= 0.5:
            return texture_rgb.convert("RGB")
        rgb = np.asarray(texture_rgb.convert("RGB")).copy()
        rgb[alpha < 0.5] = np.asarray(base).reshape(-1, 3).mean(axis=0).astype(np.uint8)
        out = Image.fromarray(rgb).convert("RGBA")
        out.putalpha(Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8)))
        return out

    # Face feathered in rather than Poisson-blended: Poisson solves the face's
    # colours to match the generated skin around it, which dragged the whole
    # face towards that tone and washed the likeness out.
    # Always 2048. The photo covers only the face; ears, scalp, neck, the
    # carried beard and the hairline are generated, and at 1024 they were
    # soft even when the photo itself had nothing more to give.
    head_size = SKIN_SIZE
    head_base = skin.albedo(head_size, skin_mean, skin_spread, seed=3)
    # The head stays opaque. Its only covered skin is the top of the neck under
    # a collar, which the collar hides anyway, and cutting it out forced the
    # head texture to PNG: 1.4 MB at 2048 instead of a few hundred KB as JPEG.
    head_texture = facetexture.bake(vertices, head_faces, head_uv_all, head_uv_faces,
                                    sampler, face_centre, face_radius, skin_rgb, head_base,
                                    size=head_size, blend="none")
    # A beard's hairs are finer than the photo's pixels: grain where it was.
    head_texture = appearance.stubble(head_texture, vertices, head_faces, head_uv_all, head_uv_faces,
                                      sampler, skin_rgb, cheeks, jaw_beard, landmarks)
    # The haircut's cards if there are any, else a buzz cut's scalp layer.
    hair_meshes = [(("mhclo" in item), m) for item, m in worn if item["category"] == "hair"]
    haircut = max(hair_meshes, key=lambda pair: pair[0])[1] if hair_meshes else None
    if haircut is not None and hair_rgb is not None:
        # A soft hairline painted on the skin, so the hair cards do not end on
        # bare forehead in a hard line.
        head_texture, hair_cover, hair_below = appearance.hairline_tint(head_texture, vertices, head_faces, head_uv_all,
                                                            head_uv_faces, appearance.opaque_hair(haircut), hair_rgb,
                                                            spare=sampler.sample(vertices)[1], return_weight=True,
                                                            hairline=curve)
    # The body's one map runs neck to toe; at 1024 a nail was 8 texels wide.
    rest_base = skin.albedo(SKIN_SIZE, skin_mean, skin_spread, seed=7)

    # Creases shaded from the mesh's own shape: ears, nostrils, eye corners,
    # knuckles. On the face only where the photo is not, as the photo already
    # carries its own.
    photo_strength = sampler.sample(vertices)[1]
    cavity = appearance.skin_cavity(vertices, body_faces)
    shade = lambda image, faces_, uv_, uv_faces_, values: np.clip(
        np.asarray(image, dtype=np.float64) * (1.0 - CAVITY_DEPTH * facetexture.rasterize(
            values, faces_, uv_, uv_faces_, np.asarray(image).shape[0], fill=0.0))[..., None], 0, 255
    ).astype(np.uint8)
    head_texture = Image.fromarray(shade(head_texture.convert("RGB"), head_faces, head_uv_all, head_uv_faces,
                                         cavity * (1.0 - photo_strength)))
    # Ears: warmer and redder than the skull's generated skin.
    ears = appearance.ear_weight(len(vertices)) * (1.0 - photo_strength)
    head_texture = Image.fromarray(appearance.warm_ears(
        np.asarray(head_texture), facetexture.rasterize(ears, head_faces, head_uv_all, head_uv_faces,
                                                        head_texture.size[0], fill=0.0)))
    rest_base = shade(rest_base, rest_faces, uv_all, rest_uv_faces, cavity)
    if side_views:
        # The cheeks' sides and the jaw towards the ear, which the front
        # photo never saw, from the turned photos. Not the ears: no landmark
        # finds them, and a few millimetres' misfit painted the photo's ear
        # half beside the mesh's.
        head_texture, _ = views.paint_sides(head_texture, vertices, head_faces, head_uv_all, head_uv_faces,
                                            photo_strength, to_fused,
                                            [(view[2], camera) for view, camera in zip(side_views, cameras[1:])],
                                            skip=appearance.ear_weight(len(vertices)))

    # Fingernails: the base mesh's fingertips are smooth skin to the end. The
    # nail is found from the rig (appearance.nail_weight), painted lighter and
    # pinker with a faint fold, and given a glossier finish.
    skeleton = load_skeleton(vertices)
    joints, weights = weight_table(skeleton, len(vertices))
    nails = appearance.nail_weight(vertices, trimesh.Trimesh(vertices, body_faces, process=False).vertex_normals,
                                   skeleton, joints, weights)
    on_nail = nails[rest_faces].max(axis=1) > 0
    nail_map = facetexture.rasterize(nails, rest_faces[on_nail], uv_all, rest_uv_faces[on_nail],
                                     rest_base.shape[0], fill=0.0)
    rest_base = appearance.paint_nails(rest_base, nail_map)
    # The inside of the mouth, dark, but for the lips' wet inner side, which
    # a smile turns outwards: the colour the head was baked with at the rims.
    in_mouth = mouth[rest_faces].max(axis=1) > 0
    rest_base = appearance.paint_mouth(rest_base, facetexture.rasterize(
        mouth, rest_faces[in_mouth], uv_all, rest_uv_faces[in_mouth], rest_base.shape[0], fill=0.0),
        rim_colour(head_texture, head_faces, head_uv_all, head_uv_faces, morph.UPPER_RIM + morph.LOWER_RIM))
    body_rough = appearance.SKIN_ROUGHNESS["skin"] + (appearance.NAIL_ROUGHNESS - appearance.SKIN_ROUGHNESS["skin"]) * nail_map
    body_roughness = Image.fromarray(np.dstack([np.full_like(body_rough, 255), body_rough * 255,
                                                np.zeros_like(body_rough)]).astype(np.uint8))
    rest_texture = finish(Image.fromarray(rest_base), rest_base, rest_faces, uv_all, rest_uv_faces)

    # Roughness by region on the face: the T-zone oilier, the lips glossier,
    # cheeks matte; off the face, plain skin.
    regional = appearance.skin_roughness(landmarks, (pw, ph))
    pixels = sampler.sample(vertices)[0]
    at_photo = regional[np.clip(pixels[:, 1].astype(int), 0, ph - 1), np.clip(pixels[:, 0].astype(int), 0, pw - 1)]
    rough = appearance.SKIN_ROUGHNESS["skin"] * (1.0 - photo_strength) + at_photo * photo_strength
    rough_map = facetexture.rasterize(rough, head_faces, head_uv_all, head_uv_faces, 512,
                                      fill=appearance.SKIN_ROUGHNESS["skin"])
    head_roughness = Image.fromarray(np.dstack([np.full_like(rough_map, 255), rough_map * 255,
                                                np.zeros_like(rough_map)]).astype(np.uint8))

    head, head_uv = submesh_uv(head_faces, head_uv_faces, vertices, uv=head_uv_all)
    body, body_uv = submesh_uv(rest_faces, rest_uv_faces, vertices)
    # A full beard carried round under the jaw, where the photo never sees it,
    # and on down the neck, which is the body's texture: by the same rule on
    # both, so it does not stop on the seam between them.
    if jaw_beard > 0.4:
        photo_head = head_texture
        head_texture = appearance.beard_underside(head, head_uv, photo_head, sampler, beard_map,
                                                  landmarks, (pw, ph))
        rest_texture = appearance.beard_underside(head, head_uv, photo_head, sampler, beard_map,
                                                  landmarks, (pw, ph), onto=(body, body_uv, rest_texture))

    # Eyes: generated, with only the iris colour taken from the photo, judged
    # against the whites of the eyes, as a person sees it, and snapped halfway
    # to a natural eye colour (appearance.iris_colour).
    iris, eye_colour = appearance.iris_colour(photo_rgb, landmarks)
    eyes, eye_uv, eye_texture = build_eyes(vertices, groups, iris)

    # --- rig it --- (skeleton and weights made above, with the nails)

    parts = []
    # The lips' and cheeks' as the average face moves, so a laugh does not
    # turn the fitted lips inside out (morph.transfer_shapes).
    face_shapes = morph.transfer_shapes(base, vertices, groups["body"], corr, groups)
    photo_face = morph.photo_expression(expression_of(result))
    if mouth_filled:
        # The photo's open mouth was painted over as lips (fill_mouth): parted
        # as far as the photo's, its laugh shows the teeth (whitened, see
        # garments.whiten_teeth) where the photo does. A fixed jaw opening of
        # 0.25 did this before, from photo-10 (identity at the pose / level:
        # lips together 0.534 / 0.514, parted as MediaPipe scored them 0.527 /
        # 0.504, with that jaw 0.545 / 0.515), and showed too much of a wider
        # laugh.
        photo_face = open_as_photo(photo_face, vertices, face_shapes, morph.lip_corr(base, corr),
                                   landmarks, (pw, ph))
    # And the mouth then put on the photo's own (morph.photo_mouth), shown
    # with the photo's expression.
    on_skin = np.zeros(len(vertices), bool)
    on_skin[np.unique(groups["body"])] = True
    teeth = next(m for item, m in worn if item["category"] == "teeth")
    teeth_moves = expression_morphs(teeth, face_shapes)
    laughing_teeth = np.asarray(teeth.vertices, dtype=np.float64)
    if teeth_moves is not None:
        laughing_teeth = laughing_teeth + sum(w * teeth_moves[morph.ARKIT_NAMES.index(k)] for k, w in photo_face.items())
    face_shapes = np.concatenate([face_shapes, morph.photo_mouth(
        vertices, face_shapes, photo_face, morph.lip_corr(base, corr), person, on_skin, laughing_teeth)[None]])
    photo_face["photoMouth"] = 1.0
    # Normals from the whole skin, not from each part's own triangles: split
    # at the neck, each side averaged only its own, and light broke along the
    # seam in angular patches under the ear.
    skin_normal = trimesh.Trimesh(vertices, body_faces, process=False).vertex_normals
    # Under the brows the photo already carries its own shading, and
    # MakeHuman's eye socket is deeper than most faces': its skin below the
    # brow faces the floor, every studio light is above, and that band came out
    # twice as dark as the photo's (56 L* under the forehead against 25), read
    # as eyeshadow that swamped the brows. There the normals keep the face's
    # broad form only. Smoothing the whole face instead tipped cheeks and chin
    # towards the neck and darkened them.
    broad = appearance.broad_normals(body_faces, skin_normal, PHOTO_NORMAL_ROUNDS)
    socket = (appearance.under_brow(vertices, corr) * photo_strength)[:, None]
    skin_normal = skin_normal * (1.0 - socket) + broad * socket
    skin_normal /= np.maximum(np.linalg.norm(skin_normal, axis=1, keepdims=True), 1e-12)
    for name, mesh, uv, texture in (("head", head, head_uv, head_texture),
                                    ("body", body, body_uv, rest_texture)):
        source = mesh.metadata["base_index"]
        parts.append(skinned_glb.Part(
            name, mesh.vertices, mesh.faces, joints[source], weights[source],
            uv=gltf_uv(uv), texture=texture, normals=skin_normal[source],
            normal_texture=skin_normals, alpha_mask=texture.mode == "RGBA",
            roughness=appearance.SKIN_ROUGHNESS["skin"], normal_scale=PORE_STRENGTH[name],
        ))
        parts[-1].morphs = expression_morphs(mesh, face_shapes)
        # Skin reflects less than the glTF default (IOR 1.5): with it the
        # face wore a pale sheen of reflected room that read as grey.
        parts[-1].ior = SKIN_IOR
        # Light passes into skin and comes back out warm, brightest where the
        # surface turns away: glTF has no subsurface scattering, and without
        # it faces looked waxy. A faint red-orange sheen gives the same warm
        # edge glow at grazing angles.
        parts[-1].sheen = SKIN_SHEEN
        parts[-1].roughness_texture = head_roughness if name == "head" else body_roughness
    if haircut is not None and hair_rgb is not None:
        # A cap in the hair's root shade under the cards, opaque where the
        # tint is solid and dissolving into the same wisps below the hair's
        # edge. The lifted front of a swept-back cut stands off the forehead,
        # and between its strands the tinted skin showed, lit as skin: a
        # brown band under black hair. An opaque shell there reads as the
        # hair's shadowed underside, as a real hairline does.
        centres = (head_uv_all[head_uv_faces].mean(axis=1) * (hair_cover.shape[0] - 1))
        rows = np.clip((hair_cover.shape[0] - 1 - centres[:, 1]).astype(int), 0, hair_cover.shape[0] - 1)
        cols = np.clip(centres[:, 0].astype(int), 0, hair_cover.shape[1] - 1)
        # Above the brows only: lower, beside the ears, the cap showed as
        # dark patches where no card covers it.
        brow_line = float(vertices[corr[[105, 334]], 1].mean())
        under = (hair_cover[rows, cols] > CAP_CUT) & (vertices[head_faces, 1].min(axis=1) > brow_line)
        if under.sum() > 10:
            cap, cap_uv = submesh_uv(head_faces[under], head_uv_faces[under],
                                     vertices + CAP_OFFSET * skin_normal, uv=head_uv_all)
            shade = np.clip(dye_colour(hair_rgb) * 0.6, 0, 255).astype(np.uint8)
            # Opaque where the tint is solid, gone where it has thinned to
            # wisps (CAP_FADE): cut at the tint's own fade it stood a hard,
            # dark edge below the hair, 16 L* a row against the photo's 11.
            # Blended, not cut (cut at half alpha, the wisps' breakup drew the
            # edge as a sawtooth), opaque above the hair's edge and fading
            # over CAP_FADE below it; the tint's own plateau reaches further
            # down, and a cap cut to it covered skin the photo shows between
            # the hairline's sparse hairs (the strip under the hair L* 85
            # against the photo's 137).
            cover = np.clip(1.0 - (np.nan_to_num(hair_below, nan=1.0) + CAP_RAISE) / CAP_FADE, 0.0, 1.0)
            cover = cv2.GaussianBlur(cover, (0, 0), 6)
            cover = Image.fromarray((cover * 255).astype(np.uint8)).resize((1024, 1024))
            cap_texture = Image.merge("RGBA", (*(Image.new("L", (1024, 1024), int(c)) for c in shade), cover))
            source = cap.metadata["base_index"]
            parts.append(skinned_glb.Part(
                "scalp_cap", cap.vertices, cap.faces, joints[source], weights[source],
                uv=gltf_uv(cap_uv), texture=cap_texture,
                normals=skin_normal[source], roughness=0.95,
            ))
            parts[-1].alpha_blend = True
            parts[-1].morphs = expression_morphs(cap, face_shapes)
    for item, mesh in worn:
        finish = item.get("roughness", ROUGHNESS.get(item["category"], 0.8))
        parts.append(rigged_part(item["id"], mesh, finish, vertices, groups,
                                 joints, weights, len(skeleton)))
        parts[-1].morphs = expression_morphs(mesh, face_shapes)
        if item["category"] in ("outfit", "top", "bottom"):
            # Cloth scatters light at grazing angles: a soft rim along folds
            # and edges that plain roughness does not give (KHR_materials_sheen).
            parts[-1].sheen = CLOTH_SHEEN
        if item["category"] == "hair" and "mhclo" in item:
            # Hair's highlight runs across the strands, a band rather than a
            # spot; the strands lie along the texture's v axis.
            parts[-1].anisotropy = HAIR_ANISOTROPY

    eye_base = eyes.metadata["base_index"]
    parts.append(skinned_glb.Part(
        "eyes", eyes.vertices, eyes.faces, joints[eye_base], weights[eye_base],
        uv=gltf_uv(eye_uv),
        texture=eye_texture, normals=eyes.vertex_normals,
        roughness=ROUGHNESS["eyes"],                 # wet, so the highlight shows
    ))
    parts[-1].clearcoat = CORNEA_COAT
    parts[-1].morphs = expression_morphs(eyes, face_shapes)

    if face_patch:
        # The fitted landmark patch laid over the head. Opt-in: once the skull
        # was shaped to the person it ended up behind the head's own surface
        # almost everywhere, and showed only as a hard edge on the forehead.
        head_bone = skeleton.index.get("head", 0)
        face_joints, face_weights = rigid_binding(len(placed.vertices), head_bone)
        face_uv = np.asarray(placed.visual.uv, dtype=np.float32).copy()
        face_uv[:, 1] = 1.0 - face_uv[:, 1]  # trimesh is bottom-up; glTF is top-down
        parts.append(skinned_glb.Part(
            "face", placed.vertices, placed.faces, face_joints, face_weights,
            uv=face_uv, texture=Image.fromarray(skin_rgb),
            normals=placed.vertex_normals, roughness=ROUGHNESS["skin"],
            # Same grain as the skin around it, a little lighter: the photo
            # already carries its own surface detail.
            normal_texture=skin_normals, normal_scale=0.6,
        ))

    # The arms are posed on the male base body; on a fuller figure they are
    # swung out until the hands clear the hips.
    body_part = next(p for p in parts if p.name == "body")
    skinned_glb.export(parts, skeleton, out_path, pose=arms_clear(POSES[pose](), skeleton, body_part),
                       morph_names=morph.ARKIT_NAMES + ["photoMouth"],
                       expression=photo_face,
                       head_pose=head_pose)

    info = {
        "advice": photo_advice(landmarks, (pw, ph), expression_of(result)),
        "source_photo": photo_path.name,
        "photo_size": list(photo.size),
        "kind": "full-body",
        "side_views": len(side_views),
        "shape": {
            "explained": round(shape_fit, 3),
            "active_targets": int((shape_weights > 0.05).sum()),
            "strongest": [
                [shape_names[i], round(float(shape_weights[i]), 2)]
                for i in np.argsort(-shape_weights)[:6] if shape_weights[i] > 0.05
            ],
        },
        "rigged": True,
        "pose": pose,
        "bones": len(skeleton),
        "gender": gender,
        "wearing": [item["id"] for item, _ in worn],
        # CC-BY items name their authors, to be credited where the avatar is shown.
        "credits": sorted({f"{item['label']}: {item['credit']} (CC-BY)" for item, _ in worn if item.get("credit")}),
        "skin_tone": [int(c) for c in tone],
        "hair_colour": None if hair_rgb is None else [int(c) for c in hair_rgb],
        "body_triangles": int(len(body.faces) + len(head.faces)),
        "face": "dense landmark bake + patch" if face_patch else "dense landmark bake",
        "iris_colour": [int(c) for c in iris],
        "eye_colour": eye_colour,
        "mouth_filled": mouth_filled,
        "top_colour": None if top_rgb is None else [int(c) for c in top_rgb],
        "jaw_beard": jaw_beard,
        "hair_comb": hair_comb,
        "cast_corrected": list(cast_shift),
        "hair_reading": None if hair_reading is None else
            {k: v for k, v in hair_reading.items() if k != "colour"},
        "lighting": lighting,
        "triangles": int(len(body.faces) + len(head.faces) + sum(len(m.faces) for _, m in worn)),
        "landmarks": len(landmarks),
        "size_kb": round(out_path.stat().st_size / 1024),
        "expression": expression_of(result),
        # What was taken off to give a resting face: MakeHuman expression units
        # and how far each was applied, 0-1.
        "expression_removed": {unit: round(w, 2) for unit, w in
                               sorted(expression.items(), key=lambda kv: -kv[1]) if w > 0.02},
    }
    # Same sidecar shape as a face-only run, so the studio gallery and the
    # review page do not care which kind of avatar this is.
    out_path.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="A dressed avatar from one photo.")
    parser.add_argument("photo", type=pathlib.Path)
    parser.add_argument("-o", "--out", type=pathlib.Path, required=True)
    parser.add_argument("--gender", choices=["male", "female"], default="male")
    parser.add_argument("--side", type=pathlib.Path, action="append", default=[],
                        help="a photo of the head turned to one side (30-45 degrees); give it once per side")
    parser.add_argument("--wear", help="comma-separated catalogue ids, overriding the default outfit")
    parser.add_argument("--face-patch", dest="face_patch", action="store_true",
                        help="also lay the fitted landmark patch over the head. Off by default: "
                             "since the skull is shaped to the face, the patch sits behind it "
                             "except for a hard edge across the forehead")
    parser.add_argument("--pose", choices=sorted(POSES), default="relaxed",
                        help="the stance the file opens in (the skeleton is the same)")
    parser.add_argument("--compat", action="store_true",
                        help="write normals and texture coordinates as floats, without "
                             "KHR_mesh_quantization, for old viewers such as Windows' 3D Viewer")
    args = parser.parse_args()

    try:
        info = compose(args.photo, args.out, args.gender,
                       args.wear.split(",") if args.wear else None, args.face_patch, args.pose,
                       side_photos=args.side or None)
    except NoFaceFound:
        print("no face found in the photo", file=sys.stderr)
        return 1
    except (FileNotFoundError, KeyError) as err:
        print(err, file=sys.stderr)
        return 2

    if args.compat:
        args.out.write_bytes(skinned_glb.dequantised(args.out.read_bytes()))
    print(f"{args.out}  {info['triangles']} tris  wearing {', '.join(info['wearing'])}"
          f"  {info['size_kb']} KB" + ("  (compat)" if args.compat else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
