
# mvn_vizard_gait_with_live_pitch.py
# Single merged Vizard script:
# 1) MVN MXTP02 UDP -> avatar gait drive
# 2) Live pitch UDP -> on-screen numeric display + scrolling graph

import viz
import viztask
import vizshape
import socket
import struct
import threading
import time
import math
import collections

# ---------------- Vizard world ----------------
viz.setMultiSample(4)
viz.go()
viz.clearcolor(0.92, 0.92, 0.95)

viz.MainView.setPosition([0, 1.7, -3.5], viz.ABS_GLOBAL)
viz.MainView.lookAt([0, 1.2, 2.0], viz.ABS_GLOBAL)

# Avatar
man = viz.addAvatar('Avatar/vcc_male.cfg')
man.setPosition(0, 0, 2.0)

ground = vizshape.addPlane(size=(8, 8), axis=vizshape.AXIS_Y)
ground.color(viz.GRAY)

axis = vizshape.addAxes(length=0.25)
axis.visible(viz.ON)

# ---------------- On-screen pitch display ----------------
pitch_text = viz.addText("Pitch: 0.0000", parent=viz.SCREEN)
pitch_text.setPosition(0.02, 0.92)
pitch_text.fontSize(22)
pitch_text.color(viz.BLACK)

phase_text = viz.addText("Phase: Unknown", parent=viz.SCREEN)
phase_text.setPosition(0.02, 0.88)
phase_text.fontSize(18)
phase_text.color(viz.BLUE)

# Graph background
graph_origin = [0.02, 0.72]
graph_width = 0.32
graph_height = 0.14
graph_points = []

graph_bg = viz.addTexQuad(parent=viz.SCREEN)
graph_bg.setPosition(graph_origin[0] + graph_width/2, graph_origin[1] + graph_height/2)
graph_bg.setScale(graph_width, graph_height, 1)
graph_bg.color(0.92, 0.92, 0.92)
graph_bg.alpha(0.7)

zero_line = viz.addTexQuad(parent=viz.SCREEN)
zero_line.setPosition(graph_origin[0] + graph_width/2, graph_origin[1] + graph_height/2)
zero_line.setScale(graph_width, 0.002, 1)
zero_line.color(viz.BLACK)
zero_line.alpha(0.8)

# ---------------- Segment -> Bone mapping ----------------
SEG_TO_BONE = {
    'Pelvis':'Bip01 Pelvis',
    'L5':'Bip01 Spine',
    'L3':'Bip01 Spine1',
    'T12':'Bip01 Spine2',
    'T8':'Bip01 Spine2',
    'Neck':'Bip01 Neck',
    'Head':'Bip01 Head',

    'RightShoulder':'Bip01 R Clavicle',
    'RightUpperArm':'Bip01 R UpperArm',
    'RightForeArm':'Bip01 R Forearm',
    'RightHand':'Bip01 R Hand',

    'LeftShoulder':'Bip01 L Clavicle',
    'LeftUpperArm':'Bip01 L UpperArm',
    'LeftForeArm':'Bip01 L Forearm',
    'LeftHand':'Bip01 L Hand',

    'RightUpLeg':'Bip01 R Thigh',
    'RightLeg':'Bip01 R Calf',
    'RightFoot':'Bip01 R Foot',

    'LeftUpLeg':'Bip01 L Thigh',
    'LeftLeg':'Bip01 L Calf',
    'LeftFoot':'Bip01 L Foot',
}

PARENT = {
    'Pelvis': None,
    'L5':'Pelvis',
    'L3':'L5',
    'T12':'L3',
    'T8':'T12',
    'Neck':'T8',
    'Head':'Neck',

    'RightShoulder':'T8',
    'RightUpperArm':'RightShoulder',
    'RightForeArm':'RightUpperArm',
    'RightHand':'RightForeArm',

    'LeftShoulder':'T8',
    'LeftUpperArm':'LeftShoulder',
    'LeftForeArm':'LeftUpperArm',
    'LeftHand':'LeftForeArm',

    'RightUpLeg':'Pelvis',
    'RightLeg':'RightUpLeg',
    'RightFoot':'RightLeg',

    'LeftUpLeg':'Pelvis',
    'LeftLeg':'LeftUpLeg',
    'LeftFoot':'LeftLeg',
}

bones = {}
for seg, bname in SEG_TO_BONE.items():
    try:
        b = man.getBone(bname)
    except Exception:
        b = None
    if b:
        bones[seg] = b
        b.lock()

# ---------------- MXTP02 network ----------------
MVN_UDP_PORT = 9763
PITCH_UDP_PORT = 5055

HEADER_FMT   = '>6s I B B I B B B B H H'
ITEM_FMT_POS = '>i f f f f f f f'
ITEM_FMT_VEL = '>i f f f f f f f f f f'

SEG_NAMES = [
    'Pelvis','L5','L3','T12','T8','Neck','Head',
    'RightShoulder','RightUpperArm','RightForeArm','RightHand',
    'LeftShoulder','LeftUpperArm','LeftForeArm','LeftHand',
    'RightUpLeg','RightLeg','RightFoot','RightToe',
    'LeftUpLeg','LeftLeg','LeftFoot','LeftToe'
]

def seg_name_from_id(i):
    return SEG_NAMES[i-1] if 0 < i <= len(SEG_NAMES) else None

# ---------------- Shared state ----------------
latest_quat_global = {}
latest_pos_global = {}
_last_xyzw = {}
latest_root_vel = [0.0, 0.0, 0.0]

latest_pitch = 0.0
pitch_history = collections.deque(maxlen=120)

# ---------------- Mapping helpers ----------------
def map_quat_mvn_to_vizard(qw, qx, qy, qz):
    return viz.Quat(-qy, qz, qx, -qw)

def map_pos_mvn_to_vizard(px, py, pz):
    return [py, pz, -px]

def map_vec_mvn_to_vizard(vx, vy, vz):
    return [-vy, vz, vx]

# ---------------- Packet parser ----------------
def parse_mxtp02(data):
    if len(data) < 24:
        return None, {}, None, {}

    try:
        idstr, sc, dg, num_items, tc, char_id, nbody, nprops, nfingers, _r, _p = struct.unpack(
            HEADER_FMT, data[:24]
        )
    except struct.error:
        return None, {}, None, {}

    if not idstr.startswith(b'MXTP02'):
        return char_id, {}, None, {}

    offset = 24
    remain = len(data) - offset

    if num_items > 0:
        guess = remain // num_items
        item_sz = 44 if guess == 44 else 32
    else:
        item_sz = 44 if remain % 44 == 0 else 32

    out_quats = {}
    out_pos = {}
    pelvis_vel_viz = None

    count = remain // item_sz
    for i in range(count):
        s = offset + i * item_sz
        try:
            if item_sz == 44:
                seg_id, px, py, pz, qw, qx, qy, qz, vx, vy, vz = struct.unpack(
                    ITEM_FMT_VEL, data[s:s+item_sz]
                )
            else:
                seg_id, px, py, pz, qw, qx, qy, qz = struct.unpack(
                    ITEM_FMT_POS, data[s:s+item_sz]
                )
                vx = vy = vz = None
        except struct.error:
            continue

        name = seg_name_from_id(seg_id)
        if not name:
            continue

        S = map_quat_mvn_to_vizard(qw, qx, qy, qz)
        sx, sy, sz, sw = S.get()

        prev = _last_xyzw.get(name)
        if prev is not None and (sx*prev[0] + sy*prev[1] + sz*prev[2] + sw*prev[3]) < 0.0:
            sx, sy, sz, sw = -sx, -sy, -sz, -sw

        S = viz.Quat(sx, sy, sz, sw)
        _last_xyzw[name] = (sx, sy, sz, sw)
        out_quats[name] = S
        out_pos[name] = map_pos_mvn_to_vizard(px, py, pz)

        if seg_id == 1 and vx is not None:
            pelvis_vel_viz = map_vec_mvn_to_vizard(vx, vy, vz)

    return char_id, out_quats, pelvis_vel_viz, out_pos

# ---------------- UDP threads ----------------
def mvn_udp_loop():
    global latest_root_vel, latest_pos_global
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', MVN_UDP_PORT))
    viz.logNotice('Listening for MVN MXTP02 on {}.'.format(MVN_UDP_PORT))

    while True:
        data, _ = s.recvfrom(16384)
        char_id, qmap, vroot, pmap = parse_mxtp02(data)

        for seg, q in qmap.items():
            if seg in bones:
                latest_quat_global[seg] = q

        for seg, p in pmap.items():
            if seg in bones:
                latest_pos_global[seg] = p

        if vroot is not None:
            latest_root_vel = vroot

def pitch_udp_loop():
    global latest_pitch
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", PITCH_UDP_PORT))
    viz.logNotice("Listening for live pitch on UDP {}.".format(PITCH_UDP_PORT))

    while True:
        data, _ = s.recvfrom(1024)
        try:
            latest_pitch = float(data.decode("utf-8").strip())
            pitch_history.append(latest_pitch)
        except:
            pass

threading.Thread(target=mvn_udp_loop, daemon=True).start()
threading.Thread(target=pitch_udp_loop, daemon=True).start()

# ---------------- Optional bone-length scaling ----------------
def calibrate_bone_lengths():
    if not latest_pos_global:
        viz.logNotice('No MVN positions yet. Wait briefly and try again.')
        return

    from viz import Vector
    viz.logNotice('Calibrating bone lengths from MVN mannequin...')

    for seg, bone in bones.items():
        parent = PARENT.get(seg)
        if not parent or parent not in bones:
            continue
        if seg not in latest_pos_global or parent not in latest_pos_global:
            continue

        p_mvn_child = Vector(latest_pos_global[seg])
        p_mvn_parent = Vector(latest_pos_global[parent])
        L_mvn = (p_mvn_child - p_mvn_parent).length()

        p_viz_child = Vector(bone.getPosition(viz.ABS_GLOBAL))
        p_viz_parent = Vector(bones[parent].getPosition(viz.ABS_GLOBAL))
        L_viz = (p_viz_child - p_viz_parent).length()

        if L_viz > 1e-6 and L_mvn > 0:
            r = L_mvn / L_viz
            bone.setScale([r, r, r])

    viz.logNotice('Bone-length calibration complete.')

def onKeyDown(key):
    if key.lower() == 'c':
        calibrate_bone_lengths()

viz.callback(viz.KEYDOWN_EVENT, onKeyDown)

# ---------------- Pitch graph ----------------
def update_pitch_graph():
    global graph_points

    for g in graph_points:
        g.remove()
    graph_points = []

    if len(pitch_history) < 2:
        return

    vals = list(pitch_history)
    y_min, y_max = -0.05, 0.05

    for i, val in enumerate(vals):
        x = graph_origin[0] + graph_width * (i / max(1, len(vals)-1))
        y_norm = (max(y_min, min(y_max, val)) - y_min) / (y_max - y_min)
        y = graph_origin[1] + graph_height * y_norm

        p = viz.addTexQuad(parent=viz.SCREEN)
        p.setPosition(x, y)
        p.setScale(0.003, 0.006, 1)

        if abs(val) < 0.01:
            p.color(viz.GREEN)
        elif abs(val) < 0.03:
            p.color(viz.YELLOW)
        else:
            p.color(viz.RED)

        graph_points.append(p)

# ---------------- Optional stance/swing label ----------------
def estimate_phase_from_pitch(p):
    if abs(p) < 0.01:
        return "Stable / Stance-like"
    elif abs(p) < 0.03:
        return "Transition"
    else:
        return "Unstable / Swing-like"

# ---------------- Update loop ----------------
def update_avatar():
    last_fallback_time = None

    segment_order = [
        'Pelvis',
        'L5','L3','T12','T8','Neck','Head',
        'RightShoulder','RightUpperArm','RightForeArm','RightHand',
        'LeftShoulder','LeftUpperArm','LeftForeArm','LeftHand',
        'RightUpLeg','RightLeg','RightFoot',
        'LeftUpLeg','LeftLeg','LeftFoot'
    ]

    while True:
        now = time.time()

        # Root translation from pelvis position
        pelvis_pos = latest_pos_global.get('Pelvis')
        if pelvis_pos is not None:
            man.setPosition(pelvis_pos, viz.ABS_GLOBAL)
            axis.setPosition(pelvis_pos, viz.ABS_GLOBAL)
        else:
            if last_fallback_time is None:
                last_fallback_time = now
            dt = max(0.0, min(0.05, now - last_fallback_time))
            if dt > 0.0 and any(latest_root_vel):
                cx, cy, cz = man.getPosition(viz.ABS_GLOBAL)
                vx, vy, vz = latest_root_vel
                man.setPosition([cx + vx*dt, cy + vy*dt, cz + vz*dt], viz.ABS_GLOBAL)
                axis.setPosition(man.getPosition(viz.ABS_GLOBAL), viz.ABS_GLOBAL)
            last_fallback_time = now

        # Root orientation from pelvis quaternion
        S_pel = latest_quat_global.get('Pelvis')
        if S_pel is not None:
            man.setQuat(S_pel, viz.ABS_GLOBAL)

        # Local quaternions
        local_quat = {}
        for seg in segment_order:
            S = latest_quat_global.get(seg)
            if S is None:
                continue
            parent = PARENT.get(seg)
            if parent is None:
                local_quat[seg] = viz.Quat(0, 0, 0, 1)
            else:
                Sp = latest_quat_global.get(parent)
                if Sp is None:
                    continue
                local_quat[seg] = Sp.inverse() * S

        if 'Pelvis' in bones and 'Pelvis' in local_quat:
            bones['Pelvis'].setQuat(local_quat['Pelvis'], viz.AVATAR_LOCAL)

        for seg in segment_order:
            if seg == 'Pelvis':
                continue
            bone = bones.get(seg)
            if not bone:
                continue
            qloc = local_quat.get(seg)
            if qloc is not None:
                bone.setQuat(qloc, viz.AVATAR_LOCAL)

        # Live pitch display
        pitch_text.message("Pitch: {:+.4f} m/rad".format(latest_pitch))
        phase_text.message("Phase: {}".format(estimate_phase_from_pitch(latest_pitch)))
        update_pitch_graph()

        yield viztask.waitTime(0.016)

viztask.schedule(update_avatar)

viz.logNotice(
    'Merged gait+pitch system active. '
    'MVN on UDP 9763, pitch on UDP 5055. '
    'Press "c" once only if optional bone scaling is needed.'
)