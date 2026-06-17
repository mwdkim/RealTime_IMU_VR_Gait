# mvn_to_vizard_gait_clean.py
# Clean gait-only MVN -> Vizard direct drive
# - Pelvis world position from streamed pelvis position
# - Pelvis world orientation from streamed pelvis quaternion
# - Child bones from local quaternions (parent^-1 * child)
# - Optional bone-length scaling on key 'c'

import viz, viztask, vizshape
import socket, struct, threading, time, math

# ---------------- Vizard world & camera ----------------
viz.setMultiSample(4)
viz.go()
viz.clearcolor(0.92, 0.92, 0.95)

viz.MainView.setPosition([0, 1.7, -3.5], viz.ABS_GLOBAL)
viz.MainView.lookAt([0, 1.2, 2.0], viz.ABS_GLOBAL)

# Avatar & visualization
man = viz.addAvatar('Avatar/vcc_male.cfg')
man.setPosition(0, 0, 2.0)

ground = vizshape.addPlane(size=(8, 8), axis=vizshape.AXIS_Y)
ground.color(viz.GRAY)

axis = vizshape.addAxes(length=0.25)
axis.visible(viz.ON)

# ---------------- Segment -> Bone mapping ----------------
SEG_TO_BONE = {
    'Pelvis':'Bip01 Pelvis',
    'L5':'Bip01 Spine',
    'L3':'Bip01 Spine1',
    'T12':'Bip01 Spine2',
    'T8':'Bip01 Spine2',   # fixed from original duplicate mapping
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

# Collect and lock bones
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
UDP_PORT = 9763
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
latest_quat_global = {}   # seg -> viz.Quat in Vizard coords
latest_pos_global  = {}   # seg -> [x,y,z] in Vizard coords
_last_xyzw         = {}   # hemisphere continuity
latest_root_vel    = [0.0, 0.0, 0.0]  # kept for fallback only

# ---------------- Mapping helpers ----------------
# Keep your original working quaternion remap:
# (w,x,y,z) -> (-y, z, x, -w)
def map_quat_mvn_to_vizard(qw, qx, qy, qz):
    return viz.Quat(-qy, qz, qx, -qw)

# Keep your original position mapping
def map_pos_mvn_to_vizard(px, py, pz):
    return [py, pz, -px]

# Keep your original velocity mapping (fallback only)
def map_vec_mvn_to_vizard(vx, vy, vz):
    return [-vy, vz, vx]

# ---------------- Packet parser ----------------
def parse_mxtp02(data):
    """
    Returns (char_id, global_quats_dict, pelvis_velocity_or_None, pos_dict)
      global_quats_dict: seg -> viz.Quat (GLOBAL S_seg in Vizard coords)
      pelvis_velocity_or_None: [vx,vy,vz] in Vizard coords if present
      pos_dict: seg -> [x,y,z] in Vizard coords
    """
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
    out_pos   = {}
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

        # hemisphere continuity
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

# ---------------- UDP receive thread ----------------
def udp_loop():
    global latest_root_vel, latest_pos_global

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', UDP_PORT))
    viz.logNotice('Listening for MVN MXTP02 on {}.'.format(UDP_PORT))

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

threading.Thread(target=udp_loop, daemon=True).start()

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

        p_mvn_child  = Vector(latest_pos_global[seg])
        p_mvn_parent = Vector(latest_pos_global[parent])
        L_mvn = (p_mvn_child - p_mvn_parent).length()

        p_viz_child  = Vector(bone.getPosition(viz.ABS_GLOBAL))
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

        # 1) Root translation from pelvis POSITION (preferred)
        pelvis_pos = latest_pos_global.get('Pelvis')
        if pelvis_pos is not None:
            man.setPosition(pelvis_pos, viz.ABS_GLOBAL)
            axis.setPosition(pelvis_pos, viz.ABS_GLOBAL)
        else:
            # fallback only if pelvis pos missing
            if last_fallback_time is None:
                last_fallback_time = now
            dt = max(0.0, min(0.05, now - last_fallback_time))
            if dt > 0.0 and any(latest_root_vel):
                cx, cy, cz = man.getPosition(viz.ABS_GLOBAL)
                vx, vy, vz = latest_root_vel
                man.setPosition([cx + vx*dt, cy + vy*dt, cz + vz*dt], viz.ABS_GLOBAL)
                axis.setPosition(man.getPosition(viz.ABS_GLOBAL), viz.ABS_GLOBAL)
            last_fallback_time = now

        # 2) Avatar world orientation from pelvis GLOBAL quat
        S_pel = latest_quat_global.get('Pelvis')
        if S_pel is not None:
            man.setQuat(S_pel, viz.ABS_GLOBAL)

        # 3) Build local (joint-space) quats
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

        # 4) Apply local quats
        # pelvis local left as identity by construction
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

        yield viztask.waitTime(0.016)

viztask.schedule(update_avatar)

viz.logNotice(
    'Clean gait pipeline active: pelvis world pos/orient from streamed pelvis; '
    'children use joint-space locals. Press "c" once if you want optional bone scaling.'
)