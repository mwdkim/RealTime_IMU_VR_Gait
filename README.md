# Real-Time IMU Screw-Theoretic Gait Analysis in Virtual Reality

This repository contains the software implementation accompanying the chapter:

**“Real-Time Integration of Optic and Haptic Flow in Virtual Reality:  
An IMU-Based Screw-Theoretic Framework for Gait Analysis and Locomotor Awareness.”**

The software implements a real-time computational pipeline that integrates:

- wearable inertial motion capture (Xsens MVN)
- UDP-based live data streaming
- quaternion-based kinematic processing
- screw-theoretic motion analysis
- immersive visualization using the Vizard virtual reality platform.

The framework enables real-time estimation of biomechanical invariants such as **instantaneous screw axis and pitch** during locomotion.

---

# System Architecture

The computational pipeline consists of four main components:

1. **Live motion capture**
   - Xsens MVN inertial motion capture system
   - wearable IMU sensors attached to body segments

2. **Network streaming**
   - MVN Network Streamer
   - UDP-based quaternion orientation transmission

3. **Kinematic processing**
   - quaternion parsing
   - angular velocity estimation
   - computation of relative segment twists

4. **Screw-theoretic analysis and visualization**
   - instantaneous screw axis estimation
   - pitch computation
   - avatar animation in Vizard VR environment

---

# Repository Structure
