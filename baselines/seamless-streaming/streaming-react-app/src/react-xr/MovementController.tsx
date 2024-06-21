import {useRef} from 'react';
import {useFrame} from '@react-three/fiber';
import {useController, useXR} from '@react-three/xr';
import * as THREE from 'three';

const USE_HORIZONTAL = true;
const USE_VERTICAL = true;
const USE_ROTATION = true;
const HORIZONTAL_AXIS = 2;
const VERTICAL_AXIS = 3;
const ROTATION_AXIS = 2;
const SENSITIVITY = 0.05;
const DEADZONE = 0.05;

/**
 * Component to add into the ThreeJS canvas that reads controller (Quest) inputs to change camera position
 */
export default function MovementController() {
  const xr = useXR();
  const controller = useController('right');
  const forward = useRef(new THREE.Vector3());
  const horizontal = useRef(new THREE.Vector3());

  useFrame(() => {
    const player = xr.player;
    const camera = xr.player.children[0];
    const cameraMatrix = camera.matrixWorld.elements;
    forward.current
      .set(-cameraMatrix[8], -cameraMatrix[9], -cameraMatrix[10])
      .normalize();

    const axes = controller?.inputSource?.gamepad?.axes ?? [0, 0, 0, 0];

    if (USE_HORIZONTAL) {
      horizontal.current.copy(forward.current);
      horizontal.current.cross(camera.up).normalize();

      player.position.add(
        horizontal.current.multiplyScalar(
          (Math.abs(axes[HORIZONTAL_AXIS]) > DEADZONE
            ? axes[HORIZONTAL_AXIS]
            : 0) * SENSITIVITY,
        ),
      );
    }

    if (USE_VERTICAL) {
      player.position.add(
        forward.current.multiplyScalar(
          (Math.abs(axes[VERTICAL_AXIS]) > DEADZONE ? axes[VERTICAL_AXIS] : 0) *
            SENSITIVITY,
        ),
      );
    }

    if (USE_ROTATION) {
      player.rotation.y -=
        (Math.abs(axes[ROTATION_AXIS]) > DEADZONE ? axes[ROTATION_AXIS] : 0) *
        SENSITIVITY;
    }
  });

  return <></>;
}
