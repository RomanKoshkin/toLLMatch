/**
 * EXPERIMENTAL components to play around with but not officially use in the demo while
 * we develop.
 */
import {useEffect, useState} from 'react';
import {Object3DNode, extend} from '@react-three/fiber';
import ThreeMeshUI from 'three-mesh-ui';

import {} from '@react-three/xr';
import {Sparkles, Shadow} from '@react-three/drei';

// import FontImage from './assets/Roboto-msdf.png';
import {FontLoader} from 'three/examples/jsm/loaders/FontLoader.js';
import {TextGeometry} from 'three/examples/jsm/geometries/TextGeometry.js';
import ThreeMeshUIText from './ThreeMeshUIText';
import {ContactShadows, BakeShadows} from '@react-three/drei';

extend({TextGeometry});
extend(ThreeMeshUI);

declare module '@react-three/fiber' {
  interface ThreeElements {
    textGeometry: Object3DNode<TextGeometry, typeof TextGeometry>;
  }
}

// This is for textGeometry.. not using three-mesh-ui to display text
export function TitleMesh() {
  const font = new FontLoader().parse();
  console.log('font', font);
  const [text, setText] = useState('Text');

  useEffect(() => {
    setTimeout(() => {
      setText(text + ' more ');
      console.log('adding more tex..', text);
    }, 1000);
  }, [text]);

  return (
    <mesh>
      <textGeometry args={[text, {font, size: 5, height: 1}]} />
      <meshPhysicalMaterial attach={'material'} color={'white'} />
    </mesh>
  );
}

export function Sphere({
  size = 1,
  amount = 50,
  color = 'white',
  emissive,
  ...props
}) {
  return (
    <mesh {...props}>
      <sphereGeometry args={[size, 64, 64]} />
      <meshPhysicalMaterial
        roughness={0}
        color={color}
        emissive={emissive || color}
        envMapIntensity={0.2}
      />
      <Sparkles count={amount} scale={size * 2} size={6} speed={0.4} />
      <Shadow
        rotation={[-Math.PI / 2, 0, 0]}
        scale={size}
        position={[0, -size, 0]}
        color={emissive}
        opacity={0.5}
      />
    </mesh>
  );
}

export function Title({accentColor}) {
  return (
    <block
      args={[
        {
          width: 1,
          height: 0.25,
          backgroundOpacity: 0,
          justifyContent: 'center',
        },
      ]}>
      <ThreeMeshUIText content={'Hello '} />
      <ThreeMeshUIText content={'world!'} args={[{fontColor: accentColor}]} />
    </block>
  );
}

export function RandomComponents() {
  return (
    <>
      <color args={['#eee']} attach={'background'} />
      <Sphere
        color="white"
        amount={50}
        emissive="green"
        glow="lightgreen"
        position={[1, 1, -1]}
      />
      <Sphere
        color="white"
        amount={30}
        emissive="purple"
        glow="#ff90f0"
        size={0.5}
        position={[-1.5, 0.5, -2]}
      />
      <Sphere
        color="lightpink"
        amount={20}
        emissive="orange"
        glow="#ff9f50"
        size={0.25}
        position={[-1, 0.25, 1]}
      />
      <ContactShadows
        renderOrder={2}
        color="black"
        resolution={1024}
        frames={1}
        scale={10}
        blur={1.5}
        opacity={0.65}
        far={0.5}
      />
      <BakeShadows />
    </>
  );
}
