import {useRef, useEffect} from 'react';
import * as THREE from 'three';
import {extend} from '@react-three/fiber';
import ThreeMeshUI from 'three-mesh-ui';
import ThreeMeshUIText, {ThreeMeshUITextType} from './ThreeMeshUIText';
import {Interactive} from '@react-three/xr';

/**
 * Using `?url` at the end of this import tells vite this is a static asset, and
 * provides us a URL to the hashed version of the file when the project is built.
 * See: https://vitejs.dev/guide/assets.html#explicit-url-imports
 */
import robotoFontFamilyJson from '../assets/RobotoMono-Regular-msdf.json?url';
import robotoFontTexture from '../assets/RobotoMono-Regular.png';

extend(ThreeMeshUI);

/**
 * Button component that renders as a three-mesh-ui block
 */
export default function Button({
  onClick,
  content,
  width,
  height,
  fontSize,
  borderRadius,
  padding,
}) {
  const button = useRef<JSX.IntrinsicElements['block']>();
  const textRef = useRef<ThreeMeshUITextType>();

  useEffect(() => {
    if (textRef.current != null) {
      textRef.current.set({content});
    }
  }, [textRef, content]);

  useEffect(() => {
    if (!button.current) {
      return;
    }
    button.current.setupState({
      state: 'hovered',
      attributes: {
        offset: 0.002,
        backgroundColor: new THREE.Color(0x607b8f),
        fontColor: new THREE.Color(0xffffff),
      },
    });
    button.current.setupState({
      state: 'idle',
      attributes: {
        offset: 0.001,
        backgroundColor: new THREE.Color(0x465a69),
        fontColor: new THREE.Color(0xffffff),
      },
    });
    button.current.setupState({
      state: 'selected',
      attributes: {
        offset: 0.005,
        backgroundColor: new THREE.Color(0x000000),
        fontColor: new THREE.Color(0xffffff),
      },
    });
    button.current.setState('idle');
  }, []);

  const args = [
    {
      width,
      height,
      fontSize,
      padding,
      justifyContent: 'end',
      textAlign: 'center',
      alignItems: 'center',
      borderRadius,
      fontFamily: robotoFontFamilyJson,
      fontTexture: robotoFontTexture,
      backgroundOpacity: 1,
      backgroundColor: new THREE.Color(0x779092),
      fontColor: new THREE.Color(0x000000),
    },
  ];

  return (
    <Interactive
      // These are for XR mode
      onSelect={() => {
        onClick();
      }}
      onHover={() => button.current.setState('hovered')}
      onBlur={() => button.current.setState('idle')}
      onSelectStart={() => button.current.setState('selected')}
      onSelectEnd={() => button.current.setState('idle')}>
      <block
        // These are for non-XR modes
        onPointerEnter={() => button.current.setState('hovered')}
        onPointerLeave={() => button.current.setState('idle')}
        onPointerDown={() => button.current.setState('selected')}
        onPointerUp={() => {
          button.current.setState('hovered');
          onClick();
        }}>
        <block args={args} ref={button}>
          <ThreeMeshUIText
            ref={textRef}
            fontColor={new THREE.Color(0xffffff)}
            content={content}
          />
        </block>
      </block>
    </Interactive>
  );
}
