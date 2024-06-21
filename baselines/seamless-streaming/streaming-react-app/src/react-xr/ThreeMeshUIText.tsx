import {extend} from '@react-three/fiber';
import {forwardRef} from 'react';
import ThreeMeshUI, {TextOptions} from 'three-mesh-ui';

extend(ThreeMeshUI);

/**
 * Hacky but component that wraps <text> since this has typescript issues because it collides with
 * the native <text> SVG element. Simple enough so abstracting it away in this file
 * so it could be used in other places with low risk. e.g:
 * <ThreeMeshUIText content="Hello" />
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ThreeMeshUITextType = any;

const ThreeMeshUIText = forwardRef<ThreeMeshUITextType, TextOptions>(
  function ThreeMeshUIText(props, ref) {
    return <text {...props} ref={ref} />;
  },
);

export default ThreeMeshUIText;
