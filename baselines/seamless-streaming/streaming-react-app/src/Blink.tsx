import Box from '@mui/material/Box';
import {useEffect, useState} from 'react';

type Props = {
  intervalMs: number;
  children: React.ReactNode;
  shouldBlink: boolean;
  // display?: 'block' | 'inline' | 'inline-block';
};

export default function Blink({
  // display = 'inline-block',
  shouldBlink,
  intervalMs,
  children,
}: Props): React.ReactElement {
  const [cursorBlinkOn, setCursorBlinkOn] = useState(false);

  useEffect(() => {
    if (shouldBlink) {
      const interval = setInterval(() => {
        setCursorBlinkOn((prev) => !prev);
      }, intervalMs);

      return () => clearInterval(interval);
    } else {
      setCursorBlinkOn(false);
    }
  }, [intervalMs, shouldBlink]);

  return (
    <Box
      component="span"
      sx={{
        display: 'inline-block',
        visibility: cursorBlinkOn ? 'visible' : 'hidden',
      }}>
      {children}
    </Box>
  );
}
