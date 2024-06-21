import SocketWrapper from './SocketWrapper';
import {ThemeProvider} from '@mui/material/styles';
import theme from './theme';
import StreamingInterface from './StreamingInterface';
import CssBaseline from '@mui/material/CssBaseline';
import {createContext, useCallback, useState} from 'react';
import packageJson from '../package.json';

console.log(`Streaming React App version: ${packageJson?.version}`);

// Roboto font for mui ui library
// import '@fontsource/roboto/300.css';
// import '@fontsource/roboto/400.css';
// import '@fontsource/roboto/500.css';
// import '@fontsource/roboto/700.css';

export const AppResetKeyContext = createContext<(newKey: string) => void>(
  () => {
    throw new Error('AppResetKeyContext not initialized');
  },
);

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <SocketWrapper>
        <StreamingInterface />
      </SocketWrapper>
    </ThemeProvider>
  );
}

function AppWrapper() {
  const [appResetKey, setAppResetKey] = useState<string>('[initial value]');
  const setAppResetKeyHandler = useCallback((newKey: string) => {
    setAppResetKey((prev) => {
      console.warn(
        `Resetting the app with appResetKey: ${newKey}; prevKey: ${prev}`,
      );
      if (prev === newKey) {
        console.error(
          `The appResetKey was the same as the previous key, so the app will not reset.`,
        );
      }
      return newKey;
    });
  }, []);

  return (
    <AppResetKeyContext.Provider value={setAppResetKeyHandler}>
      <App key={appResetKey} />
    </AppResetKeyContext.Provider>
  );
}

export default AppWrapper;
