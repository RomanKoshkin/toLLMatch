import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig(({ command }) => {
  let define = {};
  if (command === 'serve') {
    define = {
      global: {},
    };
  }
  return {
    plugins: [react()],
    define: define,
    server: {
      proxy: {
        '/ws': {
          target: 'ws://localhost:7860',
          ws: true
        }
      },
    },
  }
});
