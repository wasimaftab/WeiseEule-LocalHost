// vite.config.js
import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  base: '/',
  build: {
    outDir: 'dist',
  },
  server: {
    port: 9000,
    // port: 9001,

    open: true,
    proxy: {
      '/api': { target: 'http://localhost:3000' },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '/'),
    },
  },
});