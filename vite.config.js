// vite.config.js
import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  base: '/',
  build: {
    outDir: 'dist',
  },
  server: {
    // host: '0.0.0.0',
    port: 9000,
    open: true,
    proxy: {
      '/api': { 
        // target: 'http://127.0.0.1:3000' ,
        target: 'http://localhost:3000', // this points to Express server
        changeOrigin: true
	},
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '/'),
    },
  },
});
